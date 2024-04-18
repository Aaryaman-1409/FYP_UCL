import numpy as np
import torch
from torch import nn


def cosine_noise_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.num_timesteps = timesteps
        self.model = model
        self.betas = cosine_noise_schedule(timesteps)

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = self.to_torch(np.append(self.alphas_cumprod[1:], 0.0))
        self.sqrt_alphas_cumprod = self.to_torch(np.sqrt(self.alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = self.to_torch(np.sqrt(1.0 - self.alphas_cumprod))
        self.log_one_minus_alphas_cumprod = self.to_torch(np.log(1.0 - self.alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = self.to_torch(np.sqrt(1.0 / self.alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = self.to_torch(np.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = self.to_torch(np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ))
        self.posterior_mean_coef1 = self.to_torch((
                self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ))
        self.posterior_mean_coef2 = self.to_torch((
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        ))

        self.alphas_cumprod = self.to_torch(self.alphas_cumprod)
        self.alphas_cumprod_prev = self.to_torch(self.alphas_cumprod_prev)
        self.posterior_variance = self.to_torch(self.posterior_variance)

    @staticmethod
    def to_torch(arr, device='cuda'):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.sqrt_alphas_cumprod[t] * x_start + self.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_mean_variance(self, x, t, clip_denoised):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device='cuda')

        x_recon = self.model(x, t_tensor)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # no noise when t == 0
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, batch_size, x_shape):
        sample_shape = (batch_size, *x_shape)

        timesteps = self.num_timesteps
        res = torch.randn(sample_shape, device='cuda')
        for t in reversed(range(0, timesteps)):
            res = self.p_sample(res, t)
        return res

    @staticmethod
    def kl(mean1, logvar1, mean2, logvar2):
        kl = 0.5 * (logvar2 - logvar1) - 0.5 + (torch.exp(logvar1) + (mean1 - mean2) ** 2) / (2 * torch.exp(logvar2)) - 0.5
        return kl

    @staticmethod
    def discrete_gaussian_log_likelihood(x, means, log_scales):
        log_scales = torch.clamp(log_scales, min=1e-12)
        inv_stdv = torch.exp(-log_scales)
        return -0.5 * ((x - means) * inv_stdv) ** 2 - log_scales - 0.5 * np.log(2 * np.pi)

    def compute_loss(self, x_start, x_t, t, clip_denoised=True):
        real_mean, real_variance, real_log_variance = self.q_posterior(x_start, x_t, t)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, t, clip_denoised)
        kl = self.kl(real_mean, real_log_variance, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)
        decoder_nll = -self.discrete_gaussian_log_likelihood(x_t, model_mean, 0.5 * model_log_variance)
        decoder_nll = decoder_nll.mean(dim=list(range(1, len(x_t.shape)))) / np.log(2.0)
        output = torch.where((t == 0), decoder_nll, kl)
        return output

    def forward(self, x):
        batch_size = x.shape[0]
        t = np.random.randint(0, self.num_timesteps)
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device='cuda')
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x0_recon = self.model(x_noisy, t_tensor)
        return self.compute_loss(x, x0_recon, t)
