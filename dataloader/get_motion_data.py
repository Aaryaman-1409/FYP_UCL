import numpy as np
import torch

from bvh.load import BvhLoader
from transforms import transforms


def get_motion_data(filename, motion_repr='quaternion'):
    motion_data = None
    bvh = BvhLoader(filename)
    positions, rotations = bvh.positions, np.radians(bvh.rotations)

    if motion_repr == 'quaternion':
        frames, joints = rotations.shape[:2]
        quat_rotations = np.zeros((frames, joints, 4))

        for i in range(frames):
            for j in range(joints):
                quat_rotations[i, j] = transforms.euler_to_quaternion(rotations[i, j])

        motion_data = np.concatenate([positions, quat_rotations], axis=2)

    elif motion_repr == '6d':
        frames, joints = rotations.shape[:2]
        repr_6d_rotations = np.zeros((frames, joints, 6))

        for i in range(frames):
            for j in range(joints):
                repr_6d_rotations[i, j] = transforms.quaternion_to_6d(transforms.euler_to_quaternion(rotations[i, j]))

        motion_data = np.concatenate([positions, repr_6d_rotations], axis=2)

    motion_data = torch.from_numpy(motion_data).permute(1, 2, 0).to(torch.float32)
    motion_data.to(torch.float32)
    return motion_data
