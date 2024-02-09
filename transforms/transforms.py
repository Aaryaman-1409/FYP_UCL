import numpy as np


# all transforms assume angles are in radians

def euler_to_quaternion(euler_angles):
    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_euler(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    x1 = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    y1 = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    z1 = np.arctan2(t3, t4)
    return np.array([x1, y1, z1])


def quaternion_to_rot3(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot3 = np.array([[1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
                     [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
                     [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)]])
    return rot3


def rot3_to_quaternion(rot3):
    trace = rot3[0, 0] + rot3[1, 1] + rot3[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot3[2, 1] - rot3[1, 2]) * s
        y = (rot3[0, 2] - rot3[2, 0]) * s
        z = (rot3[1, 0] - rot3[0, 1]) * s
    else:
        if rot3[0, 0] > rot3[1, 1] and rot3[0, 0] > rot3[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot3[0, 0] - rot3[1, 1] - rot3[2, 2])
            w = (rot3[2, 1] - rot3[1, 2]) / s
            x = 0.25 * s
            y = (rot3[0, 1] + rot3[1, 0]) / s
            z = (rot3[0, 2] + rot3[2, 0]) / s
        elif rot3[1, 1] > rot3[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot3[1, 1] - rot3[0, 0] - rot3[2, 2])
            w = (rot3[0, 2] - rot3[2, 0]) / s
            x = (rot3[0, 1] + rot3[1, 0]) / s
            y = 0.25 * s
            z = (rot3[1, 2] + rot3[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot3[2, 2] - rot3[0, 0] - rot3[1, 1])
            w = (rot3[1, 0] - rot3[0, 1]) / s
            x = (rot3[0, 2] + rot3[2, 0]) / s
            y = (rot3[1, 2] + rot3[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_to_6d(quaternion):
    rot3 = quaternion_to_rot3(quaternion)
    _6d = rot3[..., :2, :]
    _6d = _6d.reshape(_6d.shape[:-2] + (6,))
    return _6d


def _6d_to_quaternion(_6d):
    _6d = _6d.reshape(_6d.shape[:-1] + (2, 3))
    rot3 = np.concatenate([_6d, np.cross(_6d[..., :1, :], _6d[..., 1:, :])], axis=-2)
    return rot3_to_quaternion(rot3)
