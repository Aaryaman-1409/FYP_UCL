import numpy as np

from bvh.bvh import Bvh


class BvhLoader:
    def __init__(self, filename):
        self.bvh = self.load_bvh(filename)
        self.joint_names = self.get_joint_names()

        self.parent_indices = self.get_parent_indices()
        self.offsets = self.get_offsets()
        self.joint_channels = self.get_joint_channels()

        self.rotations = self.get_rotations(canonical=True)
        self.positions = self.get_positions()

    @staticmethod
    def load_bvh(filename):
        with open(filename, 'r') as f:
            return Bvh(f.read())

    def get_joint_names(self):
        return self.bvh.get_joints_names()

    def get_parent_indices(self):
        return [self.bvh.joint_parent_index(joint) for joint in self.joint_names]

    def get_offsets(self):
        self.offsets = np.empty((0, 3))
        for joint in self.joint_names:
            self.offsets = np.vstack((self.offsets, self.bvh.joint_offset(joint)))
        return self.offsets

    def get_joint_channels(self):
        # assumes all joints have same channels
        return self.bvh.joint_channels(self.joint_names[0])

    def get_position(self, frame_index):
        positions = np.empty((0, 3))
        for joint in self.joint_names:
            positions = np.vstack(
                (positions, self.bvh.frame_joint_channels(frame_index, joint, self.joint_channels[:3])))
        return positions

    def get_rotation(self, frame_index, canonical=False):
        # canonical: if True, return rotations in xyz order

        rotations = np.empty((0, 3))
        for joint in self.joint_names:
            rotations = np.vstack(
                (rotations, self.bvh.frame_joint_channels(frame_index, joint, self.joint_channels[3:])))
        return rotations[:, [2, 1, 0] if canonical else [0, 1, 2]]

    def get_positions(self):
        positions = np.empty((0, len(self.joint_names), 3))
        for frame_index in range(self.bvh.nframes):
            positions = np.append(positions, np.expand_dims(self.get_position(frame_index), 0), axis=0)
        return positions

    def get_rotations(self, canonical=False):
        rotations = np.empty((0, len(self.joint_names), 3))
        for frame_index in range(self.bvh.nframes):
            rotations = np.append(rotations, np.expand_dims(self.get_rotation(frame_index, canonical), 0), axis=0)
        return rotations
