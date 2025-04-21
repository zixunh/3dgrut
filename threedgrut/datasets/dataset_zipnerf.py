# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from .dataset_colmap import ColmapDataset
from .utils import read_colmap_extrinsics_binary, read_colmap_intrinsics_binary


# class ZipnerfDataset(ColmapDataset):

#     def __init__(self, path, device="cuda", split="train", ray_jitter=None):
#         super(ZipnerfDataset, self).__init__(path, device, split, ray_jitter)

#     def load_intrinsics_and_extrinsics(self):
#         cameras_extrinsic_file = os.path.join(self.path, "sparse", "0", "images.bin")
#         cameras_intrinsic_file = os.path.join(self.path, "sparse", "0", "cameras.bin")
#         self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
#         self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)

#         # Remove camera distortions because images are already undistorted
#         for intr in self.cam_intrinsics:
#             intr.params[4:] = 0.0

#     def get_images_folder(self):
#         return "images"

class ZipnerfFisheyeDataset(ColmapDataset):

    def __init__(self, path, device="cuda", split="train", downsample_factor=1, ray_jitter=None):
        super(ZipnerfFisheyeDataset, self).__init__(path, device, split, downsample_factor, ray_jitter)

    def load_intrinsics_and_extrinsics(self):
        cameras_extrinsic_file = os.path.join(self.path, "sparse", "0", "images.bin")
        cameras_intrinsic_file = os.path.join(self.path, "sparse", "0", "cameras.bin")
        self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
        self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)

        # Remove camera distortions because images are already undistorted
        for intr in self.cam_intrinsics:
            intr.params[4:] = 0.0

    def get_images_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        # return "image_undistorted_fisheye"
        return f"images{downsample_suffix}_equidist"