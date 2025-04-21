import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from threedgrut.datasets.utils import read_colmap_intrinsics_binary


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
    return camera_id, model, width, height, params


def colmap_main(args):
    if args.cross_camera:
        if 'fisheye' in args.camera_path:
            args.camera_path = args.camera_path.replace('fisheye', 'undistorted')
        else:
            args.camera_path = args.camera_path.replace('undistorted', 'fisheye')
    camera_dir = Path(args.camera_path)
    input_image_dir = Path(args.src)
    out_image_dir = Path(args.dst)
    
    cam_intrinsics = read_colmap_intrinsics_binary(camera_dir)
    width = cam_intrinsics[0].width
    height = cam_intrinsics[0].height
    params = cam_intrinsics[0].params
    print(params)
    
    # adjust fx, fy, cx, cy by the actual image size
    if args.r == -1:
        ratio = 1 / float(input_image_dir.name[7])
    else:
        ratio = 1 / args.r
    
    fx = params[0] * ratio
    fy = params[1] * ratio
    cx = params[2] * ratio
    cy = params[3] * ratio
    width = int(width * ratio)
    height = int(height * ratio)
    
    distortion_params = params[4:]
    kk = distortion_params
    
    # Use prepared fisheye grid map by DAC https://github.com/yuliangguo/depth_any_camera
    grid_map_file = Path(args.camera_path).parent / "grid_fisheye.npy"
    grid_fisheye = np.load(grid_map_file)
    grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (width, height), interpolation=cv2.INTER_NEAREST)
    grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (width, height))
    grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
    
    # Reverse warping
    reverse_mapx = np.zeros((width, height), dtype=np.float32)
    reverse_mapy = np.zeros((width, height), dtype=np.float32)
    # More exact reverse warping using grid_fisheye
    for i in tqdm(range(0, width), desc="calculate_reverse_maps"):
        for j in range(0, height):
            X_c = grid_fisheye[j, i, 0]
            Y_c = grid_fisheye[j, i, 1]
            Z_c = grid_fisheye[j, i, 2]
            X_c = X_c / (Z_c + 1e-9)
            Y_c = Y_c / (Z_c + 1e-9)
            
            r = np.sqrt(X_c**2 + Y_c**2)
            theta = np.arctan(r)
            
            x2 = fx * X_c * theta / r + width // 2
            y2 = fy * Y_c * theta/ r + height // 2
            reverse_mapx[i, j] = x2
            reverse_mapy[i, j] = y2
    
    # # TODO: this remapping not exact, because in undistortion, r was calculated from different domain's theta
    # for i in tqdm(range(0, width), desc="calculate_reverse_maps"):
    #     for j in range(0, height):
    #         x = float(i)
    #         y = float(j)
    #         x1 = (x - cx) / fx
    #         y1 = (y - cy) / fy
    #         # Inaccurcy from not easy to theta from theta_d computed from source KB space
    #         theta_d = np.sqrt(x1**2 + y1**2)
    #         r = (1.0 + kk[0] * theta_d**2 + kk[1] * theta_d**4 + kk[2] * theta_d**6 + kk[3] * theta_d**8)
    #         x2 = fx * x1 / r + width // 2
    #         y2 = fy * y1 / r + height // 2
    #         reverse_mapx[i, j] = x2
    #         reverse_mapy[i, j] = y2
    
    frames = os.listdir(input_image_dir)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        undistorted_image = cv2.imread(str(image_path))
        
        reversed_image = cv2.remap(
            undistorted_image,
            reverse_mapx.T,
            reverse_mapy.T,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        reversed_image_path = Path(out_image_dir) / frame
        reversed_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reversed_image_path), reversed_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--camera-path', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/fisheye/berlin/sparse/0/cameras.bin")
    parser.add_argument('--src', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/fisheye/berlin/images_8_undistorted_fisheye")
    parser.add_argument('--dst', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/fisheye/berlin/images_8_reproduced_fisheye")
    parser.add_argument('-r', type=int, default=-1)
    parser.add_argument('--cross_camera', action='store_true')
    args = parser.parse_args()
    colmap_main(args)
