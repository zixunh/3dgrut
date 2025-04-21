
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
    root_dir = args.path
    camera_dir = Path(root_dir) / "sparse" / "0" / "cameras.bin"
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    cam_intrinsics = read_colmap_intrinsics_binary(camera_dir)
    width = cam_intrinsics[1].width
    height = cam_intrinsics[1].height
    params = cam_intrinsics[1].params
    print(params)
    
    # adjust fx, fy, cx, cy by the actual image size
    ratio = 1 / float(input_image_dir.name[7])
    
    fx = params[0] * ratio
    fy = params[1] * ratio
    cx = params[2] * ratio
    cy = params[3] * ratio
    width = int(width * ratio)
    height = int(height * ratio)
    
    distortion_params = params[4:]
    kk = distortion_params
    
    mapx = np.zeros((width, height), dtype=np.float32)
    mapy = np.zeros((width, height), dtype=np.float32)
    
    for i in tqdm(range(0, width), desc="calculate_maps"):
        for j in range(0, height):
            x = float(i)
            y = float(j)
            x1 = (x - width // 2) / fx
            y1 = (y - height // 2) / fy
            # From source space equidistant, original theta already in pixel space. Refer to paper eq. (6) to derive theta
            theta = np.sqrt(x1**2 + y1**2)
            r = (1.0 + kk[0] * theta**2 + kk[1] * theta**4 + kk[2] * theta**6 + kk[3] * theta**8)
            x2 = fx * x1 * r + cx
            y2 = fy * y1 * r + cy
            mapx[i, j] = x2
            mapy[i, j] = y2
    
    frames = os.listdir(input_image_dir)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            mapx.T,
            mapy.T,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        out_image_path = Path(out_image_dir) / frame.replace(".JPG", ".png") # rgba can not use .jpg
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # generate valid region mask
        mask = np.ones_like(undistorted_image[:, :, 0], dtype=np.uint8)*255
        mask[mapx.T < 0] = 0
        mask[mapx.T >= width] = 0
        mask[mapy.T < 0] = 0
        mask[mapy.T >= height] = 0
        
        # assign the alpha channel
        undistorted_image = np.concatenate([undistorted_image, mask[:, :, None]], axis=2)
        cv2.imwrite(str(out_image_path), undistorted_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/fisheye/berlin/")
    parser.add_argument('--src', type=str, default="images_4")
    parser.add_argument('--dst', type=str, default="images_4_equidist")
    args = parser.parse_args()
    colmap_main(args)

