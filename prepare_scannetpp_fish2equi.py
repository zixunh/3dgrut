
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


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
    camera_dir = Path(root_dir) / "colmap" / "cameras.txt"
    camera_dir_new = Path(root_dir) / "colmap" / "cameras_equidist.txt"
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    _, _, width, height, params = read_intrinsics_text(camera_dir)
    print(params)
    
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    
    distortion_params = params[4:]
    kk = distortion_params
    
    width_org = np.copy(width)
    height_org =  np.copy(height)
    fx_tgt = fx * 0.85
    fy_tgt = fy * 0.85
    
    # write modified camera intrinsics in file
    with open(camera_dir_new, 'w') as f:
        f.write(f"# Camera list with one line of data per camera:\n")
        f.write(f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"{1} OPENCV_FISHEYE {width} {height} {fx_tgt} {fy_tgt} {width//2} {height//2} {kk[0]} {kk[1]} {kk[2]} {kk[3]}\n")
    
    mapx = np.zeros((width, height), dtype=np.float32)
    mapy = np.zeros((width, height), dtype=np.float32)
    
    for i in tqdm(range(0, width), desc="calculate_maps"):
        for j in range(0, height):
            x = float(i)
            y = float(j)
            x1 = (x - width // 2) / fx_tgt
            y1 = (y - height // 2) / fy_tgt
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
            # borderMode=cv2.BORDER_REFLECT_101,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        out_image_path = Path(out_image_dir) / frame.replace(".JPG", ".png") # rgba can not use .jpg
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # also generate valid region mask
        mask = np.ones_like(undistorted_image[:, :, 0], dtype=np.uint8)*255
        mask[mapx.T < 0] = 0
        mask[mapx.T >= width_org] = 0
        mask[mapy.T < 0] = 0
        mask[mapy.T >= height_org] = 0
        
        # assign the alpha channel
        undistorted_image = np.concatenate([undistorted_image, mask[:, :, None]], axis=2)
        cv2.imwrite(str(out_image_path), undistorted_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/mnt/data_ssd_4tb/Datasets/scannetpp_tiny/data/0a5c013435/dslr")
    parser.add_argument('--src', type=str, default="resized_images")
    parser.add_argument('--dst', type=str, default="images_equidist")
    args = parser.parse_args()
    colmap_main(args)

