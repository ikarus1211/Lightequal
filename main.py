from LightEqual import equalize
import os
import cv2 as cv
import numpy as np
import argparse
import logging
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Fixes lighting in precisely aligned images (photos of a near-planar object).")
    parser.add_argument('--input-path', type=str, default="./imgs/example1", help='Path to the images directory.')
    parser.add_argument('--output-path', type=str, default="./plots", help='Path to save the adjusted images.')
    parser.add_argument('--grid-size', type=int, default=64, help='Size of the grid for light adjustment.')
    parser.add_argument('--mode', type=str, default="scale", choices=["scale", "affine"], help='Mode of light adjustment.')
    parser.add_argument('--scale-factor', type=float, default=1.0, help='Scale factor for light adjustment.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    return args


def load_examples(path, scale_factor=1.0):
    file_names = os.listdir(path)
    images = {}

    for name in tqdm(file_names):

        sp_name = name.split('.')[0].split('_')

        if int(sp_name[1]) in images.keys():
            continue
        mask = cv.imread(os.path.join(path, f"mask_{sp_name[1]}.jpg"))
        if mask is None:
            logging.warning(f"Mask not found for {name}. Skipping.")
            exit(-1)
        img = cv.imread(os.path.join(path, f"img_{sp_name[1]}.jpg"))
        if img is None:
            logging.warning(f"Image not found for {name}. Skipping.")
            exit(-1)

        if scale_factor != 1.0:
            img = cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
            mask = cv.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
        images[int(sp_name[1])] = (img, mask)

    return images


def main():
    args = parse_args()

    # Select example1, example2, ....
    # Load imgs in format {name: (img, mask)}
    logging.info(f"Loading images from {args.input_path}")
    images = load_examples(args.input_path, scale_factor=args.scale_factor)

    # Equalize light
    logging.info(f"Equalizing light in {len(images)} images")
    adjusted = equalize(images, grid_size=args.grid_size, mode=args.mode)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    logging.info(f"Saving adjusted images to {args.output_path}")
    for key, value in tqdm(adjusted.items()):
        cv.imwrite(os.path.join(args.output_path, f"{key}_adjusted_img.jpg"), value[0])
        cv.imwrite(os.path.join(args.output_path, f"mask_img_{key}.jpg"), value[1])
        cv.imwrite(os.path.join(args.output_path, f"{key}_img.jpg"), images[key][0])


if __name__ == "__main__":
    main()
