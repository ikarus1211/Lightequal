from LightEqual import equalize
import os
import cv2 as cv
import numpy as np

def load_examples(path):

    file_names = os.listdir(path)
    images = {}

    for name in file_names:

        sp_name = name.split('.')[0].split('_')

        if int(sp_name[1]) in images.keys():
            continue
        mask = cv.imread(os.path.join(path, f"mask_{sp_name[1]}.jpg"))
        img = cv.imread(os.path.join(path, f"img_{sp_name[1]}.jpg"))
        images[int(sp_name[1])] = (img, mask)

    return images

if __name__ == "__main__":

    # Select example1, example2, ....
    path = "./imgs/example1"
    # Load imgs in format {name: (img, mask)}
    images = load_examples(path)
    # Equalize light
    adjusted = equalize(images)

    for key, value in adjusted.items():
        cv.imwrite(f"./plots/original_img_{key}.jpg", images[key][0])
        cv.imwrite(f"./plots/adjusted_img_{key}.jpg", value[0])

        cv.imwrite(f"./plots/diff_img{key}.jpg", np.abs(value[0] - images[key][0]))
