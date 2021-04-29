import cv2
import os
import pandas as pd
import time

def map_meta_csv_and_img(df, images_dir="archive"):
    """
    Given the HAM10000_metadata.csv, maps the corresponding image array to the row
    Also scale and normalize images. Neural Networks need 0-1 normalized data to converge. 
    The images have been scaled down by a factor of 8 since it is heavy holding so much information.
    This factor can be played around with, scaling less may result in a better model but is computationally
    more expensive.
    """
    images = []
    for i in df['image_id']:
        image_path = os.path.join(f"{images_dir}/HAM10000_images_part_1", f"{i}.jpg")
        if os.path.isfile(image_path):
            img = cv2.imread(image_path)
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            RGB_img = resize(RGB_img, (RGB_img.shape[0] // 8, RGB_img.shape[1] // 8), anti_aliasing=True)
            # scale down
            images.append(RGB_img)
        else:
            image_path = os.path.join(f"{images_dir}/HAM10000_images_part_2", f"{i}.jpg")
            # scale down
            if os.path.isfile(image_path):
                img = cv2.imread(image_path)
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                RGB_img = resize(RGB_img, (RGB_img.shape[0] // 8, RGB_img.shape[1] // 8), anti_aliasing=True)
                images.append(RGB_img)
            else:
                print("image not found")
                images.append([])
    df["images"] = images
    return df