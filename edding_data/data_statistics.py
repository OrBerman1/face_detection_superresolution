import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def histogram(ls, title):
    plt.hist(ls, bins=50)
    plt.title(title)
    plt.show()


def math_statistics(ls, name, upper=None, lower=None):
    arr = np.array(ls)
    if upper:
        arr = arr[arr<upper]
    if lower:
        arr = arr[arr>lower]
    print(len(arr))
    std = np.std(arr)
    mn = np.mean(arr)
    print(f"std of {name} is: {std}")
    print(f"mean of {name} is: {mn}")


def height_and_width_statistics(folder_path):
    images_paths = os.listdir(folder_path)
    widths, heights = [], []
    for path in images_paths:
        img = Image.open(f"{folder_path}/{path}")
        width, height = img.size
        widths.append(width)
        heights.append(height)
    histogram(widths, "width")
    histogram(heights, "height")
    math_statistics(widths, "width", 300)
    math_statistics(heights, "height", 300)




height_and_width_statistics("/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged/merged_even_data_train")


