import os

import cv2
from PIL import ImageDraw, Image
import numpy as np
import math


def read_video(video_path):
    vidObj = cv2.VideoCapture(video_path)
    success = 1
    frames = []
    while success:
        success, image = vidObj.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image))
    return frames


def read_image(image_path):
    return [Image.open(f"{image_path}")]


def read_images_from_dir(dir_path):
    images_names = os.listdir(dir_path)
    images_ls = []
    for image in images_names:
        images_ls.append(Image.open(f"{dir_path}/{image}"))
    return images_ls


def save_images_to_dir(image_ls, dir_path="bb_examples"):
    os.makedirs(dir_path, exist_ok=True)
    for i, image in enumerate(image_ls):
        image.save(f"{dir_path}/{i+1}.png")


def resize_image(img, size: int):
    return cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_LINEAR)


def pad_image_square(img: Image):
    width, height = img.size
    if width > height:
        result = Image.new(img.mode, (width, width), (0, 0, 0))
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), (0, 0, 0))
        result.paste(img, ((height - width) // 2, 0))
        return result


def resize_image_to_power_of_2(img: Image):
    def nearestPowerOf2(N):
        # Calculate log2 of N
        a = int(math.log2(N))

        # If 2^a is equal to N, return N
        if 2 ** a == N:
            return N

        # Return 2^(a + 1)
        return 2 ** (a + 1)
    img = pad_image_square(img)
    width, height = img.size
    closest = nearestPowerOf2(width)
    return img.resize((closest, closest), Image.BILINEAR)


def resize_image_even(img: Image):
    width, height = img.size
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    return img.resize((width, height), Image.BILINEAR)


def video_to_images_for_detection(video):
    new_video = []
    for frame in video:
        frame = np.array(pad_image_square(frame))
        new_video.append(frame)
    return new_video


def draw_bbs_on_video(video, bboxes):
    draw_frames = []
    for frame, bbs in zip(video, bboxes):
        if type(frame) != Image:
            frame = Image.fromarray(frame)
        draw_frame = draw_bbs_on_image(frame, bbs)
        draw_frames.append(draw_frame)
    return draw_frames


def draw_bbs_on_image(img, bbs: list):
    image_to_draw = ImageDraw.Draw(img)
    for bb in bbs:
        start = (bb[0], bb[1])
        end = (bb[2], bb[3])
        image_to_draw.rectangle([start, end], outline="red")
    return img
