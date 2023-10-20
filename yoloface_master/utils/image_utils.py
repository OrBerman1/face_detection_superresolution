import os

import cv2
from PIL import ImageDraw, Image
import numpy as np


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


def read_images_from_dir(dir_path):
    images_names = os.listdir(dir_path)
    images_ls = []
    for image in images_names:
        images_ls.append(Image.open(f"{dir_path}/{image}"))
    return images_ls, images_names


def read_image(image_path):
    return [Image.open(f"{image_path}")], [os.path.basename(image_path)]


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


def video_to_images_for_detection(video):
    new_video = []
    for frame in video:
        frame = np.array(pad_image_square(frame))
        if frame.shape[0] > 3 and frame.shape[-1] > 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
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


def get_bbs_size_of_images(images):
    bbs = []
    images = video_to_images_for_detection(images)
    for img in images:
        width, height = img.size
        bbs.append([0, 0, width-1, height-1])
    return images, bbs


def draw_bbs_on_image(img, bbs: list):
    image_to_draw = ImageDraw.Draw(img)
    for bb in bbs:
        start = (bb[0], bb[1])
        end = (bb[2], bb[3])
        image_to_draw.rectangle([start, end], outline="red")
    return img


def sharp_edges(img):
    img = np.array(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened
