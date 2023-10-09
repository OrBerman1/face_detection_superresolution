import os

import cv2
from PIL import ImageDraw, Image
import numpy as np


def read_video(video_path):
    """
    read video using path to video
    @param video_path: path to video to read
    @return: list of PIL images of the video's frames
    """
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
    """
    read all the images in a dir. Note, the dir most contain only images
    @param dir_path: path to directory of images
    @return: list of PIL images
    """
    images_names = os.listdir(dir_path)
    images_ls = []
    for image in images_names:
        images_ls.append(Image.open(f"{dir_path}/{image}"))
    return images_ls


def save_images_to_dir(image_ls, dir_path="bb_examples"):
    """
    save a list of images
    @param image_ls: list of images to save
    @param dir_path: a path to a directory for the images
    """
    os.makedirs(dir_path, exist_ok=True)
    for i, image in enumerate(image_ls):
        image.save(f"{dir_path}/{i+1}.png")


def resize_image(img, size: int):
    """
    reshape image to a square by size
    @param img: the image
    @param size: the size of the square
    @return: resized image
    """
    return cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_LINEAR)


def pad_image_square(img: Image):
    """
    pad image with zero to a square shape according to it's dimensions
    @param img: image
    @return: padded squared image
    """
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
    """
    convert a list of images to format suited for the detector model
    @param video: a list of images
    @return: list of formatted images
    """
    new_video = []
    for frame in video:
        frame = np.array(pad_image_square(frame))
        new_video.append(frame)
    return new_video


def draw_bbs_on_video(video, bboxes):
    """
    drqw bounding boxes on the images
    @param video: a list of images
    @param bboxes: list of all the bounding boxes of each image
    @return: images with drawn bounding boxes
    """
    draw_frames = []
    for frame, bbs in zip(video, bboxes):
        if type(frame) != Image:
            frame = Image.fromarray(frame)
        draw_frame = draw_bbs_on_image(frame, bbs)
        draw_frames.append(draw_frame)
    return draw_frames


def draw_bbs_on_image(img, bbs: list):
    """
    draw bounding boxes on single image
    @param img: image
    @param bbs: list of image bounding boxes
    @return: image with drawn bounding boxes
    """
    image_to_draw = ImageDraw.Draw(img)
    for bb in bbs:
        start = (bb[0], bb[1])
        end = (bb[2], bb[3])
        image_to_draw.rectangle([start, end], outline="red")
    return img
