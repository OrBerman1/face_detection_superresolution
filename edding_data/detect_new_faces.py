import gc
import os
import time

from tqdm import tqdm
from face_detection import detect_faces_in_images, load_face_detector
from yoloface_master.utils.image_utils import read_video, video_to_images_for_detection
from PIL import Image
import cv2
import random


def get_all_files_originated_from_a_dir_by_ending(path_to_dir, ending):
    all_possible_files = os.walk(path_to_dir, topdown=True)
    all_files = []
    for root, dirs, files in all_possible_files:
        for file in files:
            if file.endswith(ending):
                all_files.append(f"{root}/{file}")
    return all_files


def detect_on_all_videos_in_directory(path_to_dir, path_to_save_dir):
    all_videos = get_all_files_originated_from_a_dir_by_ending(path_to_dir, ".mp4")
    os.makedirs(path_to_save_dir, exist_ok=True)
    margin = 50
    for a, video_path in tqdm(enumerate(all_videos), total=len(all_videos)):

        print("start video")
        if a == 1:
            print("continue")
            print("end video")
            continue
        video_name = os.path.basename(video_path)
        video = read_video(video_path)
        video = video_to_images_for_detection(video)
        target_size = max(video[0].shape[0], video[0].shape[1])
        model = load_face_detector(target_size=target_size, device="cuda", min_face=100)
        bboxes = []
        # for frame in video:
        #     bboxes += detect_faces_in_images(model, [frame], 1)
        bboxes = detect_faces_in_images(model, video, 5)
        for i, (bb_frame, frame) in enumerate(zip(bboxes, video)):
            for j, bb in enumerate(bb_frame):
                xl, yl, xr, yr = bb
                xl = max(0, xl - margin)
                xr = min(frame.shape[1], xr + margin)
                yl = max(0, yl - margin)
                yr = min(frame.shape[0], yr + margin)
                face = frame[yl:yr, xl:xr]
                face = Image.fromarray(face)
                face.save(f"{path_to_save_dir}/{video_name}___frame_{i}_face_{j}.png")
        print("end video")
        try:
            del video
            del bboxes
            del model
            del face
            gc.collect()
            try:
                print(video)
            except:
                print("success")
        except:
            pass


def detect_on_all_images_in_directory(path_to_dir, path_to_save_dir):
    all_images = get_all_files_originated_from_a_dir_by_ending(path_to_dir, ".png")
    all_images += get_all_files_originated_from_a_dir_by_ending(path_to_dir, ".jpg")
    all_images += get_all_files_originated_from_a_dir_by_ending(path_to_dir, ".jpeg")
    all_images += get_all_files_originated_from_a_dir_by_ending(path_to_dir, ".JPEG")
    all_images += get_all_files_originated_from_a_dir_by_ending(path_to_dir, ".JPG")
    os.makedirs(path_to_save_dir, exist_ok=True)
    for image_path in tqdm(all_images, total=len(all_images)):
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = video_to_images_for_detection([image])
        if image[0].shape[0] > 3 and image[0].shape[-1] > 3:
            # print(f"image {image_path} is weird: {image[0].shape}")
            # continue
            image[0] = cv2.cvtColor(image[0], cv2.COLOR_BGRA2BGR)
        target_size = max(image[0].shape[0], image[0].shape[1])
        model = load_face_detector(target_size=target_size, device="cuda", min_face=100)
        bboxes = detect_faces_in_images(model, image, 1)
        for i, (bb_frame, frame) in enumerate(zip(bboxes, image)):
            for j, bb in enumerate(bb_frame):
                margin = random.randint(100, 200)
                xl, yl, xr, yr = bb
                xl = max(0, xl - margin)
                xr = min(frame.shape[1], xr + margin)
                yl = max(0, yl - margin)
                yr = min(frame.shape[0], yr + margin)
                face = frame[yl:yr, xl:xr]
                face = Image.fromarray(face)
                face.save(f"{path_to_save_dir}/{image_name}___face_{j}.png")


detect_on_all_videos_in_directory(f"{os.getcwd()}/even_videos",
                                  "faces_from_videos")
# detect_on_all_images_in_directory("/home/user/iron_swords/face_detection_superresolution/even_videos",
#                                   "faces_from_images")
