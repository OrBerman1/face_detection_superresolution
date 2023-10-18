import glob
import os
import random
import time

from PIL import Image
import cv2
import tqdm
from threading import Thread
import queue


def get_min_and_max_size(dir_path):
    max_width, max_height = 0, 0
    for img_path in os.listdir(dir_path):
        img = Image.open(f'{dir_path}/{img_path}')
        width, height = img.size
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
    print(max_width, max_height)


def get_all_paths(dataset_path):
    full_paths_ls = []
    files = os.listdir(dataset_path)
    for file in files:
        full_paths_ls.append(f"{dataset_path}/{file}")
    return full_paths_ls


def train_val_split(path_to_datasets, data_dir, path_to_save):
    paths = []
    for dataset in path_to_datasets:
        paths += get_all_paths(dataset)

    os.makedirs(os.path.join(path_to_save, f"{data_dir}_train"), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, f"{data_dir}_val"), exist_ok=True)

    random.shuffle(paths)
    train_paths = paths[:int(len(paths) * 0.8)]
    val_paths = [x for x in paths if x not in train_paths]

    for pth in tqdm.tqdm(train_paths, total=len(train_paths)):
        cur_img = Image.open(pth)
        cur_img.save(os.path.join(path_to_save, f"{data_dir}_train", os.path.basename(pth)))

    for pth in tqdm.tqdm(val_paths, total=len(val_paths)):
        cur_img = Image.open(pth)
        cur_img.save(os.path.join(path_to_save, f"{data_dir}_val", os.path.basename(pth)))


def create_lr_data_thread(tasks: queue.Queue, path_to_datasets, data_dir, scale):
    while not tasks.empty():
        try:
            task = tasks.get()
            img = cv2.imread(task)
            lr_img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(path_to_datasets, f"{data_dir}_lr_{scale}", os.path.basename(task)), lr_img)
        except queue.Empty:
            pass


def create_lr_data(path_to_datasets, data_dir, scale, num_threads=1):
    os.makedirs(os.path.join(path_to_datasets, f"{data_dir}_lr_{scale}"), exist_ok=True)
    tasks = queue.Queue()
    for pth in glob.glob(f"{path_to_datasets}/{data_dir}/*"):
        tasks.put(pth)
    threads = []
    for i in range(num_threads):
        t = Thread(target=create_lr_data_thread, args=(tasks, path_to_datasets, data_dir, scale))
        t.start()
        threads.append(t)

    while not tasks.empty():
        print(tasks.qsize())
        time.sleep(2)

    for t in threads:
        t.join()


if __name__ == '__main__':
    train_val_split(["/home/user/iron_swords/face_detection_superresolution/edding_data/faces_from_images",
                     "/home/user/iron_swords/face_detection_superresolution/edding_data/faces_from_videos",
                     "/home/user/iron_swords/face_detection_superresolution/PAN/datasets/celeba_original/img_align_celeba_train",
                     "/home/user/iron_swords/face_detection_superresolution/PAN/datasets/celeba_original/img_align_celeba_val"],
                    "merged_even_data", path_to_save="/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged")
    print("finished train test split")
    create_lr_data("/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged", "merged_even_data_train", 4)
    create_lr_data("/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged", "merged_even_data_val", 4)
    # get_min_and_max_size("/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged/merged_even_data_train")
    # get_min_and_max_size("/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged/merged_even_data_val")
