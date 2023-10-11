import glob
import os
import random
import time

from PIL import Image
import cv2
import tqdm
from threading import Thread
import queue


def train_val_split(path_to_datasets, data_dir):
    paths = [x for x in glob.glob(f"{path_to_datasets}/{data_dir}/*")]
    os.makedirs(os.path.join(path_to_datasets, f"{data_dir}_train"))
    os.makedirs(os.path.join(path_to_datasets, f"{data_dir}_val"))

    random.shuffle(paths)
    train_paths = paths[:int(len(paths) * 0.8)]
    val_paths = [x for x in paths if x not in train_paths]

    for pth in train_paths:
        cur_img = Image.open(pth)
        cur_img.save(os.path.join(path_to_datasets, f"{data_dir}_train", os.path.basename(pth)))

    for pth in val_paths:
        cur_img = Image.open(pth)
        cur_img.save(os.path.join(path_to_datasets, f"{data_dir}_val", os.path.basename(pth)))


def create_lr_data_thread(tasks: queue.Queue, path_to_datasets, data_dir, scale):
    while not tasks.empty():
        try:
            task = tasks.get()
            img = cv2.imread(task)
            lr_img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(path_to_datasets, f"{data_dir}_lr_{scale}", os.path.basename(task)), lr_img)
        except queue.Empty:
            pass


def create_lr_data(path_to_datasets, data_dir, scale, num_threads=20):
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
    create_lr_data("../PAN/datasets", "img_align_celeba_val", 4)


