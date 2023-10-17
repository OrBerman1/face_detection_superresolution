import argparse

args = argparse.ArgumentParser()
args.add_argument("--detection_batch_size", type=int, default=20, help="number of frames for inference batch,"
                                                                       "-1 means all the video in one batch")
args.add_argument("--min-face", type=int, default=10, help="minimum possible size of faces in image")
args.add_argument("--visualize-bbs", default=True, action="store_true")
args.add_argument("--scale", type=int, help="Super resolution model scale to use", default=2)
args.add_argument("--save-path", type=str, help="path to where to save output", default=None)
args.add_argument("--video_path", type=str, help="path to video", default=None)
args.add_argument("--image_path", type=str, help="path to image", default=None)

args.add_argument("--device", type=str, default="cuda")


def parse_args():
    return args.parse_args()
