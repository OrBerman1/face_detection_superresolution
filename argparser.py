import argparse

args = argparse.ArgumentParser()
args.add_argument("--detection_batch_size", type=int, default=20, help="number of frames for inference batch,"
                                                                       "-1 means all the video in one batch")
args.add_argument("--min-face", type=int, default=10, help="minimum possible size of faces in image")
args.add_argument("--visualize-bbs", default=True, action="store_true")
args.add_argument("--margin", type=int, help="enlarge the bbs by a margin", default=20)
args.add_argument("--save-path", type=str, help="path to where to save output"
                                                "if you dont wont to save the faces enter None", default="./msfsr_results_img_sharp")
args.add_argument("--checkpoint", type=str, help="path to the model checkpoint", default="/home/user/iron_swords/face_detection_superresolution/MSFSR/pretrained_weights/MSFSR")
args.add_argument("--device", type=str, default="cuda")


def parse_args():
    return args.parse_args()
