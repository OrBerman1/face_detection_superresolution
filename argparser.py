import argparse

args = argparse.ArgumentParser()
args.add_argument("--detection_batch_size", type=int, default=5, help="number of frames for inference batch,"
                                                                       "-1 means all the video in one batch")
args.add_argument("--min_face", type=int, default=10, help="minimum possible size of faces in image")
args.add_argument("--visualize-bbs", default=True, action="store_true")
args.add_argument("--scale", type=int, help="Super resolution model scale to use", default=2)
args.add_argument("--save_path", type=str, help="path to where to save output", default=None)
args.add_argument("--video_path", type=str, help="path to video", default=None)
args.add_argument("--image_path", type=str, help="path to image", default=None)
args.add_argument("--folder_path", type=str, help="path to a folder of images", default=None)
args.add_argument("--margin", type=int, help="how many pixels beyond face to take", default=45)
args.add_argument("--keep_original_names", action="store_true", default=False, help="keep images original "
                                                                                   "name while saving, relevant only "
                                                                                   "for image_path or folder_path,"
                                                                                   "note! There most be only one face "
                                                                                   "in each image for it to work!")  # IMPORTANT! read this documentation!!!
args.add_argument("--sharp_edges", action="store_true", default=False, help="use edges enhancing algorithm"
                                                                            "before the super-resolution model. this "
                                                                            "might yield better or worse results")
args.add_argument("--detector_off", action="store_true", default=False, help="if true skip the face detector phase")
args.add_argument("--checkpoint_path", type=str, help="path to model", default="/home/user/iron_swords/face_detection_superresolution/PAN/experiments/pretrained_models/PANx2_DF2K.pth")

args.add_argument("--device", type=str, default="cuda")


def parse_args():
    return args.parse_args()
