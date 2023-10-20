from argparser import parse_args
from image_utils import read_video, read_image
from detection_and_super_resolution import video_face_detection_and_super_resolution

args = parse_args()
if args.video_path is None and args.image_path is None:
    raise IOError("no input path was given, please give image_path or video_path")
if args.video_path and args.image_path:
    raise IOError("video_path or image_path most be None")
if args.video_path:
    video = read_video(args.video_path)
else:
    video = read_image(args.image_path)
super_resolution_faces = video_face_detection_and_super_resolution(video, args)
