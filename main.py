from argparser import parse_args
from image_utils import read_video, read_image, read_images_from_dir
from detection_and_super_resolution import video_face_detection_and_super_resolution
import warnings

args = parse_args()

if args.video_path is None and args.image_path is None and args.folder_path is None:
    raise IOError("no input path was given, please give image_path, video_path, or folder_path")
if (args.video_path and args.image_path) or (args.video_path and args.folder_path) or (args.image_path and args.folder):
    raise IOError("2 of the 3 possible path options: video_path, image_path, and folder_path, most be None")
if args.video_path and args.keep_original_names:
    warnings.warn("original names are impossible for a video")
    names = None

if args.video_path:
    video = read_video(args.video_path)[200:300]
elif args.image_path:
    video, names = read_image(args.image_path)
else:
    args.detection_batch_size = 1
    video, names = read_images_from_dir(args.folder_path)
if not args.keep_original_names:
    names = None

video_face_detection_and_super_resolution(video, args, names)
