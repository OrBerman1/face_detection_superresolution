import face_detection
from super_resolution import load_upscaler, upscale_crops
from PIL import Image
import os
import warnings
from utils.image_utils import get_bbs_size_of_images


def video_face_detection_and_super_resolution(video, args, names=None):
    """
    This function executes the super resolution process for a given video. Crops of faces are extracted from each frame
    and are than upsampled with super resolution. The function saves images of the crops in a chosen save path.
    @param video: a video as a list of PIL images
    @param args: the argument parser object
    @param names: a list of names of the images
    @return: a list of PIL images of super resolution faces
    """
    if args.save_path is None:
        path = args.video_path if args.video_path else args.image_path
        args.save_path = f"results/{os.path.basename(path)}"
    if not args.detector_off:   # use detector
        if args.visualize_bbs:
            video, bbs, _ = face_detection.detection_pipeline(video, args)
        else:
            video, bbs = face_detection.detection_pipeline(video, args)
    else:
        args.detector_batch_size = 1
        video, bbs = get_bbs_size_of_images(video)
    model = load_upscaler(args.checkpoint_path, args.device, args.scale)
    model.eval()

    os.makedirs(f"{args.save_path}", exist_ok=True)
    os.makedirs(f"{args.save_path}/original", exist_ok=True)
    for i, (frame, bb) in enumerate(zip(video, bbs)):
        if len(bb) > 0:
            upscaled_crops, img_crops = upscale_crops(frame, bb, model, args.margin, args.device, args.sharp_edges)
            img_name = None
            if len(upscaled_crops) == 1 and names:
                img_name = names[i]
            elif names:
                warnings.warn(f"face detector find more then one face in image: {names[i]}.\n"
                              f"save with generic names")
            for j, (item, crop) in enumerate(zip(upscaled_crops, img_crops)):
                pil_img = Image.fromarray(item)
                pil_crop = Image.fromarray(crop)
                if args.save_path is not None:
                    if img_name:
                        pil_img.save(f"{args.save_path}/{img_name}")
                        pil_crop.save(f"{args.save_path}/original/{img_name}")

                    else:
                        pil_img.save(f"{args.save_path}/frame_{i}_face_{j}_scale={args.scale}.jpg")
                        pil_crop.save(f"{args.save_path}/original/frame_{i}_face_{j}_scale={args.scale}.jpg")

        else:
            if names:
                warnings.warn(f"face detector could not find faces in image {names[i]}")

