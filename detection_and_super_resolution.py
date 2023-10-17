import face_detection
from super_resolution import load_upscaler, upscale_crops
from PIL import Image
import os


def video_face_detection_and_super_resolution(video, args):
    """
    This function executes the super resolution process for a given video. Crops of faces are extracted from each frame
    and are than upsampled with super resolution. The function saves images of the crops in a chosen save path.
    @param video: a video as a list of PIL images
    @return: a list of PIL images of super resolution faces
    """
    if args.save_path is None:
        path = args.video_path if args.video_path else args.image_path
        args.save_path = f"results/{os.path.basename(path)}"
    if args.visualize_bbs:
        video, bbs, _ = face_detection.detection_pipeline(video)
    else:
        video, bbs = face_detection.detection_pipeline(video)
    model = load_upscaler(args.checkpoint_path, args.device, args.scale)
    model.eval()

    os.makedirs(f"{args.save_path}", exist_ok=True)
    super_resolution_faces = []
    for i, (frame, bb) in enumerate(zip(video, bbs)):
        if len(bb) > 0:
            upscaled_crops = upscale_crops(frame, bb, model, args.margin, args.device)
            for j, pil_img in enumerate(upscaled_crops):
                if args.save_path is not None:
                    pil_img.save(f"{args.save_path}/frame_{i}_face_{j}_scale={args.scale}.jpg")
                super_resolution_faces.append(pil_img)
    return super_resolution_faces

