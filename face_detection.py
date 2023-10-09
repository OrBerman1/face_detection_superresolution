from face_detector import YoloDetector
from argparser import parse_args
from utils import image_utils


def detect_faces_in_images(model, img, bs):
    """
    detect the faces in the image
    @param model: the detector
    @param img: the images to detect faces on
    @param bs: batch size for inference
    @return: bounding boxes for each frame in format: [[(x00, y00, x01, y01), ..., (x0n, y0n, x1n, y1n)], ... [(...)]]
    """
    if bs != -1:
        bboxes = []
        list_chunked = [img[i:i + bs] for i in range(0, len(img), bs)]
        for ls in list_chunked:
            bbs, points = model.predict(ls)
            for bb in bbs:
                bboxes.append(bb)
    else:
        bboxes, points = model.predict(img)
    return bboxes


def load_face_detector(target_size=720, device="cpu", min_face=20):
    """
    return face detector model
    @param target_size: the size of the squared image
    @param device: device to run on
    @param min_face: minimum possible size of faces in images
    @return: a model
    """
    return YoloDetector(target_size=target_size, device=device, min_face=min_face)


def detection_pipeline(video):
    """
    preprocess the video and returns the processes video and bboxes
    @param video: a list of frames
    @return: process video and a list of lists of bboxes, one for each frame. If visualize, return also video with bboxes
    """
    args = parse_args()
    frame = video[0]
    image_size = max(frame.size)
    model = load_face_detector(target_size=image_size, device=args.device, min_face=args.min_face)
    video = image_utils.video_to_images_for_detection(video)
    bboxes = detect_faces_in_images(model, video, args.detection_batch_size)
    if args.visualize_bbs:
        draw_video = image_utils.draw_bbs_on_video(video, bboxes)
        return video, bboxes, draw_video
    return video, bboxes
