import torch
from yoloface_master.utils.image_utils import read_video, read_image, read_images_from_dir
from detection_and_super_resolution import video_face_detection_and_super_resolution

if __name__ == '__main__':
    # model = YoloDetector(target_size=720, device="cpu", min_face=20)
    # img = Image.open("manyfaces.jpg")
    # real_image = img.resize((720, 720))
    # img = np.array(real_image)
    # bboxes, _ = detect_faces_in_image(model, img)
    # image_to_draw = ImageDraw.Draw(real_image)
    # for bb in bboxes[0]:
    #     start = (bb[0], bb[1])
    #     end = (bb[2], bb[3])
    #     image_to_draw.rectangle([start, end], outline="red")
    # real_image.show()

    # elif args.video_path is not None:
    #     video = read_video(video_path)
    # elif args.image_path is not None:
    #     video = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)
    # else:
    #     raise ValueError("Both video_path and image_path arguments are not set, one should be passed")

    # video_path = "VID-20231007-WA0155.mp4"
    # video_path = "/home/user/iron_swords/face_detection_superresolution/video2.mp4"
    # video = read_video(video_path)
    # image_path = "/home/user/iron_swords/face_detection_superresolution/snir_bar.PNG"
    # video = read_image(image_path)
    dir_path = r"/home/user/iron_swords/face_detection_superresolution/images_for_even"
    video = read_images_from_dir(dir_path)
    super_resolution_faces = video_face_detection_and_super_resolution(video)
