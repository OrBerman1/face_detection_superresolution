"""
Script for performing super resolution on a given image or crops from an image.
"""
import torch
import torch.nn as nn
import numpy as np
import model.MSFSR as MSFSR
from PIL import Image
from yoloface_master.utils.image_utils import resize_image_to_power_of_2, sharp_edges

from torchvision import transforms


class MSFSRmodel(nn.Module):
    def __init__(self, stg1, stg2, stg3, stages):
        super().__init__()
        self.stg1 = stg1
        self.stg2 = stg2
        self.stg3 = stg3
        self.stg_list = [self.stg1, self.stg2, self.stg3]
        self.stages = stages

    def forward(self, img):
        for i in range(self.stages):
            out = self.stg_list[i](img)[2]
        return out


def load_upscaler(device="cpu",  stages=1):
    """
    loads the super resolution model for a chosen scale.
    @param checkpoint_path: the path to the model checkpoint to load
    @param device: the device to load the model to: cpu or cuda
    @param scale: the scale of the model to load: 2,3,4
    @return: the super resolution model
    """
    X2_SR_net_stg1 = MSFSR.defineThreeStageGenerator(input_nc=3, output_nc=3)
    X2_SR_net_stg2 = MSFSR.defineThreeStageGenerator(input_nc=3, output_nc=3)
    X2_SR_net_stg3 = MSFSR.defineThreeStageGenerator(input_nc=3, output_nc=3)

    weights1 = torch.load(f"MSFSR/pretrained_weights/MSFSR/model_stg1_state.pth")
    weights2 = torch.load(f"MSFSR/pretrained_weights/MSFSR/model_stg2_state.pth")
    weights3 = torch.load(f"MSFSR/pretrained_weights/MSFSR/model_stg3_state.pth")

    X2_SR_net_stg1.load_state_dict(weights1)
    X2_SR_net_stg2.load_state_dict(weights2)
    X2_SR_net_stg3.load_state_dict(weights3)

    X2_SR_net_stg1.to(device)
    X2_SR_net_stg1.eval()
    X2_SR_net_stg2.to(device)
    X2_SR_net_stg2.eval()
    X2_SR_net_stg3.to(device)
    X2_SR_net_stg3.eval()

    model = MSFSRmodel(X2_SR_net_stg1, X2_SR_net_stg2, X2_SR_net_stg3, stages).to(device)
    return model.to(device)


def img2tensor(img):
    """
    converts image to tensor and prepares it for super resolution
    @param img: image as a numpy array (for example when loaded with cv2)
    @return: the image as a tensor
    """
    imgt = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return imgt


def upscale_image(img, model, device="cpu", sharp_before=False):
    """
    applies super resolution to the image with the provided model.
    @param img: image as a numpy array
    @param model: super resolution model that should return a new image
    @param device: device
    @param sharp_before: sharp edges before super resolution model
    @return: a new upsampled image
    """
    to_image = transforms.Compose([
        transforms.ToPILImage()
    ])
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        img = resize_image_to_power_of_2(Image.fromarray(img))
        if sharp_before:
            img = sharp_edges(img)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img.to(device)

        out = model(img.to(device))
        output = out.cpu().clone()
        output = output.squeeze(0)

    return sharp_edges(to_image(output))


def upscale_crops(img, crops, model, margin, device="cpu", sharp_before=False):
    """
    this function receives an image, crops coordinates and a super resolution model. The function upsamples each
    crop with super resolution
    @param img: an image as a numpy array
    @param crops: list of crop coordinates, bounding boxes of format xyxy
    @param model: super resolution model
    @param margin: enlarge the bb by a margin
    @param device: the device to load the crops to
    @param sharp_before: sharp image before super resolution model
    @return: a list of upsampled crops from the image
    """
    img_crops = []

    for crop_coords in crops:
        xl, yl, xr, yr = crop_coords
        xl = max(0, xl - margin)
        xr = min(img.shape[1], xr + margin)
        yl = max(0, yl - margin)
        yr = min(img.shape[0], yr + margin)
        img_crops.append(img[yl:yr, xl:xr])

    upscaled_crops = []
    for crop in img_crops:
        output = upscale_image(crop, model, device, sharp_before)
        upscaled_crops.append(output)

    return [item for item in upscaled_crops]
