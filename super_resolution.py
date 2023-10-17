"""
Script for performing super resolution on a given image or crops from an image.
"""
import torch
import numpy as np
from yoloface_master.utils.image_utils import sharp_edges

from PAN.codes.models.archs import PAN_arch
from PAN.codes.utils.util import single_forward, tensor2img
import cv2


def load_upscaler(device="cpu", scale=2):
    """
    loads the super resolution model from a chosen scale.
    @param device: the device to load the model to: cpu or cuda
    @param scale: the scale of the model to load: 2,3,4
    @return: the super resolution model
    """
    model = PAN_arch.PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=scale)
    if scale == 2:
        model_weight = torch.load('./PAN/experiments/pretrained_models/PANx2_DF2K.pth')
    elif scale == 3:
        model_weight = torch.load('./PAN/experiments/pretrained_models/PANx3_DF2K.pth')
    elif scale == 4:
        model_weight = torch.load('./PAN/experiments/pretrained_models/PANx4_DF2K.pth')
    else:
        raise ValueError(f"no model with scale {scale} exists")
    model.load_state_dict(model_weight)
    model = model.to(device)
    return model


def img2tensor(img):
    """
    converts image to tensor and prepares it for super resolution
    @param img: image as a numpy array (for example when loaded with cv2)
    @return: the image as a tensor
    """
    imgt = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return imgt


def upscale_image(img, model, device="cpu"):
    """
    applies super resolution to the image with the provided model.
    @param img: image as a numpy array
    @param model: super resolution model that should return a new image
    @return: a new upsampled image
    """
    img = sharp_edges(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255
    imgt = img2tensor(img)
    imgt = imgt[None, ...]

    if device == "cpu":
        output = single_forward(model, imgt)
    else:
        output = single_forward(model, imgt.cuda())

    return sharp_edges(tensor2img(output))


def upscale_crops(img, crops, model, device="cpu"):
    """
    this function receives an image, crops coordinates and a super resolution model. The function upsamples each
    crop with super resolution
    @param img: an image as a numpy array
    @param crops: list of crop coordinates, bounding boxes of format xyxy
    @param model: super resolution model
    @param device: the device to load the crops to
    @return: a list of upsampled crops from the image
    """
    img_crops = []

    for crop_coords in crops:
        xl, yl, xr, yr = crop_coords
        xl = max(0, xl - 75)
        xr = min(img.shape[1], xr + 75)
        yl = max(0, yl - 75)
        yr = min(img.shape[0], yr + 75)
        img_crops.append(img[yl:yr, xl:xr])

    upscaled_crops = []
    for crop in img_crops:
        output = upscale_image(crop, model, device)
        upscaled_crops.append(output)

    return [item for item in upscaled_crops]


