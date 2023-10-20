"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from codes.utils.util import ProgressBar  # noqa: E402
import codes.data.util as data_util  # noqa: E402
from yoloface_master.utils.image_utils import sharp_edges


def main():
    mode = 'pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = {}
    opt['n_thread'] = 1
    opt['compression_level'] = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    if mode == 'single':
#         opt['input_folder'] = '/mnt/hyzhao/Documents/datasets/DIV2K_train800/DIV2K_train_LR_bicubic/X4_blur'
#         opt['save_folder'] = '/mnt/hyzhao/Documents/datasets/DIV2K_train800/DIV2K_train_LR_bicubic/X4_blur_sub'
        opt['input_folder'] = '/mnt/hyzhao/Documents/datasets/DIV2K_train800/DIV2K_train_Bic'
        opt['save_folder'] = '/mnt/hyzhao/Documents/datasets/DIV2K_train800/DIV2K_train_Bic/Bic_sub480'
        opt['crop_sz'] = 480  # the size of each sub-image
        opt['step'] = 120  # step of the sliding crop window
        opt['thres_sz'] = 48  # size threshold
        extract_signle(opt)
    elif mode == 'pair':
#         GT_folder = '../../datasets/DIV2K/DIV2K_train_HR'
#         LR_folder = '../../datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
#         save_GT_folder = '../../datasets/DIV2K/DIV2K800_sub'
#         save_LR_folder = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4'

        GT_folder = '/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged/merged_filtered_even_data_train'
        LR_folder = '/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged/merged_filtered_even_data_train_lr_4'
        save_GT_folder = '/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged_extracted/merged_filtered_even_data_train_sharp'
        save_LR_folder = '/home/user/iron_swords/face_detection_superresolution/PAN/datasets/even_merged_extracted/merged_filtered_even_data_train_lr_4_sharp'

        os.makedirs(save_GT_folder, exist_ok=True)
        os.makedirs(save_LR_folder, exist_ok=True)
        
        scale_ratio = 4
        # crop_sz = 360  # the size of each sub-image (GT)
        crop_sz = (216, 180)
        step = 180  # step of the sliding crop window (GT)
    
        thres_sz = 48  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        img_GT_list = data_util._get_paths_from_images(GT_folder)
        img_LR_list = data_util._get_paths_from_images(LR_folder)
        assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        # for path_GT, path_LR in zip(img_GT_list, img_LR_list):
        #     img_GT = Image.open(path_GT)
        #     img_LR = Image.open(path_LR)
        #     w_GT, h_GT = img_GT.size
        #     w_LR, h_LR = img_LR.size
        #     assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
        #         w_GT, scale_ratio, w_LR, path_GT)
        #     assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
        #         w_GT, scale_ratio, w_LR, path_GT)
        # check crop size, step and threshold size
        assert crop_sz[0] % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
            scale_ratio)
        assert crop_sz[1] % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
            scale_ratio)
        assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
            scale_ratio)
        print('process GT...')
        opt['input_folder'] = GT_folder
        opt['save_folder'] = save_GT_folder
        opt['crop_sz'] = crop_sz
        opt['step'] = step
        opt['thres_sz'] = thres_sz
        opt['blur'] = False
        opt['sharp'] = False
        extract_signle(opt)
        print('process LR...')
        opt['input_folder'] = LR_folder
        opt['save_folder'] = save_LR_folder
        opt['crop_sz'] = (crop_sz[0] // scale_ratio, crop_sz[1] // scale_ratio)
        opt['step'] = step // scale_ratio
        opt['thres_sz'] = thres_sz // scale_ratio
        opt['blur'] = True
        opt['sharp'] = True
        extract_signle(opt)
        assert len(data_util._get_paths_from_images(save_GT_folder)) == len(
            data_util._get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    else:
        raise ValueError('Wrong mode.')


def extract_signle(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        pass
        # print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        # sys.exit(1)
    img_list = data_util._get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        path = path.replace("\\", "/")
        result = pool.apply_async(worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()


    print('All subprocesses done.')


def worker(path, opt):
    crop_sz_h, crop_sz_w = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz_h + 1, step)
    if h - (h_space[-1] + crop_sz_h) > thres_sz:
        h_space = np.append(h_space, h - crop_sz_h)
    w_space = np.arange(0, w - crop_sz_w + 1, step)
    if w - (w_space[-1] + crop_sz_w) > thres_sz:
        w_space = np.append(w_space, w - crop_sz_w)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w]
            else:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w, :]
            if opt['blur']:
                crop_img = cv2.blur(img, (3, 3))
            if opt['sharp']:
                crop_img = sharp_edges(crop_img)
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
