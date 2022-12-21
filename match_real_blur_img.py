import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
import torchvision.transforms as transforms

from extract_patches.core import extract_patches

from configs import config_hpatches
from utils import common_utils
from utils.logger import logger
from model import get_model
from datasets import dataset_utils
from benchmark_test import geometry_tools, repeatability_tools, test_utils

from matcher import MatcherWrapper

from model.hardnet_pytorch import HardNet 


cv2_greyscale = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
np_reshape = lambda x: np.reshape(x, (32, 32, 1))

def get_transforms(color):
    
    MEAN_IMAGE = 0.443728476019
    STD_IMAGE = 0.20197947209

    transform = transforms.Compose([
        transforms.Lambda(cv2_greyscale), transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape), transforms.ToTensor(),
        transforms.Normalize((MEAN_IMAGE, ), (STD_IMAGE, ))
    ])

    return transform


def extract_blur_diff_features(image_RGB_norm, model, device, args, is_debugging=False):
    height_RGB_norm, width_RGB_norm = image_RGB_norm.shape[0], image_RGB_norm.shape[1]
    
    image_even = dataset_utils.make_shape_even(image_RGB_norm)
    height_even, width_even = image_even.shape[0], image_even.shape[1]
    
    image_pad = dataset_utils.mod_padding_symmetric(image_even, factor=64)
    

    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)

    t0 = time.time()
    output_pad_batch = model(image_pad_batch.to(device))
    t1 = time.time()
    extract_time = t1-t0

    # elif args.nms == 'nms_fast':
    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]


    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=args.border_size)

    ## method 1 for keypoints
    score_map_nms = repeatability_tools.apply_nms(score_map_remove_border, args.nms_size)

    pts = geometry_tools.get_point_coordinates(score_map_nms, num_points=args.num_points, order_coord='xysr')

    if pts.size == 0:
        return None, None

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:args.num_points]

    return pts_output[:,0:2], extract_time # pts_output.shape: N*4


def extract_kpts(image_RGB_norm, model, device, args, num_points, is_debugging=False):
    height_RGB_norm, width_RGB_norm = image_RGB_norm.shape[0], image_RGB_norm.shape[1]
    
    image_even = dataset_utils.make_shape_even(image_RGB_norm)
    height_even, width_even = image_even.shape[0], image_even.shape[1]
    
    image_pad = dataset_utils.mod_padding_symmetric(image_even, factor=64)
    

    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)

    t0 = time.time()
    output_pad_batch = model(image_pad_batch.to(device))
    t1 = time.time()
    extract_time = t1-t0

    # elif args.nms == 'nms_fast':
    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]


    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=args.border_size)

    ## method 2 for keypoints
    pts = repeatability_tools.get_points_direct_from_score_map(
        heatmap=score_map_remove_border, conf_thresh=args.heatmap_confidence_threshold,
        nms_size=args.nms_size, subpixel=args.sub_pixel, patch_size=args.patch_size, order_coord=args.order_coord
    )

    if pts.size == 0:
        return None, None

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:num_points]

    return pts_output[:,0:2], extract_time # pts_output.shape: N*4

def compute_sift_desc(img, kpts):
    keypoints = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 13) for pt in kpts]
    sift = cv2.xfeatures2d.SIFT_create()
    desc = sift.compute(img, keypoints)
    return desc


def compute_hardnet_desc(kpts, im, model, args, device, transforms):
    # im = cv2.cvtColor(
    #     cv2.imread(img_path),
    #     cv2.COLOR_BGR2RGB)

    keypoints = [cv2.KeyPoint(float(pt[0]), float(pt[1]), _size=1.0, _angle=0) for pt in kpts]
    patches = extract_patches(
        keypoints, im, args.patchSize, args.mrSize)
    # print(len(patches))

    patches = np.array(patches).astype(np.uint8)

    bs = 128
    descriptors = np.zeros((len(patches), 128))

    for i in range(0, len(patches), bs):
        data_a = patches[i:i + bs, :, :, :]
        data_a = torch.stack(
            [transforms(patch) for patch in data_a]).to(device)
        # compute output
        with torch.no_grad():
            out_a = model(data_a)
            descriptors[i:i + bs] = out_a.cpu().detach().numpy()

    return descriptors.astype(np.float32)


def draw_matcher(image1_path, image2_path,
                 keypoints1, keypoints2,
                 descriptors1, descriptors2,
                 match_type, feature):

    t0 = time.time()
    matcher = MatcherWrapper(
        descriptors1, descriptors2, feature, match_type,
        ratio=0.9, cross_check=True
    )
    t1 = time.time()
    time_match = t1-t0

    putative_matches, mask, time_ransac, n_inliers, F_matrix, corres1, corres2 = matcher.get_inliers(
        keypoints1, keypoints2,
        err_thld=3, ransac=True, info=feature
    )

    output = matcher.draw_matches(image1_path, keypoints1, image2_path, keypoints2, putative_matches, mask)
    cv2.namedWindow('best 200 matches', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("best 200 matches", 1440, 960)
    cv2.imshow('best 200 matches', output)

    # display_rgb = dataset_utils.rgb_tobgr(output)
    cv2.imwrite('./results_match_real_data/test_match/match.png', output)
    # jet_img = str(vis_featuremap_dir) + '/' + str(i) + '_jet.jpg'
    #         plt.figure(1)
    #         plt.axis("off")
    #         # plt.imshow(feature_map_np[i,:,:], cmap='jet')
    #         # jet_fig = plt.gcf()
    #         # jet_fig.savefig(jet_img, dpi=100, bbox_inches='tight', pad_inches=0)
    cv2.waitKey(1000)
    # return output, time_match, time_ransac, len(putative_matches), n_inliers
    return F_matrix, corres1, corres2


def extract_corres(points1, points2, F):
    correspondences = []
    for idx1, point1 in enumerate(points1):
        point1_hom = np.array([point1[0], point1[1], 1.0])
        
        min_dis = np.inf
        for idx2, point2 in enumerate(points2):
            point2_hom = np.array([point2[0], point2[1], 1.0])

            distance = np.dot(point2_hom, np.dot(F, point1_hom.transpose()))

            if distance < min_dis:
                min_dis = distance
                match_point2 = point2.copy()

        print(match_point2)
        correspondences.append([point1, match_point2])

    return correspondences


# def extract_sharp_blur_corres(sharpA_kpts, sharpB_kpts, blur_kpts):
#     blur_corres = []
#     sharpA_corres = []
#     sharpB_corres = []
#     # correspondences = []
#     for idx1, blur_point in enumerate(blur_kpts):
#         # blur_point_hom = np.array([blur_point[0], blur_point[1], 1.0])
        
#         min_dis = np.inf
#         for idx2, sharpB_point in enumerate(sharpB_kpts):
#             # point2_hom = np.array([point2[0], point2[1]])
            
#             distance = np.absolute(np.sum(blur_point-sharpB_point))

#             if distance < min_dis:
#                 min_dis = distance
#                 sharpB_match_point = sharpB_point.copy()
#                 sharpA_match_point = sharpA_kpts[idx2,:].copy()

#         # print(min_dis)
#         blur_corres.append(blur_kpts[idx1, :])
#         sharpA_corres.append(sharpA_match_point)
#         sharpB_corres.append(sharpB_match_point)
#         # correspondences.append([point1, match_point2])

#     return np.asarray(blur_corres), np.asarray(sharpA_corres), np.asarray(sharpB_corres)

def extract_sharp_blur_corres(sharpA_kpts, sharpB_kpts, blur_kpts, dis_thr):
    blur_corres = []
    sharpA_corres = []
    sharpB_corres = []
    # correspondences = []
    for idx1, blur_point in enumerate(blur_kpts):
        # blur_point_hom = np.array([blur_point[0], blur_point[1], 1.0])
        
        found_possible_match = False
        for idx2, sharpB_point in enumerate(sharpB_kpts):
            # point2_hom = np.array([point2[0], point2[1]])
            
            # distance = np.absolute(np.sum(blur_point-sharpB_point))

            distance = (((sharpB_point[0] - blur_point[0]) ** 2) + ((sharpB_point[1] - blur_point[1]) ** 2)) ** 0.5

            if distance < dis_thr and not found_possible_match:
                # print(distance)
                # min_dis = distance
                found_possible_match = True
                sharpB_match_point = sharpB_point.copy()
                sharpA_match_point = sharpA_kpts[idx2,:].copy()
                blur_corres.append(blur_kpts[idx1, :])
                sharpA_corres.append(sharpA_match_point)
                sharpB_corres.append(sharpB_match_point)
                continue


        # print(min_dis)
        
        # correspondences.append([point1, match_point2])

    return np.asarray(blur_corres), np.asarray(sharpA_corres), np.asarray(sharpB_corres)



def draw_corres(img1, img2, corres1, corres2):

    image_stack = np.hstack((img1,img2))
    detection_im1 = test_utils.draw_keypoints(image_stack, corres1, radius=3)
    cv2.imshow('detection_im1', detection_im1)
    corres2[:,0] = corres2[:,0] + 965
    detection_im2 = test_utils.draw_keypoints(detection_im1, corres2, radius=3)
    cv2.imshow('detection_im2', detection_im1)
    for idx in range(corres1.shape[0]):
        # bgr = np.random.randint(0,255,3,dtype=np.int32)
        bgr = (0,255,0)
        cv2.line(detection_im2, 
            (corres1[idx, 0].astype(int), corres1[idx, 1].astype(int)),
            (corres2[idx, 0].astype(int), corres2[idx, 1].astype(int)),
            (int(bgr[0]),int(bgr[1]),int(bgr[2])), lineType=cv2.LINE_AA)
    cv2.imshow('match_blur_sharp', detection_im2)
    # cv2.imwrite('./results_match_real_data/test_match/blur_sharp_viewpoint.bmp', detection_im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detection_im2

def draw_corres_opencv(img1, cv_kpts1, img2, cv_kpts2, good_matches, mask, 
                    match_color=(0, 255, 0), pt_color=(0, 255, 100)):
    if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
        cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1)
                    for i in range(cv_kpts1.shape[0])]
        cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1)
                    for i in range(cv_kpts2.shape[0])]
    display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                None,
                                matchColor=match_color,
                                singlePointColor=pt_color,
                                matchesMask=mask.ravel().tolist(),
                                flags=4)
    return display

def main():
    args, cfg = config_hpatches.parse_match_real_blur_img_config()

    output_dir = Path(args.results_match_dir, args.exper_name)
    common_utils.check_directory(output_dir)
    
    logger.initialize(args, output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model.load_model(cfg['model'])
    _,_ = get_model.load_pretrained_model(model=model, filename=args.ckpt_file, logger=logger)
    model.eval()
    model = model.to(device)

    sharp_imgA_path = args.sharp_imgA_dir
    sharp_imgB_path = args.sharp_imgB_dir
    blur_imgB_path = args.blur_imgB_dir

    if not Path(sharp_imgA_path).exists() or not Path(sharp_imgB_path).exists():
        print('[ERROR]: images not found!')
        return


    logger.info("================ Strat Extractiton ================ ")
    logger.info('Read image from path: {} and {}'.format(sharp_imgA_path, sharp_imgB_path))     

    image1_BGR = dataset_utils.read_bgr_image(str(sharp_imgA_path))
    image2_BGR = dataset_utils.read_bgr_image(str(sharp_imgB_path))
    image3_BGR = dataset_utils.read_bgr_image(str(blur_imgB_path))

    image1_BGR = dataset_utils.ratio_preserving_resize(image1_BGR, [595,965])
    image2_BGR = dataset_utils.ratio_preserving_resize(image2_BGR, [595,965])
    image3_BGR = dataset_utils.ratio_preserving_resize(image3_BGR, [595,965])
    
    image1_RGB = dataset_utils.bgr_to_rgb(image1_BGR)
    image2_RGB = dataset_utils.bgr_to_rgb(image2_BGR)
    image3_RGB = dataset_utils.bgr_to_rgb(image3_BGR)
        
    image1_RGB_norm = image1_RGB / 255.0
    image2_RGB_norm = image2_RGB / 255.0
    image3_RGB_norm = image3_RGB / 255.0


    # if args.is_debugging:
    #     fig = plt.figure()
    #     rows = 1 ; cols = 3
    #     ax1 = fig.add_subplot(rows, cols, 1)
    #     ax1.imshow(image1_RGB_norm)
    #     ax1.set_title('image_1')
    #     ax1.axis("off")

    #     ax2 = fig.add_subplot(rows, cols, 2)
    #     ax2.imshow(image2_RGB_norm)
    #     ax2.set_title('image_2')
    #     ax2.axis("off")

    #     ax3 = fig.add_subplot(rows, cols, 3)
    #     ax3.imshow(image3_RGB_norm)
    #     ax3.set_title('image_3')
    #     ax3.axis("off")

    #     plt.show()

    kpts1, _ = extract_kpts(image1_RGB_norm, model, device, args, 2000)
    kpts2, _ = extract_kpts(image2_RGB_norm, model, device, args, 2000)
    kpts3, _ = extract_blur_diff_features(image3_RGB_norm, model, device, args)

    # print(kpts2.max())
    logger.info('{} kpts are extracted in image1'.format(kpts1.shape))
    logger.info('{} kpts are extracted in image2'.format(kpts2.shape))
    logger.info('{} kpts are extracted in image3'.format(kpts3.shape))

    # # sift_descriptors1 = compute_sift_desc(image1_BGR, kpts1)
    # # sift_descriptors2 = compute_sift_desc(image1_BGR, kpts2)

    # # print(sift_descriptors1.shape)
    # # print(sift_descriptors2.shape)

    transforms = get_transforms(False)
    model_hardnet_weights = './logs/hardnet/HardNet++.pth'
    model_hardnet = HardNet()
    checkpoint = torch.load(model_hardnet_weights)
    model_hardnet.load_state_dict(checkpoint['state_dict'])
    model_hardnet.eval()
    model_hardnet = model_hardnet.to(device)
    print('Loaded weights: {}'.format(model_hardnet_weights))

    hardnet_descriptors1 = compute_hardnet_desc(kpts1, image1_RGB, model_hardnet, args, device, transforms)
    hardnet_descriptors2 = compute_hardnet_desc(kpts2, image2_RGB, model_hardnet, args, device, transforms)

    print('{} hardnet descriptors in image1'.format(hardnet_descriptors1.shape))
    print('{} hardnet descriptors in image2'.format(hardnet_descriptors2.shape))

    F_matrix, sharpA_corres1, sharpB_corres1 = draw_matcher(
        sharp_imgA_path, sharp_imgB_path,
        kpts1, kpts2, 
        hardnet_descriptors1, hardnet_descriptors2, 
        match_type='FLANN', feature='Ours')

    blurB_corres, sharpA_corres2, sharpB_corres2 = extract_sharp_blur_corres(
        sharpA_corres1, sharpB_corres1, kpts3, dis_thr=8)

    print('**************************')
    print('sharpA corres1: ', sharpA_corres1.shape)
    print('**************************')
    print('sharpA corres2: ', sharpA_corres2.shape)
    print('**************************')
    print('sharpB corres1: ', sharpB_corres1.shape)
    print('**************************')
    print('sharpB corres2: ', sharpB_corres2.shape)
    print('**************************')
    print('blurB corres: ', blurB_corres.shape)
    print('**************************')


    # corres = extract_corres(kpts1, kpts2, F_matrix)
    # print(np.asarray(corres).shape)
    # correspondences = np.asarray(corres)



    
    img_corres_bgr = draw_corres(image1_BGR, image3_BGR, sharpA_corres2, blurB_corres)


    plt.figure(figsize=(7, 2.5), dpi=600, facecolor='w', edgecolor='k')
    plt.axis("off")

    img_corres_rgb = dataset_utils.bgr_to_rgb(img_corres_bgr)

    plt.imshow(img_corres_rgb)
    img_corres_plt = plt.gcf()
    vis_file = str(output_dir)+'/'+'ours_match.pdf'
    img_corres_plt.savefig(vis_file, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()