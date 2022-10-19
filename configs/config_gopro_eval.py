import argparse
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='gopro evaluation detection extraction')

    parser.add_argument('--exper_name', type=str, required=True)
    parser.add_argument('--test_cfg_file', type=str, default='configs/detection_repeatability.yaml')
    parser.add_argument('--results_detection_dir', type=str, default='results_gopro_src_blur_dst_sharp_detection',
                        help='Path for saving extraction results')
    parser.add_argument('--ckpt_file', type=str, 
                        help='The path to the checkpoint file to load the detector weights.', required=True)
    parser.add_argument('--gopro_list_file', type=str, 
                        help='File containing the image paths for extracting features.', required=True)

    parser.add_argument('--split', type=str, default='src_blur_dst_sharp',
                        help='The gopro comparison type.', required=True)
    parser.add_argument('--resize_image', type=bool, default=False,
                        help='Extract detections in the resized image')
    parser.add_argument('--cell_size', type=int, default=8,
                        help='Pixel shuffle.')
    parser.add_argument('--nms', type=str, default='nms_fast',
                        help='apply_nms, apply_nms_fast, nms_fast, box_nms (default: nms_fast)')
    parser.add_argument('--nms_size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')
    parser.add_argument('--num_points', type=int, default=10000,
                        help='The number of desired features to extract.')
    parser.add_argument('--border_size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')
    parser.add_argument('--order_coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use yxsr or xysr.')
    parser.add_argument('--heatmap_confidence_threshold', type=float, default=0.015,
                        help='Keypoints confidence threshold.')
    parser.add_argument('--sub_pixel', type=bool, default=True,
                        help='Extract subpixel detection')
    parser.add_argument('--patch_size', type=int, default=5,
                        help='subpixel patch size.')
    parser.add_argument('--is_debugging', type=bool, default=False,
                        help='Set variable to True if you desire to check read image.')

    args = parser.parse_args()

    cfg = common_utils.get_cfg_from_yaml_file(args.test_cfg_file)

    return args, cfg


def parse_eval_config():
    parser = argparse.ArgumentParser(description='GOPRO Compute Repeatability')

    parser.add_argument('--data_dir', type=str, default='data/gopro_exp2_with_hom_sharp/',
                        help='The root path to GOPRO test dataset.')
    parser.add_argument('--results_bench_dir', type=str, default='results_gopro_src_blur_dst_sharp_bench/',
                        help='The output path to save the results.')
    parser.add_argument('--results_detection_dir', type=str,
                        help='Path for saving extraction results', required=True)

    parser.add_argument('--split', type=str, default='src_blur_dst_sharp',
                        help='The gopro comparison type.', required=True)
    parser.add_argument('--split_path', type=str, default='benchmark_test/gopro_test_splits.json',
                        help='The path to the split json file.')
    parser.add_argument('--top_k_points', type=int, default=1000,
                        help='The number of top points to use for evaluation. Set to None to use all points')
    parser.add_argument('--overlap', type=float, default=0.6,
                        help='The overlap threshold for a correspondence to be considered correct.')
    parser.add_argument('--pixel_threshold', type=int, default=5,
                        help='The distance of pixels for a matching correspondence to be considered correct.')

    parser.add_argument('--dst_to_src_evaluation', type=bool, default=True,
                        help='Order to apply homography to points. Use True for dst to src, False otherwise.')
    parser.add_argument('--order_coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use either xysr or yxsr.')

    parser.add_argument('--output_img', type=bool, default=False,
                        help='Output the detection in the image.')

    args = parser.parse_args()

    return args