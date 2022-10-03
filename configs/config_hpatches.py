import argparse
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='hsequences detection extraction')

    parser.add_argument('--exper_name', type=str, required=True)
    parser.add_argument('--test_cfg_file', type=str, default='configs/gopro_test_detection.yaml')
    parser.add_argument('--results_detection_dir', type=str, default='results_hsequences_detection',
                        help='Path for saving extraction results')
    parser.add_argument('--ckpt_file', type=str, 
                        help='The path to the checkpoint file to load the detector weights.', required=True)
    parser.add_argument('--hsequences_list_file', type=str, 
                        help='File containing the image paths for extracting features.', required=True)

    args = parser.parse_args()

    cfg = common_utils.get_cfg_from_yaml_file(args.test_cfg_file)

    return args, cfg


def parse_eval_config():
    parser = argparse.ArgumentParser(description='HSequences Compute Repeatability')

    parser.add_argument('--data_dir', type=str, default='data/hpatches-sequences-release/',
                        help='The root path to HSequences dataset.')
    parser.add_argument('--results_bench_dir', type=str, default='results_hsequences_bench/',
                        help='The output path to save the results.')
    parser.add_argument('--results_detection_dir', type=str,
                        help='Path for saving extraction results', required=True)

    parser.add_argument('--split', type=str, default='full',
                        help='The name of the HPatches (HSequences) split. Use full, debug_view, debug_illum, view or illum.')
    parser.add_argument('--split_path', type=str, default='benchmark_test/splits.json',
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

    args = parser.parse_args()

    return args