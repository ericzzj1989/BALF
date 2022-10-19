import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from configs import config_gopro_eval
from utils import common_utils
from utils.logger import logger
from datasets import GOPRO_TEST
from benchmark_test import test_utils, geometry_tools, repeatability_tools


def hsequences_metrics():
    args = config_gopro_eval.parse_eval_config()

    print('Evaluate {} sequences'.format(args.split))
    if args.split == 'src_sharp_dst_blur':
        args.results_bench_dir = 'results_gopro_src_sharp_dst_blur_bench/'
    common_utils.check_directory(args.results_bench_dir)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = Path(args.results_bench_dir, args.results_detection_dir.split('/')[1], start_time, 'split-{}_overlap{}_top-k{}_pixel-threshold{}'.format(args.split, args.overlap, args.top_k_points, args.pixel_threshold))
    common_utils.check_directory(output_dir)

    logger.initialize(args, output_dir)

    if args.output_img:
        visualization_dir = Path(output_dir, 'visualization_detection')
        common_utils.check_directory(visualization_dir)

    dataloader = GOPRO_TEST.GOPRO_test(args.data_dir, args.split, args.split_path)
    metrics_results = test_utils.create_metrics_results(args.split, args.top_k_points, args.overlap, args.pixel_threshold)

    counter_sequences = 0
    iterate = tqdm(range(len(dataloader.sequences)), total=len(dataloader.sequences), desc="HSequences Eval")

    # for sequence_index in tqdm(range
    for sequence_index in iterate:
        sequence_data = dataloader.get_sequence_data(sequence_index)

        counter_sequences += 1

        sequence_name = sequence_data['sequence_name']
        im_src_BGR = sequence_data['im_src_BGR']
        images_dst_BGR = sequence_data['images_dst_BGR']
        h_src_2_dst = sequence_data['h_src_2_dst']
        h_dst_2_src = sequence_data['h_dst_2_src']

        print('\n Computing '+sequence_name+' sequence {} / {} \n'.format(counter_sequences, len(dataloader.sequences)))
        
        for im_dst_index in tqdm(range(len(images_dst_BGR))):
            mask_src, mask_dst = geometry_tools.create_common_region_masks(
                h_dst_2_src[im_dst_index], im_src_BGR.shape, images_dst_BGR[im_dst_index].shape
            )

            if args.split == 'src_blur_dst_sharp':
                pts_src_file = Path(args.results_detection_dir, '{}/blur_gamma/1.png.kpt.npz'.format(sequence_data['sequence_name']))
                pts_dst_file =Path(args.results_detection_dir, '{}/sharp/{}.png.kpt.npz'.format(sequence_data['sequence_name'], im_dst_index+1))
            elif args.split == 'src_sharp_dst_blur':
                pts_src_file = Path(args.results_detection_dir, '{}/sharp/1.png.kpt.npz'.format(sequence_data['sequence_name']))
                pts_dst_file =Path(args.results_detection_dir, '{}/blur_gamma/{}.png.kpt.npz'.format(sequence_data['sequence_name'], im_dst_index+1))

            if not pts_src_file.exists():
                logger.info("Could not find the file: " + str(pts_src_file))
                continue

            if not pts_dst_file.exists():
                logger.info("Could not find the file: " + str(pts_dst_file))
                continue


            pts_src = np.load(pts_src_file, allow_pickle=True)['kpts']
            pts_dst = np.load(pts_dst_file, allow_pickle=True)['kpts']

            src_pts_raw_num = len(pts_src)
            dst_pts_raw_num = len(pts_dst)

            if args.order_coord == 'xysr':
                pts_src = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_src)))
                pts_dst = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_dst)))

            idx_src = repeatability_tools.check_common_points(pts_src, mask_src)
            if idx_src.size == 0:
                continue
            pts_src = pts_src[idx_src]

            idx_dst = repeatability_tools.check_common_points(pts_dst, mask_dst)
            if idx_dst.size == 0:
                continue
            pts_dst = pts_dst[idx_dst]

            if args.top_k_points:
                idx_src = repeatability_tools.select_top_k(pts_src, args.top_k_points)
                pts_src = pts_src[idx_src]

                idx_dst = repeatability_tools.select_top_k(pts_dst, args.top_k_points)
                pts_dst = pts_dst[idx_dst]

            pts_src = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_src)))
            pts_dst = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_dst)))

            pts_src_to_dst = geometry_tools.apply_homography_to_points(
                pts_src, h_src_2_dst[im_dst_index])

            pts_dst_to_src = geometry_tools.apply_homography_to_points(
                pts_dst, h_dst_2_src[im_dst_index])

            if args.dst_to_src_evaluation:
                points_src = pts_src
                points_dst = pts_dst_to_src
            else:
                points_src = pts_src_to_dst
                points_dst = pts_dst

            repeatability_results = repeatability_tools.compute_repeatability(points_src, points_dst, overlap_err=1-args.overlap,
                                                                    dist_match_thresh=args.pixel_threshold)

            if args.output_img:
                detection_im_src_BGR = test_utils.draw_keypoints(im_src_BGR, pts_src[:,:2], radius=5)
                detection_im_dst_BGR = test_utils.draw_keypoints(images_dst_BGR[im_dst_index], pts_dst[:,:2], radius=5)

                test_utils.plot_imgs(
                    [detection_im_src_BGR.astype(np.uint8), detection_im_dst_BGR.astype(np.uint8)],
                    titles=['src pts: {}/{}'.format(len(pts_src),src_pts_raw_num), 'dst pts: {}/{}'.format(len(pts_dst),dst_pts_raw_num)], dpi=150)
                plt.suptitle('{} {} / {} - {}, rep_s {:.2f}, min_features {:d}, matches {:d}'.format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                    repeatability_results['rep_single_scale'], repeatability_results['total_num_points'], repeatability_results['possible_matches']))

                plt.tight_layout()
                vis_file = str(visualization_dir)+'/'+sequence_name+'_'+str(im_dst_index+1)+'.png'
                plt.savefig(vis_file, dpi=150, bbox_inches='tight')


            metrics_results['rep_single_scale'].append(
                repeatability_results['rep_single_scale'])
            metrics_results['rep_multi_scale'].append(
                repeatability_results['rep_multi_scale'])
            metrics_results['num_points_single_scale'].append(
                repeatability_results['num_points_single_scale'])
            metrics_results['num_points_multi_scale'].append(
                repeatability_results['num_points_multi_scale'])
            metrics_results['error_overlap_single_scale'].append(
                repeatability_results['error_overlap_single_scale'])
            metrics_results['error_overlap_multi_scale'].append(
                repeatability_results['error_overlap_multi_scale'])
            metrics_results['num_features'].append(
                repeatability_results['total_num_points'])
            metrics_results['num_matches'].append(
                repeatability_results['possible_matches'])

            ## logging
            iterate.set_description("{}  {} / {} - {} rep_s {:.2f} , rep_m {:.2f}, p_s {:d} , p_m {:d}, eps_s {:.2f}, eps_m {:.2f}, min_features {:d}, matches {:d}, avg rep_s: {:0.4f}"
                .format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                    repeatability_results['rep_single_scale'], repeatability_results['rep_multi_scale'], repeatability_results['num_points_single_scale'], 
                    repeatability_results['num_points_multi_scale'], repeatability_results['error_overlap_single_scale'],
                    repeatability_results['error_overlap_multi_scale'], repeatability_results['total_num_points'], repeatability_results['possible_matches'],
                    np.array(metrics_results['rep_single_scale']).mean()
            ))
            logger.info("{} {} / {} - {} src_num: {:d}/{:d} dst_num: {:d}/{:d} rep_s {:.2f} , rep_m {:.2f}, p_s {:d} , p_m {:d}, eps_s {:.2f}, eps_m {:.2f}, min_features {:d}, matches {:d}"
                .format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index, len(pts_src),src_pts_raw_num, len(pts_dst),dst_pts_raw_num,
                    repeatability_results['rep_single_scale'], repeatability_results['rep_multi_scale'], repeatability_results['num_points_single_scale'], 
                    repeatability_results['num_points_multi_scale'], repeatability_results['error_overlap_single_scale'],
                    repeatability_results['error_overlap_multi_scale'], repeatability_results['total_num_points'], repeatability_results['possible_matches']
            ))

    # average the results
    rep_single = np.array(metrics_results['rep_single_scale']).mean()
    rep_multi = np.array(metrics_results['rep_multi_scale']).mean()
    error_overlap_s = np.array(metrics_results['error_overlap_single_scale']).mean()
    error_overlap_m = np.array(metrics_results['error_overlap_multi_scale']).mean()
    num_features = np.array(metrics_results['num_features']).mean()
    num_matches = np.array(metrics_results['num_matches']).mean()


    logger.info('\n## Overlap @{0}:\n \
           ## top_k @{1}:\n \
           ## pixel_threshold @{2}:\n \
           #### Rep. Multi: {3:.4f}\n \
           #### Rep. Single: {4:.4f}\n \
           #### Overlap Multi: {5:.4f}\n \
           #### Overlap Single: {6:.4f}\n \
           #### Num Feats: {7:.4f}\n \
           #### Num Matches: {8:.4f}'.format(
           args.overlap, args.top_k_points, args.pixel_threshold,
           rep_multi, rep_single, error_overlap_s, error_overlap_m, num_features, num_matches
    ))

    metrics_file = Path(output_dir, 'metrics')
    np.savez(metrics_file, rep_single=rep_single, rep_multi=rep_multi, error_overlap_s=error_overlap_s, error_overlap_m=error_overlap_m, num_features=num_features)


if __name__ == '__main__':
    hsequences_metrics()