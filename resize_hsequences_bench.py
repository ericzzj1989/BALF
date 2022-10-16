import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from configs import config_hpatches
from utils import common_utils
from utils.logger import logger
from datasets import Resize_HSequences
from benchmark_test import test_utils, repeatability_tools


def hsequences_metrics():
    args = config_hpatches.parse_resize_eval_config()

    print('Evaluate {} sequences with size {}'.format(args.split, args.resize_shape))
    common_utils.check_directory(args.results_bench_dir)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = Path(args.results_bench_dir, args.results_detection_dir.split('/')[1], start_time, 'split-{}_overlap{}_top-k{}_pixel-threshold{}'.format(args.split, args.overlap, args.top_k_points, args.pixel_threshold))
    common_utils.check_directory(output_dir)

    logger.initialize(args, output_dir)

    if args.output_img:
        visualization_dir = Path(output_dir, 'visualization_detection')
        common_utils.check_directory(visualization_dir)

    dataloader = Resize_HSequences.Resize_HSequences(args.data_dir, args.split, args.split_path, args)
    metrics_results = test_utils.create_resize_metrics_results(args.split, args.top_k_points, args.pixel_threshold)

    counter_sequences = 0
    iterate = tqdm(range(len(dataloader.sequences)), total=len(dataloader.sequences), desc="HSequences Eval")

    # for sequence_index in tqdm(range
    for sequence_index in iterate:
        sequence_data = dataloader.get_sequence_data(sequence_index)

        counter_sequences += 1

        sequence_name = sequence_data['sequence_name']
        im_src_BGR = sequence_data['im_src_BGR']
        images_dst_BGR = sequence_data['images_dst_BGR']
        homographies = sequence_data['homographies']

        print('\n Computing '+sequence_name+' sequence {} / {} \n'.format(counter_sequences, len(dataloader.sequences)))
        
        for im_dst_index in tqdm(range(len(images_dst_BGR))):
            pts_src_file = Path(args.results_detection_dir, '{}/1.ppm.kpt.npz'.format(sequence_data['sequence_name']))
            pts_dst_file =Path(args.results_detection_dir, '{}/{}.ppm.kpt.npz'.format(sequence_data['sequence_name'], im_dst_index+2))

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

            pts_src_yxr = np.asarray(list(map(lambda x: [x[1], x[0], x[3]], pts_src)))
            pts_dst_yxr = np.asarray(list(map(lambda x: [x[1], x[0], x[3]], pts_dst)))

            homography = homographies[im_dst_index]

            repeatability_results = repeatability_tools.compute_resize_repeatability(
                keypoints=pts_src_yxr, warped_keypoints=pts_dst_yxr, h=homography,
                shape_src=im_src_BGR.shape[:2], shape_dst=images_dst_BGR[im_dst_index].shape[:2],
                keep_k_points=args.top_k_points, distance_thresh=args.pixel_threshold
            )

            
            if args.output_img:
                detection_im_src_BGR = test_utils.draw_keypoints(im_src_BGR, pts_src[:,:2], radius=1)
                detection_im_dst_BGR = test_utils.draw_keypoints(images_dst_BGR[im_dst_index], pts_dst[:,:2], radius=1)

                src_pts_common_num = repeatability_results['common_src_num']
                dst_pts_common_num = repeatability_results['common_dst_num']
                src_pts_rep_num = repeatability_results['rep_src_num']
                dst_pts_rep_num = repeatability_results['rep_dst_num']

                test_utils.plot_imgs(
                    [detection_im_src_BGR.astype(np.uint8), detection_im_dst_BGR.astype(np.uint8)],
                    titles=['src pts: {}/{}/{}'.format(src_pts_rep_num,src_pts_common_num,src_pts_raw_num),
                            'dst pts: {}/{}/{}'.format(dst_pts_rep_num, dst_pts_common_num,dst_pts_raw_num)], dpi=150)
                plt.suptitle('{} {} / {} - {}, rep {:.2f}, loc_err {:.4f}'.format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                    repeatability_results['repeatability'], repeatability_results['localization_err']))

                plt.tight_layout()
                vis_file = str(visualization_dir)+'/'+sequence_name+'_'+str(im_dst_index+2)+'.png'
                plt.savefig(vis_file, dpi=150, bbox_inches='tight')


            metrics_results['repeatability'].append(
                repeatability_results['repeatability'])
            metrics_results['localization_err'].append(
                repeatability_results['localization_err'])
            metrics_results['common_src_num'].append(
                repeatability_results['common_src_num'])
            metrics_results['common_dst_num'].append(
                repeatability_results['common_dst_num'])
            metrics_results['rep_src_num'].append(
                repeatability_results['rep_src_num'])
            metrics_results['rep_dst_num'].append(
                repeatability_results['rep_dst_num'])


            ## logging
            iterate.set_description("{}  {} / {} - {} rep {:.2f} , loc_err {:.5f}, common_src {:d} , common_dst {:d}, rep_src {:d}, rep_dst {:d}, avg_rep: {:.4f}"
                .format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                    repeatability_results['repeatability'], repeatability_results['localization_err'], repeatability_results['common_src_num'], 
                    repeatability_results['common_dst_num'], repeatability_results['rep_src_num'], repeatability_results['rep_dst_num'],
                    np.array(metrics_results['repeatability']).mean()
            ))
            logger.info("{}  {} / {} - {} rep {:.2f} , loc_err {:.5f}, common_src {:d} , common_dst {:d}, rep_src {:d}, rep_dst {:d}"
                .format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                    repeatability_results['repeatability'], repeatability_results['localization_err'], repeatability_results['common_src_num'], 
                    repeatability_results['common_dst_num'], repeatability_results['rep_src_num'], repeatability_results['rep_dst_num']
            ))

    # average the results
    repeatability_avg = np.array(metrics_results['repeatability']).mean()
    localization_err_avg = np.array(metrics_results['localization_err']).mean()
    common_src_num_avg = np.array(metrics_results['common_src_num']).mean()
    common_dst_num_avg = np.array(metrics_results['common_dst_num']).mean()
    rep_src_num_avg = np.array(metrics_results['rep_src_num']).mean()
    rep_dst_num_avg = np.array(metrics_results['rep_dst_num']).mean()


    logger.info('\n## Overlap @{0}:\n \
           ## top_k @{1}:\n \
           ## pixel_threshold @{2}:\n \
           #### Repeatability: {3:.4f}\n \
           #### Localization Error: {4:.4f}\n \
           #### Common Src Num: {5:.4f}\n \
           #### Common Dst Num: {6:.4f}\n \
           #### Repeated Src Num: {7:.4f}\n \
           #### Repeated Dst Num: {8:.4f}'.format(
           args.overlap, args.top_k_points, args.pixel_threshold,
           repeatability_avg, localization_err_avg, common_src_num_avg, common_dst_num_avg, rep_src_num_avg, rep_dst_num_avg
    ))

    metrics_file = Path(output_dir, 'metrics')
    np.savez(
        metrics_file, rep=repeatability_avg, loc_err=localization_err_avg,
        common_src=common_src_num_avg, common_dst=common_dst_num_avg,
        rep_src=rep_src_num_avg, rep_dst=rep_dst_num_avg)


if __name__ == '__main__':
    hsequences_metrics()