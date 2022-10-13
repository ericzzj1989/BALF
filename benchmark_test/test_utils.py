import cv2
import matplotlib .pyplot as plt


def create_results():
       return {
       'num_features': [],
       'rep_single_scale': [],
       'rep_multi_scale': [],
       'num_points_single_scale': [],
       'num_points_multi_scale': [],
       'error_overlap_single_scale': [],
       'error_overlap_multi_scale': [],
       'mma': [],
       'mma_corr': [],
       'num_matches': [],
       'num_mutual_corresp': [],
       'avg_mma': []
    }

def create_metrics_results(sequences, top_k, overlap, pixel_threshold):
    
    results = create_results()
    results['sequences'] = sequences
    results['top_k'] = top_k
    results['overlap'] = overlap
    results['pixel_threshold'] = pixel_threshold
    return results


def draw_keypoints(im_BGR, corners, color=(0,255,0), radius=5):
    for idx, center in enumerate(corners):
        cv2.circle(im_BGR, tuple(center.astype(int)), radius, color, thickness=-1)
    return im_BGR


def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()