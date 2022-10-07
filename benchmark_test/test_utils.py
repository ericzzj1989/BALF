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