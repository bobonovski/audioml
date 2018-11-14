#-*- coding: utf8 -*-
"""Fisher Linear Semi-Discriminant Analysis For Spearker Diarization"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import librosa
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist
import numpy as np
from six.moves import xrange

def get_segment_points(audio_data, sample_rate=16000, fft_window=1600, hop_length=800):
    """Get segmentation time points

      Args:
        audio_data: audio time series
        sample_rate: audio sample rate

      Returns:
        segmentation time points sequence
    """
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio_data,
                                sr=sample_rate,
                                n_fft=fft_window,
                                hop_length=hop_length)
    frame_time = librosa.frames_to_time(mfcc, sr=sample_rate)
    feats = normalize(mfcc.T)
    # Clustering with several candidate cluster numbers,
    # the optimal cluster number will be chosen with
    # Silhouette score.
    speaker_num = list(xrange(2, 6))
    best_score = -0.1
    best_labels = []
    best_cluster_num = 0
    for c in speaker_num:
        kmeans = KMeans(c).fit(feats)
        labels = kmeans.labels_
        scores = []
        for s in xrange(c):
            # Compute the cluster size ratio
            if np.sum(labels == s) / float(feats.shape[0]) < 0.1:
                scores.append(0.0)
                continue
            subset = feats[:, labels == s]
            # Compute intra cluster pairwise distances
            intra_dist_avg = np.mean(pdist(subset), axis=1)
            # Compute cross cluster pairwise distances
            cross_dist = None
            for ss in xrange(c):
                if s == ss:
                    continue
                cross_subset = feats[:, labels == ss]
                cross_dist_avg = np.mean(cdist(subset, cross_subset), axis=1)
                if cross_dist is None:
                    cross_dist = cross_dist_avg[:, None]
                else:
                    cross_dist = np.hstack([cross_dist, cross_dist_avg])
            cross_dist_min = np.min(cross_dist, axis=1)
            # Compute silhouette score for each instance
            instance_scores = (cross_dist_min - intra_dist_avg) / \
                               np.maximum(cross_dist_min, intra_dist_avg)
            scores.extend(instance_scores)
        mean_score = np.mean(scores)
        logging.info("silhouette score for %d cluster: %f", c, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_labels = labels
            best_cluster_num = c
    logging.info("Best cluster number %d, best silhouette score %f", best_cluster_num, best_score)
    # Compute segmentation time points
    frame_duration = len(audio_data) / sample_rate
    hop_duration = frame_duration * hop_length
    time_points = np.arange(len(best_labels)) * hop_duration + frame_duration / 2.0
    return list(zip(time_points, labels))

def main():
    """Main entrance function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_filename',
                        type=str, help='audio filename')
    parser.add_argument('--output_filename',
                        type=str, help='output filename for segmentation result')
    args = parser.parse_args()
    time_points = get_segmentation_points(args.audio_filename,
                                          args.sample_rate,
                                          args.fft_window,
                                          args.hop_length)
    output = open(args.output_filename, 'w')
    for tc in time_points:
        output.write('%f,%d\n' % (tc[0], tc[1]))
    output.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
