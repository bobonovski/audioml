#-*- coding: utf8 -*-
"""Fisher Linear Semi-Discriminant Analysis For Spearker Diarization"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import errno
import os
import pickle

import librosa
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np
from six.moves import xrange
from feature_extraction import extract_middle_term_feature

class KNN:
    """k nearest neighbours classification"""
    def __init__(self, features, labels, k):
        self.features = features
        self.labels = labels
        self.k = k

    def classify(self, data):
        """Classify new data.

        Args:
          data: Test data point.

        Returns:
          Class label of the data point.
        """
        num_classes = np.unique(self.labels).shape[0]
        distances = cdist(self.features, data.reshape(1, data.shape[0]))
        indices = np.argsort(distances.flatten())
        prob = np.zeros(num_classes)
        for i in xrange(num_classes):
            top_k_labels = self.labels[indices[0:self.k]]
            prob[i] = np.sum(top_k_labels == i) / float(self.k)
        return (np.argmax(prob), prob)

def load_model(model_path):
    """Load pre-trained kNN model.

    Args:
      model_path: File path of pre-trained model.

    Raises:
      FileNotFoundError: If the file is not found.
    """
    try:
        model_file = open(model_path, 'rb')
    except IOError:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    try:
        features = pickle.load(model_file)
        labels = pickle.load(model_file)
        mean = pickle.load(model_file)
        std = pickle.load(model_file)
        classes = pickle.load(model_file)
        # k in kNN
        k = pickle.load(model_file)
        middle_term_window = pickle.load(model_file)
        middle_term_step = pickle.load(model_file)
        short_term_window = pickle.load(model_file)
        short_term_step = pickle.load(model_file)
        beat = pickle.load(model_file)
    except pickle.PickleError:
        model_file.close()

    features = np.array(features)
    labels = np.array(labels)
    mean = np.array(mean)
    std = np.array(std)

    classifier = KNN(features, labels, k)
    return classifier, mean, std, classes, \
           middle_term_window, middle_term_step, \
           short_term_window, short_term_step, beat

def get_segment_points(audio_data, sample_rate=16000, model_path='model'):
    """Get segmentation time points

    Args:
      audio_data: audio time series
      sample_rate: audio sample rate

    Returns:
      segmentation time points sequence
    """
    # Load pre-trained audio classifier
    # general_model_path = os.path.join(model_path, "knnSpeakerAll")
    # gender_model_path = os.path.join(model_path, "knnSpeakerFemaleMale")
    # general_model_info = self.load_model(general_model_path)
    # gender_model_info = self.load_model(gender_model_path)
    # general_classifier, general_mean, general_std, general_classes = gender_model_info[0:4]
    # gender_classifier, gender_mean, gender_std, gender_classes = gender_model_info[0:4]
    
    # Extract features
    mt_features, st_features = extract_middle_term_feature(audio_data)
    mt_features_scaled = scale(mt_features)
    # Clustering with several candidate cluster numbers,
    # the optimal cluster number will be chosen with
    # Silhouette score.
    speaker_num = list(xrange(2, 6))
    best_score = -0.1
    best_labels = []
    best_cluster_num = 0
    for c in speaker_num:
        logging.info("Kmeans with %d cluster", c)
        kmeans = KMeans(c).fit(feats)
        labels = kmeans.labels_
        for lb in xrange(c):
            logging.info("  Cluster %d, number of samples %d", lb, np.sum(labels == lb))
        scores = []
        for s in xrange(c):
            # Compute the cluster size ratio
            if np.sum(labels == s) / float(feats.shape[0]) < 0.1:
                scores.append(0.0)
                continue
            subset = feats[labels == s, :]
            # Compute intra cluster pairwise distances
            intra_dist_avg = np.mean(squareform(pdist(subset)), axis=1)
            # Compute cross cluster pairwise distances
            cross_dist = None
            for ss in xrange(c):
                if s == ss:
                    continue
                cross_subset = feats[labels == ss, :]
                cross_dist_avg = np.mean(cdist(subset, cross_subset), axis=1)
                if cross_dist is None:
                    cross_dist = cross_dist_avg[:, None]
                else:
                    cross_dist = np.hstack([cross_dist, cross_dist_avg[:, None]])
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
    hop_duration = hop_length / sample_rate
    window_duration = fft_window / sample_rate
    time_points = np.arange(len(best_labels)) * hop_duration + window_duration / 2.0
    return list(zip(time_points, best_labels))

def main():
    """Main entrance function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_filename',
                        type=str, help='audio filename')
    parser.add_argument('--sample_rate',
                        type=int, help='sample_rate', default=16000)
    parser.add_argument('--fft_window',
                        type=int, help='fft window', default=1600)
    parser.add_argument('--hop_length',
                        type=int, help='hop_length', default=800)
    parser.add_argument('--output_filename',
                        type=str, help='output filename for segmentation result')
    args = parser.parse_args()
    # Load audio data
    audio_data, _ = librosa.load(args.audio_filename, sr=args.sample_rate)
    time_points = get_segment_points(audio_data,
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
