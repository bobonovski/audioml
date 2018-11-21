#-*- coding: utf8 -*-
""""Visualization for audio signal"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import logging

import librosa
import numpy as np

def extract_features(filename):
    """Extract audio features of the file

      Args:
        filename: audio filename
      Returns:
        feature matrix [time_step, features]
    """
    # Load audio data
    data, sample_rate = librosa.load(filename, sr=16000)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_fft=1600, hop_length=800)
    return mfcc.T

def extract(positive_dir, negative_dir, output_filename):
    """Extract audio features of the directories

      Args:
        positive_dir: directory for positive audio files
        negative_dir: directory for negative audio files
        output_dir: directory for output file of extracted features
    """
    total_feats = None
    # Extract features of positive examples
    for i, pos in enumerate(os.listdir(positive_dir)):
        logging.info("processing %d positive file", i)
        feats = extract_features(os.path.join(positive_dir, pos))
        labels = np.ones(feats.shape[0])[:, None]
        feats = np.hstack([labels, feats])
        if total_feats is None:
            total_feats = feats
        else:
            total_feats = np.vstack([total_feats, feats])
    # Extract features of negative examples
    for i, neg in enumerate(os.listdir(negative_dir)):
        logging.info("processing %d negative file", i)
        feats = extract_features(os.path.join(negative_dir, neg))
        labels = np.zeros(feats.shape[0])[:, None]
        feats = np.hstack([labels, feats])
        if total_feats is None:
            total_feats = feats
        else:
            total_feats = np.vstack([total_feats, feats])
    positive_count = np.sum(total_feats[:, 0] == 1)
    negative_count = np.sum(total_feats[:, 0] == 0)
    logging.info("Features contain %d positive examples and %d negative examples", 
                 positive_count, negative_count)
    # Random shuffle feature matrix
    np.random.shuffle(total_feats)
    # Dump features
    np.save(output_filename, total_feats)

def main():
    """Main entrance function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_dir',
                        type=str, help='directory for positive examples', required=True)
    parser.add_argument('--negative_dir',
                        type=str, help='directory for negative examples', required=True)
    parser.add_argument('--output_filename',
                        type=str, help='output filename', required=True)
    args = parser.parse_args()

    extract(args.positive_dir, args.negative_dir, args.output_filename)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
