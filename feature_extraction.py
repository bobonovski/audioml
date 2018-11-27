#-*- coding: utf8 -*-
""""Audio Feature Extraction"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import librosa
import numpy as np
from six.moves import xrange

def extract_short_term_features(audio_data,
                                sample_rate=16000, window=1600, hop_length=800):
    """Extract various short term features

    Args:
      audio_data: Audio data sequence.
      sample_rate: Sample rate of the audio.
      window: Moving window of audio.
      hop_length: Hop length of moving window.

    Returns:
      array of extracted features.
    """
    fft_window = window
    chroma_stft = librosa.feature.chroma_stft(y=audio_data,
                                              sr=sample_rate,
                                              hop_length=hop_length,
                                              n_fft=fft_window)
    mfcc = librosa.feature.mfcc(y=audio_data,
                                sr=sample_rate,
                                hop_length=hop_length,
                                n_fft=fft_window)
    rmse = librosa.feature.rmse(y=audio_data,
                                frame_length=fft_window,
                                hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data,
                                                          sr=sample_rate,
                                                          n_fft=fft_window,
                                                          hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data,
                                                            sr=sample_rate,
                                                            n_fft=fft_window,
                                                            hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data,
                                                          sr=sample_rate,
                                                          n_fft=fft_window,
                                                          hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data,
                                                          n_fft=fft_window,
                                                          hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_contrast(y=audio_data,
                                                         sr=sample_rate,
                                                         n_fft=fft_window,
                                                         hop_length=hop_length)
    poly_features = librosa.feature.poly_features(y=audio_data,
                                                  sr=sample_rate,
                                                  n_fft=fft_window,
                                                  hop_length=hop_length)
    zero_cross_rate = librosa.feature.zero_crossing_rate(y=audio_data,
                                                         frame_length=fft_window,
                                                         hop_length=hop_length)
    features = np.vstack([chroma_stft, mfcc, rmse, spectral_centroid, \
                          spectral_bandwidth, spectral_contrast, \
                          spectral_flatness, spectral_rolloff, \
                          poly_features, zero_cross_rate])
    return features

def extract_middle_term_features(audio_data, sample_rate,
                                 middle_term_window, middle_term_step,
                                 short_term_window, short_term_step):
    """Extract middle term audio features

    Args:
      audio_data: Audio signal sequence.
      sample_rate: Sample rate of the audio signal.
      middle_term_window: Window size for extracting middle term features.
      middle_term_step: Step size for extracting middle term features.
      short_term_window: Window size for extracting short term features.
      short_term_step: Step size for extracting short term features.

    Returns:
      mt_features: Middle term features.
      st_features: Short term features.
    """
    st_features = extract_short_term_features(audio_data, sample_rate,
                                              short_term_window, short_term_step)
    mt_win = int(round(middle_term_window) / short_term_step)
    mt_step = int(round(middle_term_step) / short_term_step)
    # Compute middle term features
    mt_feature_mean = np.apply_along_axis(rolling_stats, 1, st_features,
                                          mt_win, mt_step, np.mean)
    mt_feature_std = np.apply_along_axis(rolling_stats, 1, st_features,
                                         mt_win, mt_step, np.std)
    mt_features = np.vstack((mt_feature_mean, mt_feature_std))
    return mt_features, st_features

def rolling_stats(data, window, step, func):
    """Compute the rolling statistics of the data

    Args:
      data: Data array.
      window: Rolling window.
      step: Rolling hop length.
      func: Aggregate function to apply to samples in window.

    Returns:
      Rolling statistics sequence.
    """
    result = []
    for k in xrange(0, len(data), step):
        start = k
        end = min(k + window, len(data))
        result.append(func(data[start:end]))
    return np.array(result)

def main():
    """Main entrance function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_filename',
                        type=str, help='Test audio data filename', required=True)
    args = parser.parse_args()

    audio_data, sample_rate = librosa.load(args.audio_filename, sr=16000)
    feats = extract_short_term_features(audio_data,
                                        sample_rate,
                                        window=800,
                                        hop_length=400)
    print(feats.shape)
    mt_feats, st_feats = extract_middle_term_features(audio_data, sample_rate,
                                                      1600, 800, 800, 400)
    print(mt_feats.shape)
    print(st_feats.shape)
if __name__ == '__main__':
    main()
