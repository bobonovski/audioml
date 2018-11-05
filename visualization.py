""""Visualization for audio signal"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import librosa
import librosa.display
import matplotlib; matplotlib.use('TkAgg') # pylint: disable=multiple-statements
import matplotlib.pyplot as plt
import numpy as np
import seaborn # pylint: disable=unused-import

def load_data(filename, sample_rate, offset, duration):
    """Load audio data in specified sample rate

    Args:
        filename: filename of the audio
        sample_rate: sample rate
        offset: offset of audio to load
        duration: duration of audio to load

    Returns:
        Numpy array of float point audio data.
    """
    audio, _ = librosa.load(filename, sample_rate, True, offset, duration)
    return audio

def visualize(argv):
    """Visualization driver

    """
    audio = load_data(argv.audio_file, argv.sample_rate, argv.offset, argv.duration)
    # RMSE energy
    rmse = librosa.feature.rmse(y=audio, frame_length=argv.fft_window, hop_length=argv.hop_length)
    # melspectrogram
    melspectrogram = librosa.feature.melspectrogram(y=audio,
                                                    sr=argv.sample_rate,
                                                    n_fft=argv.fft_window,
                                                    hop_length=argv.hop_length)
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=argv.sample_rate, n_mfcc=argv.mfcc_num)
    # plot RMSE energy
    axe = plt.subplot(3, 1, 1)
    axe.plot(np.arange(len(rmse.flatten())), rmse.flatten())
    axe.set_title('RMSE Energy')
    # plot melspectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(melspectrogram)
    # plot MFCC
    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfcc)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file',
                        type=str, help='audio file for visualization', required=True)
    parser.add_argument('--sample_rate',
                        type=int, default=16000, help='sample rate of the audio file')
    parser.add_argument('--offset',
                        type=float, default=0.0, help='offset of audio file to load')
    parser.add_argument('--duration',
                        type=float, default=10.0, help='duration of audio file to load')
    parser.add_argument('--fft_window',
                        type=int, default=1024, help='length of FFT window')
    parser.add_argument('--hop_length',
                        type=int, default=256, help='hop length of moving window')
    parser.add_argument('--mfcc_num',
                        type=int, default=20, help='number of MFCC to return')
    args = parser.parse_args()

    visualize(args)
