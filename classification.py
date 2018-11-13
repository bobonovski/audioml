#-*- coding: utf8 -*-
""""Visualization for audio signal"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import logging
import pickle

import librosa
import numpy as np
import sklearn

def svm_train(data_filename, model_filename):
    """Train SVM classification model

      Args:
        data_file: training data filename
        model_file: output model filename
    """
    # Load data
    data = np.load(data_filename)
    features = data[:, 1:]
    labels = data[:, 0]
    # SVM model
    model = sklearn.svm.SVC(verbose=True)
    model.fit(features, labels)
    # Dump model
    output_file = open(model_filename, 'wb')
    pickle.dump(model, output_file)
    output_file.close()

def svm_predict(model_filename, audio_filename, output_filename):
    """SVM prediction on audio file

      Args:
        model_filename: pretrained SVM model filename
        audio_filename: audio filename for prediction
        output_filename: output filename for prediction result
    """
    # Load pretained model
    model_file = open(model_filename, 'rb')
    model = pickle.load(model_file)
    # Extract audio features
    data, sample_rate = librosa.load(audio_filename, sr=16000)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_fft=1600, hop_length=800)
    prediction = model.predict(mfcc.T)
    # output result
    result = open(output_filename, 'w')
    for c in prediction:
        result.write('%d\n' % int(c))
    result.close()

def main():
    """Main entrance function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=False, help='train model', action='store_true')
    parser.add_argument('--predict',
                        default=False, help='predict audio', action='store_true')
    parser.add_argument('--data_filename',
                        type=str, help='training data filename')
    parser.add_argument('--model_filename',
                        type=str, help='model filename', required=True)
    parser.add_argument('--audio_filename',
                        type=str, help='audio filename')
    parser.add_argument('--output_filename',
                        type=str, help='output filename for prediction result')
    args = parser.parse_args()

    if args.train == True:
        svm_train(args.data_filename, args.model_filename)
    if args.predict == True:
        svm_predict(args.model_filename, args.audio_filename, args.output_filename)

if __name__ == '__main__':
    main()



