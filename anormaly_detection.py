"""Audio anormaly detection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position
import seaborn # pylint: disable=unused-import
from six.moves import xrange  # pylint: disable=redefined-builtin

def vanilla_moving_average(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """compute the simple moving average of the audio signal.

    Args:
      series - dataframe with timeseries
      window - rolling window size
      plot_intervals - show confidence intervals
      plot_anomalies - show anomalies
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def compute_energy(vals, hop=320):
    """compute the energy of audio signal

    Args:
      vals: audio signal

    Returns:
      List of audio energy data.
    """
    nhop = len(vals) // hop
    result = []
    for i in xrange(nhop):
        result.append(np.mean(np.power(vals[i*hop:(i+1)*hop], 2.0)))
    return np.array(result)

def main():
    """main entrance function

    """
    data, _ = librosa.load("data/football_sample.mp3", sr=16000)
    energy = compute_energy(librosa.to_mono(data))
    vanilla_moving_average(pd.Series(energy), 4, plot_intervals=True, plot_anomalies=False)

if __name__ == '__main__':
    main()
