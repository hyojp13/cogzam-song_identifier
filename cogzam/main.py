# pyright: reportMissingImports=false, reportUnusedVariable=warning, reportUntypedBaseClass=error
from .database import Database
from .fingerprinting import fingerprints
from .peaks import find_neighborhood, find_min_amp, local_peak_locations

import numpy as np
import librosa
# import matplotlib.pyplot as plt
from IPython.display import Audio
from typing import Union, Callable, Tuple, List
from pathlib import Path

import matplotlib.mlab as mlab

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure

import os

import pickle
from IPython.display import Audio
from .Analog_to_Digital import analog_to_digital
from microphone import record_audio

# %matplotlib notebook

database = Database()


def generate_fingerprints(samples, sampling_rate):
    # perform rfft
    N = len(samples)
    T = N / sampling_rate

    times = np.arange(N) / sampling_rate
    c_k = np.fft.rfft(samples)
    # convert ck (complex Fourier coeff) -> |ak| (real-valued coeff)
    amps = np.abs(c_k) / N
    amps[1: (-1 if N % 2 == 0 else None)] *= 2

    # convert k = 0, 1, ... to freq = 0, 1/T, 2/T, ..., (int(N/2) + 1)/T
    freq = np.arange(len(amps)) / T

    spectrogram, freqs, times = mlab.specgram(
        samples,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2)
    )

    neighborhood = find_neighborhood(2, 1, 20)
    min_amp = find_min_amp(spectrogram)

    peaks = local_peak_locations(spectrogram, neighborhood, min_amp)
    song_fingerprints, times = fingerprints(15, peaks)

    return (song_fingerprints, times)


def import_song(path, song_name):
    # sample the audio file
    samples, sampling_rate = librosa.load(path, sr=44100, mono=True)

    song_fingerprints, times = generate_fingerprints(samples, sampling_rate)

    database.store_fingerprints(song_fingerprints, song_name, times)


def load_directory(directory_path):
    song_directory = directory_path

    for filename in os.listdir(song_directory):
        print("Importing: ", filename)
        file_path = song_directory + "\\" + filename
        import_song(file_path, filename)


def match_pickle(clips_directory, clips_name):
    with open(clips_directory + clips_name, mode="rb") as clips_file:
        clips = pickle.load(clips_file)

    for sample in clips:
        sample_fingerprints, times = generate_fingerprints(sample, 44100)

        counts = database.search_song(sample_fingerprints, times)

        print(counts.most_common(3))

    Audio(clips[0], rate=44100)


def match_sample(sample):

    sample_fingerprints, times = generate_fingerprints(sample, 44100)
    counts = database.search_song(sample_fingerprints, times)

    return counts.most_common(1)[0]


def microphone_match():
    # record samples
    listen_time = 10

    samples, sample_rate = record_audio(listen_time)
    print("Recording...")
    samples = np.hstack([np.frombuffer(i, np.int16) for i in samples])

    # times, digitial_signal = analog_to_digital(samples, sampling_rate=44100, bit_depth=16, duration=10)

    result = match_sample(samples)

    return result


database.load_database()
