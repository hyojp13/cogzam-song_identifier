import numpy as np
from typing import Union, Callable, Tuple
from pathlib import Path

def analog_signal(times: np.ndarray) -> np.ndarray:
    """
    Given an array of times, returns the value of an analog signal at those times.
    
    Parameters
    ----------
    times : numpy.ndarray
        The time(s) at which to evaluate the analog signal.
    
    Returns
    -------
    numpy.ndarray
        The value of the analog signal at the given times.
    """
    return 1 / (1 + np.exp(-10 * (times - 1))) 

def temporal_sampler(
    signal: Callable[[np.ndarray], np.ndarray], *, duration: float, sampling_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts samples from an analog signal according to the specified sampling rate,
    returning the times and the corresponding samples extracted at those times.

    Parameters
    ----------
    signal : Callable[[ndarray], ndarray]
        Another Python function (i.e. a "callable"), which behaves like f(t)
        and accepts a time value (in seconds) as an input and returns a
        measurement (e.g. in volts) as an output. You can expect this to behave like
        a vectorized function i.e. it can be passed a NumPy-array of input times
        and it will return a corresponding array of measurements.

    duration : float
        The duration of the signal, specified in seconds (a non-negative float)

    sampling_rate : float
        The sampling rate specified in Hertz.

    Returns
    -------
    (times, samples) : Tuple[ndarray, ndarray]
        The shape-(N,) array of times and the corresponding shape-(N,) array
        samples extracted from the analog signal

    """
    N_samples = np.floor(sampling_rate * duration) + 1

    # shape-(N,) array of times at which we sample the analog signal
    times = np.arange(N_samples) * (1 / sampling_rate)  # seconds

    # shape-(N,) array of samples extracted from the analog signal
    samples = signal(times)

    return times, samples

def quantize(samples: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Given an array of N samples and a bit-depth of M, return the array of
    quantized samples derived from the domain [samples.min(), samples.max()]
    that has been quantized into 2**M evenly-spaced values.

    Parameters
    ----------
    samples : numpy.ndarray, shape-(N,)
        An array of N samples

    bit_depth: int
        The bit-depth, M, used to quantize the samples among
        2**M evenly spaced values spanning [samples.min(), samples.max()].

    Returns
    -------
    quantized_samples : numpy.ndarray, shape-(N,)
        The corresponding array where each sample has been replaced
        by the nearest quantized value

    Examples
    --------
    >>> import numpy as np
    >>> samples = np.array([0, .25, .75, 1])
    >>> quantize(samples, 1) # quantize among 2 values
    array([0., 0., 1., 1.])
    >>> quantize(samples, 1) # quantize among 4 values
    array([0., 0.3333, .6666, 1.])
    """

    assert bit_depth <= 14, "Exceeding this bit-depth might tank your computer!"

    # create the 2**M evenly-spaced quantized values,
    # spanning [samples.min(), samples.max()]
    quantized_values = np.linspace(samples.min(), samples.max(), 2 ** bit_depth)

    # Broadcast subtract: shape-(N, 1) w/ shape-(M**2,) -> shape(N, M**2)
    # `abs_differences[i]` is the absolute difference between sample-i and
    # each of the M**2 quantized values
    abs_differences = np.abs(samples[:, np.newaxis] - quantized_values)

    # For each sample, find which quantized value it is closest to.
    # Produced shape-(N,) array on indices on [0, 2**M)
    bin_lookup = np.argmin(abs_differences, axis=1)

    # Populate a shape-(N,) array, where each sample has been
    # replaced by its nearest quantized value. This leverages
    # advanced integer-array indexing
    return quantized_values[bin_lookup]

def analog_to_digital(
    analog_signal: Callable[[np.ndarray], np.ndarray],
    *,
    sampling_rate: float,
    bit_depth: int,
    duration: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Digitizes a given analog signal based on desired sampling rate and bit-depth.
    
    Parameters
    ----------
    analog_signal : Callable[[ndarray], ndarray]
        Another Python function, f(t), which accepts a time value (in seconds) as
        an input and returns a measurement (in volts) as an output.
    
    sampling_rate : float
        The sampling rate specified in Hertz.
    
    bit_depth: int
        The bit-depth, M, used to quantize the samples among
        2**M evenly spaced values spanning [samples.min(), samples.max()].
    
    duration : float
        The duration of the signal, specified in seconds (a non-negative float).
    
    Returns
    -------
    (times, digital_signal) : Tuple[ndarray, ndarray]
        The shape-(N,) array of times and the corresponding
        shape-(N,) array representing the digital signal.
    """
    times, samples = temporal_sampler(
        analog_signal, duration=duration, sampling_rate=sampling_rate
    )
    digital_signal = quantize(samples, bit_depth)

    return times, digital_signal

