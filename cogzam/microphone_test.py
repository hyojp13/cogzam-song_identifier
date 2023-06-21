import numpy as np
from microphone import record_audio
def record(listen_time: float):

    frames, sample_rate = record_audio(listen_time)
    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    return samples