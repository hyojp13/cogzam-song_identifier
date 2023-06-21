def fingerprints(fanout, peaks):
    fingerprints = []
    times = []

    for count, peak in enumerate(peaks):
        if count >= len(peaks) - 15:
            break
        fingerprint = []
        for i in range(fanout):
            print_i = (peak[0], peaks[count + i + 1][0], peaks[count + i + 1][1] - peak[1])
            fingerprint.append(print_i)
        times.append(peak[1])    
        fingerprints.append(fingerprint)
    return fingerprints, times