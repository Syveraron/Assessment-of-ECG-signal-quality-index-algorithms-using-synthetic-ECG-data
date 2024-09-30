import numpy as np
import scipy.signal
from ecgdetectors import Detectors
import scipy.stats
import neurokit2 as nk
from bigO import BigO
#from bigO import algorithm
import time

detectors = Detectors(500)

# region set parameters
sampling_frequency = 500        # Hz
nyquist_frequency = sampling_frequency * 0.5        # Hz
max_loss_passband = 0.1     # dB
min_loss_stopband = 20      # dB
SNR_threshold = 0.5
signal_freq_band = [2, 40]      # from .. to .. in Hz
heart_rate_limits = [24, 300]       # from ... to ... in beats per minute
t = 10       # seconds
window_length = 100      # measurements
# endregion


def high_frequency_noise_filter(data):
    order, normal_cutoff = scipy.signal.buttord(20, 30, max_loss_passband, min_loss_stopband, fs=sampling_frequency)
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data


def baseline_filter(data):
    order, normal_cutoff = scipy.signal.buttord(0.5, 8, max_loss_passband, min_loss_stopband, fs=sampling_frequency)
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data


def stationary_signal_check(data, total_leads):
    res = []
    for lead in range(1, total_leads + 1):
        window_matrix = np.lib.stride_tricks.sliding_window_view(data[lead], window_length)[::10]
        for window in window_matrix:
            if np.amax(window) == np.amin(window):
                res.append(1)
                break
        if len(res) != lead:
            res.append(0)
    return res


def heart_rate_check(data, total_leads):
    res = []
    for lead in range(1, total_leads + 1):
        beats = detectors.pan_tompkins_detector(data[lead])
        if len(beats) > ((heart_rate_limits[1]*t)/60) or len(beats) < ((heart_rate_limits[0]*t)/60):
            res.append(1)
        else:
            res.append(0)
    return res


def signal_to_noise_ratio_check(data, total_leads):
    res = []
    for lead in range(1, total_leads + 1):
        f, pxx_den = scipy.signal.periodogram(data[lead], fs=sampling_frequency, scaling="spectrum")
        if sum(pxx_den):
            signal_power = sum(pxx_den[(signal_freq_band[0]*10):(signal_freq_band[1]*10)])
            SNR = signal_power / (sum(pxx_den) - signal_power)
        else:
            res.append(0)
            continue
        if SNR < SNR_threshold:
            res.append(1)
        else:
            res.append(0)
    return res






def processing1(ECG, total_leads, temp_freq):
    second = time.time()
    resampled_ECG = []
    if temp_freq != 500:
        for n in range(0, total_leads + 1):
            resampled_ECG.append(nk.signal_resample(ECG[n], sampling_rate=int(temp_freq), desired_sampling_rate=500, method="numpy"))
    else:
        resampled_ECG = ECG

    filt_ECG = [resampled_ECG[0]]
    for lead in range(1, total_leads + 1):
        x = high_frequency_noise_filter(ECG[lead]) - baseline_filter(ECG[lead])
        filt_ECG.append(x)

    SQM = []  # Signal Quality Matrix
    SQM.append(stationary_signal_check(ECG, total_leads))
    SQM.append(heart_rate_check(filt_ECG, total_leads))
    SQM.append(signal_to_noise_ratio_check(ECG, total_leads))

    combination = list("" for i in range(0, total_leads))
    for lead in range(1, total_leads + 1):
        combination[lead - 1] = SQM[0][lead - 1] + SQM[1][lead - 1] + SQM[2][lead - 1]

    res = []
    for x in range(0, total_leads):
        if combination[x] >= 1:
            combination[x] = u"\u2716"
        else:
            combination[x] = u"\u2714"

    for y in range(0, 3):
        SQM_print = []
        for x in range(0, total_leads):
            if SQM[y][x] == 1:
                SQM_print.append(u"\u2716")
            else:
                SQM_print.append(u"\u2714")
        res.append(SQM_print)

    # print(tabulate(SQM, headers=lead_name, showindex=SQM_rows))
    # print(LQI)
    res.append(combination)
    print(time.time()-second)
    return res


