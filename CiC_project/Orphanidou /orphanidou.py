from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


from ecgdetectors import Detectors

thresh = 0.66

fs=200

def filter_ecg(x, fs):
    sig = x
    # sig = sig[:,0]
    order = 3
    low_cutoff = 1  # in Hz
    high_cutoff = 15  # in Hz
    cutoff_frequency = (low_cutoff, high_cutoff)
    b, a = signal.butter(order, cutoff_frequency, btype='band', fs=fs)
    # b, a = signal.butter(3, [0.004, 0.06], 'band')    # original 
    sig = signal.filtfilt(b, a, sig, padlen=150)
    sig = (sig - min(sig)) / (max(sig) - min(sig))
    return sig

def detect_beats(sig, fs):
    
    # detect beats
    detectors = Detectors(fs)
    #beats = detectors.swt_detector(sig)
    #beats = detectors.wqrs_detector(sig) 
    beats = detectors.hamilton_detector(sig)    
    
    # find R-peaks
    tol_secs = 0.15
    tol_samps = np.floor(fs*tol_secs)
    for beat_no in range(0,len(beats)-1):
            min_el = int(max([0, beats[beat_no]-tol_samps]))
            max_el = int(min([len(sig), beats[beat_no]+tol_samps]))
            curr_samps = sig[min_el:max_el+1]
            beats[beat_no] = int(beats[beat_no]-tol_samps+np.argmax(curr_samps))
    return beats


def find_rr_ints(beats,fs):
    
    rr_int = []
    for beat_no in range(0,len(beats)-1):
        rr_int.append((1/fs)*(beats[beat_no+1]-beats[beat_no]))   # in secs
    
    return rr_int

def assess_feasibility(beats):
    
    feas = 1
    
    # find HR
    hr = 60*len(beats)/((beats[-1]-beats[0])/fs)  # in bpm

    # check HR
    if hr < 40 or hr > 180:
        print('HR out of range')
        feas = 0
        
    # find RR intervals
    rr_int = find_rr_ints(beats,fs)   # in secs
        
    # check max RR interval
    if max(rr_int) > 3:
        print('Max RR interval too large')
        feas = 0
    
    # check max to min RR interval
    rr_int_ratio = max(rr_int)/min(rr_int)
    if rr_int_ratio >= 2.2:
        print('Max to min RR interval ratio too large')
        feas = 0
    
    return feas


def calculate_template(sig, beats):
    
    # find median rr interval
    med_rr_int = calculate_med_rr_int(beats)
    
    # find no. samples either side of beat
    tol = int(np.floor(med_rr_int/2))
    sum_waves = np.zeros(1+2*tol)
    no_beats_used = 0
    for beat_no in range(0,len(beats)-1):
            min_el = beats[beat_no]-tol
            max_el = beats[beat_no]+tol
            if min_el < 0 or max_el > beats[-1]:
                continue
            curr_samps = sig[min_el:max_el+1]
            for i in range(0,len(sum_waves)):
                sum_waves[i] += curr_samps[i]
            no_beats_used +=1
    templ = sum_waves/no_beats_used
    return templ


def calculate_cc(sig, beats, templ):
    
    # find median rr interval
    med_rr_int = calculate_med_rr_int(beats)
    
    # find no. samples either side of beat
    tol = int(np.floor(med_rr_int/2))
    
    # calculate correlation coefficients for each beat
    sum_cc = 0
    no_beats_used = 0
    for beat_no in range(0,len(beats)-1):
            min_el = beats[beat_no]-tol
            max_el = beats[beat_no]+tol
            if min_el < 0 or max_el > beats[-1]:
                continue
            curr_samps = np.zeros(1+2*tol)
            for i in range(0, 1+2*tol):
                curr_samps[i] += sig[min_el+i]
            temp = np.corrcoef(curr_samps, templ)
            curr_cc = temp[0,1]
            sum_cc = np.add(sum_cc, curr_cc)
            no_beats_used +=1
            
    # find average correlation coefficient
    cc = sum_cc/no_beats_used
    return cc

def compare_cc_to_thresh(cc, thresh):
    
    if cc >= thresh:
        qual = 1
    else:
        qual = 0
        
    return qual

def calculate_med_rr_int(beats):
    
    # find RR intervals
    rr_int = find_rr_ints(beats,1)   # in samples
    
    # find median RR interval
    med_rr_int = np.median(rr_int)
    
    return med_rr_int

def assess_qual(x, fs, thresh):
    
    # filter ECG
    sig = filter_ecg(x, fs)
    
    # detect beats
    beats = detect_beats(sig, fs)
    
    # assess feasibility of beat detections
    feas = assess_feasibility(beats)
    if feas == 0:
        qual = 0
        return qual
    
    # create template beat shape
    templ = calculate_template(x, beats)
    
    # calculate correlation coefficient
    cc = calculate_cc(x, beats, templ)
    
    # compare correlation coefficient to threshold
    qual = compare_cc_to_thresh(cc, thresh)

    return qual