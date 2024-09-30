from dataclasses import dataclass
import numpy as np
from signal_generator import SignalGenerator
from noise_generator import NoiseGenerator
from beat_interval_generator import BeatIntervalGenerator
import random
from utils import find_corresponding, create_label, default_field
import matplotlib.pyplot as plt

@dataclass
class ECGWavePrms:
    p: float=0.0
    q: float=0.0
    r: float=0.0
    s: float=0.0
    t: float=0.0
    
    def to_list(self):
        return [self.p, self.q, self.r, 
                self.s, self.t]

@dataclass
class ECGGenerator(SignalGenerator):

    noise_generator: NoiseGenerator = default_field(NoiseGenerator())
    beat_interval_generator: BeatIntervalGenerator = default_field(BeatIntervalGenerator())
    fs: int = 200
    number_of_beats: int = 30
    ecg_amplitude: ECGWavePrms = default_field(ECGWavePrms(0.1, -0.08, 1.0, -0.08, 0.3))
    ecg_amplitude_low: ECGWavePrms = default_field(ECGWavePrms(0.05, -0.05, 0.8, -0.05, 0.1))
    ecg_amplitude_high: ECGWavePrms = default_field(ECGWavePrms(0.2, -0.2, 1.2, -0.2, 0.6))
    ecg_width: ECGWavePrms = default_field(ECGWavePrms(0.15, 0.1, 0.1, 0.1, 0.5))
    ecg_width_low: ECGWavePrms = default_field(ECGWavePrms(0.065, 0.03, 0.06, 0.03, 0.085))
    ecg_width_high: ECGWavePrms = default_field(ECGWavePrms(0.085, 0.08, 0.085, 0.08, 0.21))
    ecg_distance: ECGWavePrms = default_field(ECGWavePrms(-0.12, -0.04, 0.0, 0.03, 0.25))
    ecg_distance_low: ECGWavePrms = default_field(ECGWavePrms(-0.12, -0.03, 0.0, 0.03, 0.2))
    ecg_distance_high: ECGWavePrms = default_field(ECGWavePrms(-0.18, -0.05, 0.0, 0.05, 0.25))
    ecg_symmetry: ECGWavePrms = default_field(ECGWavePrms(1.0, 1.0, 1.0, 1.0, 3.0))
    ecg_symmetry_low: ECGWavePrms = default_field(ECGWavePrms(1.0, 1.0, 1.0, 1.0, 1.0))
    ecg_symmetry_high: ECGWavePrms = default_field(ECGWavePrms(1.0, 1.0, 1.0, 1.0, 5.0))


    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates clean or noisy ECG signals. Also returns labels for P and T waves and R peaks and for noise.
        
        Returns
        ----------
        signal
            Clean or noisy ECG signal.
        peak_labels
            1D array of labels for P, R and T waves.
        labels
            Noise labels: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last item in the list is the artifact label (array).
        beat_intervals
            Beat intervals in seconds
        """

        self.distance = self.ecg_distance.to_list()
        self.width = self.ecg_width.to_list()
        self.amplitude = self.ecg_amplitude.to_list()
        self.symmetry = self.ecg_symmetry.to_list()
        labels = []

        if self.noise_generator is not None:
            self.noise_generator.fs = self.fs
            noise_signal, labels = self.noise_generator.generate()     

            if self.noise_generator.noise_list:
                dur = 0
                for item in self.noise_generator.noise_list:
                    dur += item.duration
                self.number_of_beats = int(dur/self.beat_interval_generator.mu*1.1)

            # if noise_list is empty and noise was generated solely from parameters in noise_type
            else:   
                self.number_of_beats = np.min([self.number_of_beats,
                                              int((self.noise_generator.noise_type.duration/self.beat_interval_generator.mu)*1.1)])

        self.beat_interval_generator.n = self.number_of_beats
        beat_intervals = self.beat_interval_generator.generate()
        signal, beat_intervals = super().generate(beat_intervals, self.fs)

        # Find R peaks.
        r_peaks = np.zeros(len(beat_intervals), dtype=int)
        for i, nn in enumerate(beat_intervals):
            r_peaks[i] = nn // 2 if i == 0 else nn // 2 + np.sum(beat_intervals[:i])
        
        p_waves = r_peaks + np.array(self.ecg_distance.p*beat_intervals, dtype=int)
        t_waves = r_peaks + np.array(self.ecg_distance.t*beat_intervals, dtype=int)
        peak_labels = np.zeros(len(signal))
        peak_labels = create_label(peak_labels, p_waves, 0.2)
        peak_labels = create_label(peak_labels, r_peaks, 0.6)
        peak_labels = create_label(peak_labels, t_waves, 0.4)
        
        if self.noise_generator is not None:
            signal, peak_labels, labels = self.noise_generator.combine_signal_noise(signal, noise_signal, peak_labels, labels)  


        return signal, peak_labels, labels, beat_intervals/self.fs
    
    
    def generate_random_set(self, number_of_signals, duration):
        """
        Generates set of random ECG signals.
        
        Parameters
        ----------
        number_of_signals
            Number of generated signals.
        duration
            Duration of each signal.
        Returns
        ----------
        signal
            List of clean or noisy ECG signal.
        peak_labels
            List of 1D array of labels for P, R and T waves.
        labels
            List of noise labels: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last item in the list is the artifact label (array).
        beat_list
            List of beat intervals.
        """
        signals, peak_inds, labels, beats_list = [], [], [], []
        self.number_of_beats = duration
        for _ in range(number_of_signals):
            self.beat_interval_generator.beat_intervals = None
            self.ecg_distance = self._randomize_prms(self.ecg_distance_low, self.ecg_distance_high)
            self.ecg_width = self._randomize_prms(self.ecg_width_low, self.ecg_width_high)
            self.ecg_amplitude = self._randomize_prms(self.ecg_amplitude_low, self.ecg_amplitude_high)
            self.ecg_symmetry = self._randomize_prms(self.ecg_symmetry_low, self.ecg_symmetry_high)
            self.beat_interval_generator.randomize()
            
            if self.noise_generator is not None:
                self.noise_generator.randomize()
            
            signal, r_peaks, label, beats = self.generate()
            signals.append(signal)
            peak_inds.append(r_peaks)
            labels.append(label)
            beats_list.append(beats)
        
        return signals, peak_inds, labels, beats_list

    
    def _randomize_prms(self, low_prms: ECGWavePrms, 
                          high_prms: ECGWavePrms) -> np.ndarray:
        """
        Randomizes waveform parameters.
        
        Parameters
        ----------
        low_prms
            Lower limit
        high_prms
            Upper limit
        x
            Random coefficient between 0 and 1.

        Returns
        ----------
        randomized wave parameters.
        """
        arr = np.zeros(len(low_prms.__dataclass_fields__))
        for i, (f_low, f_high) in enumerate(zip(low_prms.__dataclass_fields__, 
                                                high_prms.__dataclass_fields__)):
            arr[i] = np.random.uniform(getattr(low_prms, f_low), getattr(high_prms, f_high))
        
        return ECGWavePrms(arr[0], arr[1], arr[2], arr[3], arr[4])