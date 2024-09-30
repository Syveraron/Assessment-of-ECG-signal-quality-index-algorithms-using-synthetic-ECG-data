import matplotlib.pyplot as plt
import numpy as np
import ecg_generator as eg
import ppg_generator as pg
from noise_generator import NoiseGenerator
import utils
from wfdb.io import get_record_list

gen = eg.ECGGenerator()
#gen = pg.PPGGenerator()

gen.noise_generator = None
signal, peak_inds, _ = gen.generate()
plt.figure()
plt.plot(signal)
plt.plot(peak_inds, signal[peak_inds], 'co')
plt.show()


w1 = ('walking', 50, 0.05)
h1 = ('hand_movement', 50, 0.05)
w2 = ('walking', 20, 0.1)
m1 = ('model', 20, 0.1)

gen.noise_generator = NoiseGenerator()
gen.noise_generator.noise_list = [w1,h1,w2,m1]
y, peaks_inds_noisy, labels = gen.generate()
fig, ax = plt.subplots()
ax.plot(y, color='steelblue')
ax.plot(peaks_inds_noisy, y[peaks_inds_noisy], 'co')
for noise in labels:
    noise_idx = np.arange(noise[1], noise[2])
    plt.plot(noise_idx, np.zeros(len(noise_idx)), label=noise[0])
plt.show()

signal_set_noisy, peak_inds, labels = gen.generate_random_set(5, 30)
plt.figure()
plt.plot(signal_set_noisy[2])
plt.plot(peak_inds[2], signal_set_noisy[2][peak_inds[2]], 'co')
plt.plot(labels[2][-1])
plt.show()


#mitdb_records = get_record_list('mitdb')
#_, mitdb_beats, _ = utils.data_from_records(mitdb_records, channel=0, db='mitdb')
#beats = np.array(mitdb_beats[0])
beats = np.loadtxt('measurements/3.csv', delimiter=',')
beat_intervals = np.diff(beats)
fs=360
beat_intervals = beat_intervals[0:300]/fs
beat_intervals = list(beat_intervals)
gen.beat_interval_generator.beat_intervals = beat_intervals

y, peaks_inds_noisy, labels = gen.generate()
fig, ax = plt.subplots()
ax.plot(y, color='steelblue')
ax.plot(peaks_inds_noisy, y[peaks_inds_noisy], 'co')
for noise in labels[:-1]:
    noise_idx = np.arange(noise[1], noise[2])
    plt.plot(noise_idx, np.zeros(len(noise_idx)), label=noise[0])
plt.plot(labels[-1], label = 'artifact')
plt.legend()
plt.show()
