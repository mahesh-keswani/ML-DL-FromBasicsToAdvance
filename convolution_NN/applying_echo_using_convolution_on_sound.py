import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavfile import write

# sampling rate 
# bits per sample 
# The first is quantization in time
# The second is quantization in amplitude

spf = wave.open('StarWars3.wav', 'r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, dtype=int)
print("numpy signal:", signal.shape)

plt.plot(signal)
plt.title("Star Wars 3 without echo")
plt.show()

frame_rate = spf.getframerate()
print("Sampling rate", frame_rate)

# applying delta function which returns the same output x(t)*delta(t) = x(t) 
delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print("noecho signal:", noecho.shape)

# make sure you do this, otherwise, you will get VERY LOUD NOISE
noecho = noecho.astype(np.int16) 
write('noecho.wav', frame_rate, noecho)

# now applying echo filter
filt = np.zeros(frame_rate)
filt[0] = 1
filt[4000] = 0.8
filt[8000] = 0.6
filt[12000] = 0.4
filt[16000] = 0.2
filt[20000] = 0.15
filt[22000] = 0.1


out = np.convolve(signal, filt)
# make sure you do this, otherwise, you will get VERY LOUD NOISE
out = out.astype(np.int16) 
write('out.wav', frame_rate, out)

