from PIL import Image
import numpy as np
import scipy.fftpack as fp
from scipy import signal
import pandas as pd 
from matplotlib import pyplot as plt
import cv2

## Functions to go from image to frequency-image and back
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                               axis=1)
freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                             axis=0)

## Read in data file and transform
fn = '44187543_1_4'
#fn = '06755299_0_6'
data = cv2.imread('chars/'+fn+'.png')
data = cv2.resize(data,None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
#print(data)

freq = im2freq(data)
back = freq2im(freq)
# Make sure the forward and backward transforms work!
assert(np.allclose(data, back))

## Helper functions to rescale a frequency-image to [0, 255] and save
remmax = lambda x: x/x.max()
remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def arr2im(data, fname):
    out = Image.new('RGB', data.shape[1::-1])
    out.putdata(map(tuple, data.reshape(-1, 3)))
    out.save(fname)

def arr2csv(np_array,fname):
    df = pd.DataFrame(np_array)
    df.to_csv(fname, header=None)

#filter_data= butter_highpass_filter(freq,5,30)

arr2csv(touint8(freq), 'data_'+fn+'.csv')

plt.subplot(121),plt.imshow(data, cmap = 'gray')
plt.subplot(122),plt.imshow(touint8(freq), cmap = 'gray')

#plt.subplot(122),plt.plot(touint8(data))
#plt.subplot(122),plt.plot(touint8(freq))
#plt.subplot(122),plt.plot(touint8(filter_data))
plt.savefig("fft_"+fn+".png")    