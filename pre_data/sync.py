import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import os,glob

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

def get_wave_list_recursive(path):
    dir_list = [ x for x in glob.glob(os.path.join(path, '*')) if os.path.isdir(x)]
    files = []
    for i_path in dir_list :
        files = files + glob.glob(i_path +'/*.wav')
    return files