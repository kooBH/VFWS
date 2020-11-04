import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import os,glob
import wave

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

def sync_wav(anchor,target,out_1,out_2):
    a = wave.open(anchor,'r')
    t = wave.open(target,'r')
    
    d_1 = a.readframes(-1)
    d_2 = t.readframes(-1)
    
    s_1 = a.getframerate()
    s_2 = t.getframerate()
    
    if s_1 != s_2 :
        print("ERROR::Samplerate : "+str(s_1)+' != ' +str(s_2))
        exit(-1)
   
    a.close()
    t.close()

    d_1 = np.fromstring(d_1,'Int16')
    d_2 = np.fromstring(d_2,'Int16')
    
    diff = compute_shift(d_1,d_2)
    if diff > 0 :
        d_4 = d_2[diff:]
    else :
        d_4 = d_2[-diff:]
    d_3 = d_1[:len(d_4)]
    
    o_1 = wave.open(out_1,'w')
    o_2 = wave.open(out_2,'w')

    o_1.setnchannels(1)
    o_2.setnchannels(1)

    o_1.setsampwidth(2)
    o_2.setsampwidth(2)

    o_1.setframerate(16000)
    o_2.setframerate(16000)

    o_1.writeframes(d_3)
    o_2.writeframes(d_4)

    o_1.close()
    o_2.close()
    
