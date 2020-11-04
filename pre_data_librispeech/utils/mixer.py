# Originated from https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_mixed_audio_file.py

import array
import math
import numpy as np
import wave

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()
# Read clean wav file & noise wav file and mix
# create mixed wav file
def mix_noise_wav(clean_file,noise_file,out,snr=0):
    c = wave.open(clean_file, "r")
    n = wave.open(noise_file, "r")
    
    c_amp = cal_amp(c)
    n_amp = cal_amp(n)
    
    c_rms = cal_rms(c_amp)
    
    # Choose random interval of Noise
    # Note : must be len(n_amp) > len(c_amp)
    start = np.random.randint(0, len(n_amp)-len(c_amp))
    divided_n_amp = n_amp[start: start + len(c_amp)]
    n_rms = cal_rms(divided_n_amp)
    
    adjusted_n_rms = cal_adjusted_rms(c_rms, snr)
    
    adjusted_n_amp = divided_n_amp * (adjusted_n_rms / n_rms) 
    mixed_amp = (c_amp + adjusted_n_amp)
    
    if snr < 0:
        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed_amp.min(axis=0)
            mixed_amp = mixed_amp * (reduction_rate)
            #c_amp = c_amp * (reduction_rate)
            
            
# get clean data & noise data and mix     
# return mixed data
def mix_noise(clean,noise,snr=0):  
    c_amp = clean
    n_amp = noise
    
    c_rms = cal_rms(c_amp)
    
    # Choose random interval of Noise
    # Note : must be len(n_amp) > len(c_amp)
    start = np.random.randint(0, len(n_amp)-len(c_amp))
    divided_n_amp = n_amp[start: start + len(c_amp)]
    n_rms = cal_rms(divided_n_amp)
    
    adjusted_n_rms = cal_adjusted_rms(c_rms, snr)
    
    adjusted_n_amp = divided_n_amp * (adjusted_n_rms / n_rms) 
    mixed_amp = (c_amp + adjusted_n_amp)
    
    if snr < 0:
        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed_amp.min(axis=0)
            mixed_amp = mixed_amp * (reduction_rate)
            #c_amp = c_amp * (reduction_rate)
    return mixed_amp