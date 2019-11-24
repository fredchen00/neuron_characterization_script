# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:41:12 2019

@author: fredc
"""


import matplotlib.pyplot as plt
import pyabf
import numpy as np
from pyqtgraph import FileDialog
import os
from scipy.interpolate import interp1d
from scipy.signal import gaussian 
import yaml


def moving_average(x,window_size):
    #moving average across data x within a window_size
    half_window_size=int(window_size/2)
    time_len=len(x)
    moving_average_trace=[]
    max_lim=(time_len-window_size/2)
    min_lim=half_window_size
    for i in range(time_len):
        if i >=min_lim and i <=max_lim :
            moving_average_trace.append(np.mean(x[i-half_window_size:i+half_window_size]))
        elif i<min_lim:
            print(i)
            moving_average_trace.append(np.mean(x[0:i+half_window_size]))
        elif i>max_lim:
            moving_average_trace.append(np.mean(x[i-half_window_size:time_len]))

    return np.asarray(moving_average_trace)

def get_yaml_config_params():
    #select yaml file and load all parameters
    F = FileDialog()
    fname = F.getOpenFileName(caption='Select a Config File')[0]
    #load yaml params files
    with open(fname,'r') as f:
        params = yaml.load(f)
        
    return params
def plot_impedance_trace(imp,freq,moving_avg_wind):
    #generate impedance trace over frequency with peak and cutoff frequency detection
    plt.figure()
    plt.plot(freq,imp)
    moving_average_trace=moving_average(imp,moving_avg_wind)
    plt.plot(freq,moving_average_trace)
    plt.ylim([np.min(imp)*0.9,np.max(imp)*1.1])
    Fr=freq[np.argmax(moving_average_trace)]
    #find cutoff freq(3dB below max)
    mag_3db=(np.max(moving_average_trace)/np.sqrt(2))
    freq_3db=freq[np.nonzero(moving_average_trace<mag_3db)[0][0]]
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Impedance[MOhms]')
    plt.title('Fr='+"{:.2f}".format(Fr)+'Hz, '+'Cutoff Freq='+"{:.2f}".format(freq_3db)+'Hz')
    plt.legend(['Raw Trace','Moving Averaged'])


#place all abf files in a directory, specify the path in the yaml config file
params=get_yaml_config_params()
data_file_path=params['data_file_path']
to_average=params['to_average']
sweep_end_freq=params['sweep_end_freq']
moving_avg_wind=params['moving_avg_wind']

file_list=[]
for idx,file in enumerate(os.listdir(data_file_path)):
    if file.endswith(".abf"):
        file_list.append(os.path.join(data_file_path, file))


impedance_file_array=[]
#storing data from different files in a list
v_file_array=[]
i_file_array=[]
#reference freq for interpolation(data length consistency)

ref_freq=np.linspace(0.1,sweep_end_freq,1000)
for fname in file_list:
    abf = pyabf.ABF(fname)
    recorded_var=abf.data
    dataRate=abf.dataRate
    total_time=np.arange(0,recorded_var.shape[1])/dataRate
    
    adcUnits=abf.adcUnits
    #find the corresponding index for voltage and current in data
    for unit_idx,unit in enumerate(adcUnits):
        if unit=='pA':
            current_idx=unit_idx
        elif unit=='mV':
            voltage_idx=unit_idx
#    hamming_window=np.hamming(len(time))
#    hamming_window=gaussian(len(time), std=len(time)/2)
    #count the number of data sets in one folder and segment data accordingly
    sweepList=abf.sweepList 
    sweepTimesSec=abf.sweepTimesSec
    
    voltage_array=[]
    current_array=[]
    impedance_array=[]
    for sweep_idx in sweepList: 
        voltage=recorded_var[voltage_idx,:]*1e-3
        current=recorded_var[current_idx,:]*1e-6

        start_idx=int(sweepTimesSec[sweep_idx]*dataRate)
        if sweep_idx==sweepList[-1]:
            end_idx=len(total_time)-1
        else:
            end_idx=int(sweepTimesSec[sweep_idx+1]*dataRate)-1
        
        voltage_array.append(recorded_var[voltage_idx,start_idx:end_idx]*1e-3)
        current_array.append(recorded_var[current_idx,start_idx:end_idx]*1e-6)
        time=total_time[start_idx:end_idx]-total_time[start_idx]
        #FFT on voltage and current
        sp_V= np.fft.fft(voltage)
        
        
        sp_I= np.fft.fft(current)
        freq = np.fft.fftfreq(len(time), d=time[1])
        half_freq=int(len(time)/2)
        
        #discard frequency above 20Hz and negative frequency
        freq=freq[1:half_freq]
        sp_V=np.abs(sp_V[1:half_freq])
        sp_I=np.abs(sp_I[1:half_freq])
        selected_freq=freq<21
        impedance=sp_V[selected_freq]/sp_I[selected_freq]/1e6
        
        f_v=interp1d(freq[selected_freq],sp_V[selected_freq])
        f_i=interp1d(freq[selected_freq],sp_I[selected_freq])
        f_imp=interp1d(freq[selected_freq],impedance)
        interpolated_trace=f_imp(ref_freq)
        impedance_array.append(interpolated_trace)
        voltage_array.append(f_v(ref_freq))
        current_array.append(f_i(ref_freq))
        plt.figure()
        plt.plot(time,recorded_var[voltage_idx,start_idx:end_idx])
        
    v_file_array.append(voltage_array)
    i_file_array.append(current_array)
    impedance_file_array.append(impedance_array)
    
    
#    plt.figure()
#    plt.plot(ref_freq,interpolated_trace)
#    plt.plot(freq[selected_freq],impedance)
#    

#plot median of the impedance
median_impedance_trace=0
if to_average:
    #for averaging all impedance trace and plot it on a single figure
    count=0
    for impedance_array in impedance_file_array:
        impednace_array=np.asarray(impedance_array)
        
        median_impedance_trace=median_impedance_trace+np.median(impednace_array,axis=0)
        count=count+1
    median_impedance_trace=median_impedance_trace/count
    plot_impedance_trace(median_impedance_trace/count,ref_freq,moving_avg_wind)

else:
    #for plotting all individual traces in separate figures

    for impedance_array in impedance_file_array:
        for impedance in impedance_array:
            plot_impedance_trace(impedance,ref_freq,moving_avg_wind)

    
    
    
    
    

