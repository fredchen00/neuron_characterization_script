# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:41:12 2019

@author: fredc
generate figures for impedance vs frequency and detect its resonance and cutoff frequency
"""


import matplotlib.pyplot as plt
import pyabf
import numpy as np
from pyqtgraph import FileDialog
import os
from scipy.interpolate import interp1d
from scipy.signal import gaussian 
import yaml
import pandas as  pd
from scipy.signal import savgol_filter

def subsample_average(x, width):

    """Downsamples x by averaging `width` points"""
    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg


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

def plot_trace(imp,freq,moving_avg_wind,time,voltage,current,sharpness_thr,filtered_method,fig_idx,save_path):
    
    plt.figure(figsize=(20,20))
    plt.subplot(3, 1, 1)
    cen_freq,freq_3db,res_sharpness=plot_impedance_trace(imp,freq,moving_avg_wind,fig_idx,sharpness_thr,filtered_method,True)
    
    #plot voltage and current time trace
    plt.subplot(3, 1, 2)
    plt.plot(time, voltage*1e3)

    plt.ylabel('Voltage (mV)')
    
    plt.subplot(3, 1, 3)
    plt.plot(time, current*1e12)
    plt.xlabel('time (s)')
    plt.ylabel('Current(pA)')

    plt.savefig(save_path+str(fig_idx)+'.png')
    plt.close()
    
    
    return cen_freq,freq_3db,res_sharpness
    
    
    
def plot_impedance_trace(imp,freq,moving_avg_wind,fig_idx,sharpness_thr,filtered_method,plot_raw):
    #generate impedance trace over frequency with peak and cutoff frequency detection
    imp=imp/1e6
    if plot_raw:
        plt.plot(freq,imp)
    
    prominence_factor=1.01
    if filtered_method==1:
       filtered_imp=moving_average(imp,moving_avg_wind)
    elif filtered_method==2:
       start_idx=np.argmin(freq-0.5)
       freq=freq[start_idx:]
       imp=imp[start_idx:]
       filtered_imp=moving_average(imp,moving_avg_wind)

#    filtered_imp = savgol_filter(imp, moving_avg_wind, 1)
    plt.plot(freq,filtered_imp)
    plt.ylim([np.min(imp)*0.9,np.max(imp)*1.1])
    idx_max_mag=np.argmax(filtered_imp)
    cen_freq=freq[idx_max_mag]

    
    left_imp_mean=np.median(filtered_imp[0:idx_max_mag-1])
    right_imp_mean=np.median(filtered_imp[idx_max_mag+1:])
    max_imp=filtered_imp[idx_max_mag]
    
    if (left_imp_mean*prominence_factor)>max_imp  or  (right_imp_mean*prominence_factor)>max_imp or cen_freq<0.5 :
        cen_freq=0  
        
    if cen_freq>0:
        res_sharpness=max_imp/filtered_imp[np.argmin(freq-0.5)]
    else:
        res_sharpness=0
        
    if sharpness_thr>res_sharpness:
        cen_freq=0  
    #find cutoff freq(3dB below max)
    if cen_freq>0:
        i_3db_cutoff=np.argmin(abs(filtered_imp-max_imp/np.sqrt(2)))
        freq_3db=freq[i_3db_cutoff]
    else:
        freq_3db=0
        
        
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Impedance[MOhms]')
    if cen_freq is not None:
        if freq_3db is not None:
            plt.title('Trial {fig_idx}, '.format(fig_idx=fig_idx)+'Fr={:.2f} Hz, Cutoff Freq={:.2f}Hz, Sharpness={:.2f}'.format(cen_freq,freq_3db,res_sharpness))
        else:
            plt.title('Trial {fig_idx}, '.format(fig_idx=fig_idx)+'Fr={:.2f} Hz, Cutoff Freq=None'.format(cen_freq))
    else:
        plt.title('Trial {fig_idx}, No Resonance')
    plt.legend(['Raw Trace','Moving Averaged'])
    
    
    return cen_freq,freq_3db,res_sharpness

     
def cal_imp(abf,sweep_end_freq):
    ref_freq=np.linspace(0.1,sweep_end_freq,1000)
    recorded_var=abf.data
    dataRate=abf.dataRate
    total_time=np.arange(0,recorded_var.shape[1])/dataRate
    n_sample=10000
    adcUnits=abf.adcUnits
    #find the corresponding index for voltage and current in data
    current_idx=1
    voltage_idx=0
    for unit_idx,unit in enumerate(adcUnits):
        if unit=='pA':
            current_idx=unit_idx
        elif unit=='mV':
            voltage_idx=unit_idx
#    hamming_window=np.hamming(len(time))
#    hamming_window=gaussian(len(time), std=len(time)/2)
    #count the number of data sets in one folder and segment data accordingly
    sweepList=abf.sweepList 
    sweepTimesSec=np.asarray(sweepList)*abf.sweepLengthSec
    voltage_array=[]
    current_array=[]
    time_array=[]
    impedance_array=[]
    for sweep_idx in sweepList: 

        start_idx=int(sweepTimesSec[sweep_idx]*dataRate)
        if sweep_idx==sweepList[-1]:
            end_idx=len(total_time)-1
        else:
            end_idx=int(sweepTimesSec[sweep_idx+1]*dataRate)-1
            
            
        voltage=recorded_var[voltage_idx,start_idx:end_idx]*1e-3
        N=voltage.shape[0]
        width = int(N / n_sample)
        pad = int(width*np.ceil(N/width) - N)
        
        
        voltage_detrend=subsample_average(np.pad(voltage, (pad,0), 'constant', constant_values=np.nan), width)
        current=recorded_var[current_idx,start_idx:end_idx]*1e-12
        current_detrend=subsample_average(np.pad(current, (pad,0), 'constant', constant_values=np.nan), width)
        voltage_array.append(voltage_detrend)
        current_array.append(current_detrend)
        time=total_time[start_idx:end_idx]-total_time[start_idx]
        time=time[::width]
        time_array.append(time)
        #FFT on voltage and current
        sp_V= np.fft.fft(voltage_detrend)
        
        
        sp_I= np.fft.fft(current_detrend)
        freq = np.fft.fftfreq(len(time), d=time[1])
        half_freq=int(len(time)/2)
        
        #discard frequency above 20Hz and negative frequency
        freq=freq[1:half_freq]
        sp_V=sp_V[1:half_freq]
        sp_I=sp_I[1:half_freq]
        selected_freq=freq<21
        impedance=np.abs(sp_V[selected_freq]/sp_I[selected_freq])
        
        f_v=interp1d(freq[selected_freq],sp_V[selected_freq])
        f_i=interp1d(freq[selected_freq],sp_I[selected_freq])
        f_imp=interp1d(freq[selected_freq],impedance,fill_value="extrapolate")
        interpolated_trace=f_imp(ref_freq)
        impedance_array.append(interpolated_trace)
#        plt.figure()
#        plt.plot(time,recorded_var[voltage_idx,start_idx:end_idx])
        

    return impedance_array,ref_freq,voltage_array,current_array,time_array



    


#place all abf files in a directory, specify the path in the yaml config file
params=get_yaml_config_params()
data_file_path=params['data_file_path']
sweep_end_freq=params['sweep_end_freq']
moving_avg_wind=params['moving_avg_wind']
root_result_folder=params['root_result_folder']
is_sharpness_filter=params['is_resonance_filter']
sharpness_thr=params['sharpness_thr']
filtered_method=params['filtered_method']

if not(is_sharpness_filter):
    is_sharpness_filter=0

root_result_path=os.path.join(data_file_path,root_result_folder)
#generating directory to save results
if not(os.path.exists(root_result_path)):
    os.mkdir(root_result_path)


root_result_path=os.path.join(data_file_path,root_result_folder)
#generating directory to save results
if not(os.path.exists(root_result_path)):
    os.mkdir(root_result_path)
root_imped_fig_path=os.path.join(root_result_path,'impedance_fig')
if not(os.path.exists(root_imped_fig_path)):
    os.mkdir(root_imped_fig_path)


folder_list=[]

for idx_folder,folder_name in enumerate(os.listdir(data_file_path)):
    print(folder_name)
    path=os.path.join(data_file_path,folder_name)
    if os.path.isdir(path):
        folder_list.append(path)

impedance_file_array=[]
#storing data from different files in a list
v_file_array=[]
i_file_array=[]
#reference freq for interpolation(data length consistency)

index=['trial','center_freq','3dB_freq']
df_array=[]
impedance_mean_array=[]
ref_freq_array=[]
datacount=0
for folder_path in folder_list:
    file_list=os.listdir(folder_path)
    if file_list[0].endswith('.abf'):
        folder_name=folder_path.split('\\')[-1]
        df = pd.DataFrame(columns=index)
        impedance_all=[]
        
        for idx_file,file_name in enumerate(os.listdir(folder_path)):
            if file_name.endswith(".abf"):
                fname=os.path.join(folder_path, file_name)
                dir_info=fname.split('\\')
                cell_ID=dir_info[-2]
                abf_name=dir_info[-1].split('.')[0]
                
                
                print('Processing: '+abf_name)
                if not(cell_ID==abf_name):
                    abf = pyabf.ABF(fname)
                    impedance_array,ref_freq,voltage_array,current_array,time_array=cal_imp(abf,sweep_end_freq)
                    impedance_all.append(np.asarray(impedance_array))
                    
                    for trial_idx,impedance in enumerate(impedance_array):
                        
                        imped_fig_path=os.path.join(root_imped_fig_path,cell_ID)
                        if not(os.path.exists(imped_fig_path)):
                            os.mkdir(imped_fig_path)
                        cen_freq,freq_3db,res_sharpness=plot_trace(impedance,ref_freq,moving_avg_wind,time_array[trial_idx],voltage_array[trial_idx],current_array[trial_idx],sharpness_thr,filtered_method,
                                                                   trial_idx,os.path.join(imped_fig_path,abf_name))
                        
                        
                        df=df.append({'trial':trial_idx,'center_freq':cen_freq,'3dB_freq':freq_3db,'res_sharpness':res_sharpness},ignore_index=True)
                        
        impedance_all=np.concatenate(impedance_all)
        impedance_mean=np.median(impedance_all,axis=0)
        impedance_mean_array.append(impedance_mean)
        ref_freq_array.append(ref_freq)
        cen_freq,freq_3db,res_sharpness=plot_trace(impedance_mean,ref_freq,moving_avg_wind,time_array[0],np.zeros((time_array[0].shape[0],)),np.zeros((time_array[0].shape[0],)),sharpness_thr,
                                                   filtered_method,-1,os.path.join(imped_fig_path,'avg'))
        
        
        
        df=df.append({'trial':'avg','center_freq':cen_freq,'3dB_freq':freq_3db,'res_sharpness':res_sharpness},ignore_index=True)
        df.set_index('trial')

        df_array.append([df,folder_name])

plt.figure()
max_val=0
min_val=0
for i in range(len(impedance_mean_array)):
    impedance_mean=impedance_mean_array[i]
    ref_freq=ref_freq_array[i]
    plot_impedance_trace(impedance_mean,ref_freq,moving_avg_wind,0,sharpness_thr,filtered_method,False)
    max_val=max(max_val,np.max(impedance_mean))

#set color for the first line
plt.gca().get_lines()[0].set_color("red")
#set color for the second line
plt.gca().get_lines()[1].set_color("blue")
#change the title
#plt.gca().set_title('test')
plt.gca().set_ylim([0,max_val/1e6])
plt.gca().set_xlim([0,20])
#switch this appropriately if you notice the label is flipped
plt.legend(['Resonance','No Resonance'])

    
        
with pd.ExcelWriter(os.path.join(root_imped_fig_path,'impedance_info.xlsx'), engine='xlsxwriter') as writer:
    for i in range(len(df_array)):
        df=df_array[i][0]
        abf_name=df_array[i][1]
        df.to_excel(writer,sheet_name=abf_name)


    
    
    
    
    

