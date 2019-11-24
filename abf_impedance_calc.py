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



#place all abf files in a directory, select the directory folder using the GUI
F = FileDialog()
mcdir = F.getExistingDirectory(caption='Select ABF director')

file_list=[]
for idx,file in enumerate(os.listdir(mcdir)):
    if file.endswith(".abf"):
        file_list.append(os.path.join(mcdir, file))


impedance_array=[]
#reference freq for interpolation(data length consistency)
ref_freq=np.linspace(0.2,20,1000)
for fname in file_list:
    abf = pyabf.ABF(fname)
    recorded_var=abf.data
    time=abf.sweepX
    
    voltage=recorded_var[0,:]*1e-3
    current=recorded_var[1,:]*1e-6
    #FFT on voltage and current
    sp_V= np.fft.fft(voltage-np.mean(voltage))
    sp_I= np.fft.fft(current-np.mean(current))
    freq = np.fft.fftfreq(len(time), d=time[1])
    half_freq=int(len(time)/2)
    
    #discard frequency above 20Hz and negative frequency
    freq=freq[1:half_freq]
    sp_V=sp_V[1:half_freq]
    sp_I=sp_I[1:half_freq]
    selected_freq=freq<21
    impedance=abs(sp_V[selected_freq])/abs(sp_I[selected_freq])/1e6
    f=interp1d(freq[selected_freq],impedance)
    
    impedance_array.append(f(ref_freq))
    
    
#plot median of the impedance
impednace_array=np.asarray(impedance_array)
plt.figure()
plt.plot(ref_freq,np.median(impednace_array,axis=0))
plt.ylim([np.min(impednace_array)*0.9,np.max(impednace_array)*1.1])
plt.yscale('log')

