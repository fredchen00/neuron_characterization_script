# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:32:01 2020

@author: fredc
"""
def chirp_features(nwb_fname, trace_nums):

    def subsample_average(x, width):
    
        """Downsamples x by averaging `width` points"""
        
        avg = np.nanmean(x.reshape(-1, width), axis=1)
        
        return avg
    
    # import data_set_utils
    
    def transform_trace(this_trace_data):
    
        n_sample=10000
        min_freq=1.
        max_freq=35.
        
        # sweep.select_epoch("stim")
        
        # if np.all(this_trace_data['acq'][-10:] == 0):
        
        # raise FeatureError("Chirp stim epoch truncated.")
        
        v = this_trace_data['acq'] # sweep.v
        i = this_trace_data['stim'] # sweep.i
        t = this_trace_data['time'] # sweep.t
        
        N = len(v)
        width = int(N / n_sample)
        pad = int(width*np.ceil(N/width) - N)
        v = subsample_average(np.pad(v, (pad,0), 'constant', constant_values=np.nan), width)
        i = subsample_average(np.pad(i, (pad,0), 'constant', constant_values=np.nan), width)
        t = t[::width]
        N = len(v)
        dt = t[1] - t[0]
        xf = np.linspace(0.0, 1.0/(2.0*dt), N//2)
        v_fft = fftpack.fft(v)
        i_fft = fftpack.fft(i)
        low_ind = tsu.find_time_index(xf, min_freq)
        high_ind = tsu.find_time_index(xf, max_freq)
        
        return v_fft[low_ind:high_ind], i_fft[low_ind:high_ind], xf[low_ind:high_ind]
    
    trace_data = read_nwb_traces(nwb_fname, trace_nums)
    
    array_len = 0
    
    acq_array = np.zeros((len(trace_data['acq']),len(trace_data['acq'][0])))
    
    for sweep_ct, sweep_num in enumerate(trace_nums):
    
    if array_len == 0 or array_len == len(trace_data['acq'][sweep_ct]):
        array_len = len(trace_data['acq'][sweep_ct])
    
    else:
        sys.stdout.write('Chirp length mismatch¥n')
    
    acq_array[sweep_ct] = trace_data['acq'][sweep_ct]
    
    avg_data = np.nanmean(acq_array,0)
    
    di = np.diff(trace_data['stim'][0])
    
    di_idx = np.flatnonzero(di) # != 0
    
    if trace_data['time'][0][di_idx[0]] <= 0.1:
    
        stim_range_idx = np.arange(di_idx[2] + 1, len(trace_data['stim'][0]),dtype=int)
    
    else:
    
    stim_range_idx = np.arange(di_idx[0] + 1, len(trace_data['stim'][0]),dtype=int)
    this_trace = {'time': trace_data['time'][0][stim_range_idx]
    'stim': trace_data['stim'][0][stim_range_idx]
    'acq': avg_data[stim_range_idx]}
    v_fft, i_fft, freq_fft = transform_trace(this_trace)
    Z = v_fft / i_fft
    amp = np.abs(Z)
    phase = np.angle(Z)
    
    # pick odd number, approx number of points for 2 Hz interval
    
    n_filt = int(np.rint(1/(freq_fft[1]-freq_fft[0])))*2 + 1
    
    filt = lambda x: savgol_filter(x, n_filt, 5)
    
    amp, phase = map(filt, [amp, phase])
    
    i_max = np.argmax(amp)
    
    z_max = amp[i_max]
    
    i_cutoff = np.argmin(abs(amp - z_max/np.sqrt(2)))
    
    features = {
    
    "peak_ratio": amp[i_max]/amp[0],
    
    "peak_freq": freq_fft[i_max],
    
    "3db_freq": freq_fft[i_cutoff],
    
    "z_low": amp[0],
    
    "z_high": amp[-1],
    
    "z_peak": z_max,
    
    "phase_peak": phase[i_max],
    
    "phase_low": phase[0],
    
    "phase_high": phase[-1]
    
    }
    
    return features 
