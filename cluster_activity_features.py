# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:04:40 2019

@author: fredc
"""
import pandas as pd
import yaml
from pyqtgraph import FileDialog
import os
import numpy as np
from sklearn.decomposition import  PCA
def get_yaml_config_params():
    F = FileDialog()
    fname = F.getOpenFileName(caption='Select a Config File')[0]
    #load yaml params files
    with open(fname,'r') as f:
        params = yaml.load(f)
        
        
    return params
#place all abf files in a directory, specify the path in the yaml config file
params=get_yaml_config_params()
data_file_path=params['data_file_path']
file_name=params['file_name']
sheet_array=params['sheet_array']


select_idx=6
features_array=[]
for sheet_num in sheet_array:
    df=pd.read_excel(os.path.join(data_file_path, file_name+'.xlsx'),'Sheet'+str(sheet_num))
    temp=df[select_idx:select_idx+1]
    features_array.append(temp.to_numpy(dtype='float32')[0,1:])
    
    
    
features_array=np.squeeze(np.asarray(features_array))

# calculate PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])




