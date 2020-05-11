#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:55:49 2020

@author: cecilie
"""



import wfdb
import matplotlib.pyplot as plt
import numpy as np

#%%
record = wfdb.rdsamp('./Test_data/101', sampto=3000)
annotation = wfdb.rdann('./Test_data/101', 'atr', sampto=3000)

#wfdb.plotrec(record, annotation = annotation,\
#    title='Record 100 from MIT-BIH Arrhythmia Database',\
#    timeunits = 'seconds')
    
#%%


signal = record[0]

plt.figure()
plt.plot(signal[:,0])
plt.show()

#%%


np.save('./Test_data/101.np',signal)