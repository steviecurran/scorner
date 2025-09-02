#!/opt/miniconda3/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

############# DEMOS 1 AND 2 #######################
#  USING THE corner DATA
##https://corner.readthedocs.io/en/latest/pages/custom ####
##########################################################
ndim, nsamples = 4, 50000 
np.random.seed(1234)
data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape(
    [4 * nsamples // 5, ndim]
)
mean = 4 * np.random.rand(ndim)
data2 = mean[None, :] + np.random.randn(ndim * nsamples // 5).reshape(
    [nsamples // 5, ndim]
)
samples = np.vstack([data1, data2]);
#samples_df = pd.DataFrame(samples, columns=['A','B','C','D'])
#value1 = mean; value2 = np.mean(samples, axis=0)

#######################################################
#import sys
#sys.path.insert(0,'/Users/stephencurran/steviecurran.github.io/modules/scorner/src') 
#import scorner as sc
from scorner import scorner


## DEMO 1, MINIMUM WORKING CODE
scorner.scorner(samples,test_run=True) # DON'T NEED test_run=True, BUT FULL DATA TAKES A WHILE
##  CF. 1 PLOT ON https://corner.readthedocs.io/en/latest/pages/custom

## DEMO 2, SHOWING THE TRUE VALUES
value1 = mean; value2 = np.mean(samples, axis=0)
true_v = list(value1) + list(value2) # TRUE VALUES IN A SINGLE ARRAY
true_v =  np.reshape(true_v,(2,-1)) 
true_s = [['--',1,'k'],['dotted',2,'dimgrey']] # TRUE VALUE LINE STYLES
'''
sc.scorner(samples,test_run=True,true_values=true_v,true_style = true_s,
           cols = ['a','b','c','d'],height = 1.8, fs=10,tlp = 0.5)
'''
## CF. 2 PLOT ON https://corner.readthedocs.io/en/latest/pages/custom




