import pandas as pd
import numpy as np
import random
from datetime import date, datetime
from pandas.tseries.frequencies import to_offset
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import copy
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import statsmodels.tsa.vector_ar.svar_model as svar
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import pearsonr
import requests as req
import yfinance as yf
from fredapi import Fred
fred = Fred(api_key="ef7244731efdde9698fef5d547b7094f")
import filterpy # for Kalman Filter


# Data
# Industrial Production
ind_pro_ea = fred.get_series('EA19PRINTO01GYSAM')
ind_pro_ea.rename('INDPRO_EA', inplace=True)


# CPI
cpi_ea = fred.get_series('CP0000EZ19M086NEST')
cpi_ea.rename('CPI_EA', inplace=True)

# Inflation 
infl_ea = fred.get_series('CPHPTT01EZM659N')
infl_ea.rename('Infl_EA', inplace=True)


# Yields
os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data')
factors_ea = pd.read_csv('ECB_data.csv',
                         usecols=['DATA_TYPE_FM', 'TIME_PERIOD', 'OBS_VALUE'],
                         index_col=['TIME_PERIOD'],
                         infer_datetime_format=True)
factors_ea.index = pd.to_datetime(factors_ea.index)

factors_ea_sub = factors_ea.loc[factors_ea['DATA_TYPE_FM'].isin(['BETA0', 'BETA1', 
                                                  'BETA2', 'TAU1'])]