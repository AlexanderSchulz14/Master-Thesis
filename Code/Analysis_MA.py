import pandas as pd
import numpy as np
import random
from datetime import date, datetime
from pandas.tseries.frequencies import to_offset
import os
import matplotlib.pyplot as plt
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
import yfinance as yfb
from fredapi import Fred
fred = Fred(api_key="ef7244731efdde9698fef5d547b7094f")
import filterpy # for Kalman Filter
# Get Data
# Data
# GDP
gdp_us = fred.get_series('GDPC1')

# CPI
cpi_us = fred.get_series('CPIAUCSL') # convert to YoY rate
infl_us = cpi_us.pct_change(periods=12).dropna() *100
infl_us.plot()

# FFR
ffr = fred.get_series('DFF') # convert daily to monthly!!!

# Cpaital Utilization
cap_util_us = fred.get_series('TCU')

# Term Spread
ts_10y3m_us = fred.get_series('T10Y3M')
ts_10y2y_us = fred.get_series('T10Y2Y')
# y_12m = fred.get_series('TY12MCD')


# Get Period
start_us = max(min(gdp_us.index),
               min(infl_us.index),
               min(ffr.index),
               min(cap_util_us.index),
               min(ts_10y2y_us.index))



end_us = min(max(gdp_us.index),
               max(infl_us.index),
               max(ffr.index),
               max(cap_util_us.index),
               max(ts_10y2y_us.index))
