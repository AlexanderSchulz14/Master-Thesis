# Packages
import numpy as np
import pandas as pd
import random
from datetime import date
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
from scipy.stats import pearsonr
import requests as req
import yfinance as yf
import fredapi as fred
import filterpy # for Kalman Filter

# Get Data
# API
fred_key = 'ef7244731efdde9698fef5d547b7094f'

fred = fred.Fred(api_key=fred_key)

# Data
gdp_us = fred.get_series('GDPC1')
cpi_us = fred.get_series('CPIAUCSL') # convert to YoY rate
ffr = fred.get_series('DFF') # convert daily to monthly!!!
cap_util_us = fred.get_series('TCU')
ts_10y3m_us = fred.get_series('T10Y3M')
ts_10y2y_us = fred.get_series('T10Y2Y')
# y_12m = fred.get_series('TY12MCD')
