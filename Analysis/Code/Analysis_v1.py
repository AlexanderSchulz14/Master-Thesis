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
import nelson_siegel_svensson # Nelson-Siegel YC decomposition

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



# # Get Period
# start_us = max(min(gdp_us.index),
#                min(infl_us.index),
#                min(ffr_m.index),
#                min(cap_util_us.index),
#                min(ts_10y2y_us.index))



# end_us = min(max(gdp_us.index),
#                max(infl_us.index),
#                max(ffr_m.index),
#                max(cap_util_us.index),
#                max(ts_10y2y_us.index))




# # Merge Data
# df_us = [gdp_us[start_us:end_us],
#          infl_us[start_us:end_us],
#          ffr_m[start_us:end_us],
#          cap_util_us[start_us:end_us],
#          ts_10y2y_us[start_us:end_us]]

# df_us = pd.concat(df_us, axis=1).dropna()

# # Analysis

# # Stationarity Check
# get_adf(df_us)



# # VAR
# var_us_ordered = ['GDP_US',
#                   'Infl_US',
#                   'FFR',
#                   'TS10Y2Y_US'
#                   ]

# df_var_us = df_us[var_us_ordered]


# model_us = VAR(df_var_us)
# print(model_us.select_order())

# result = model_us.fit(ic='aic')
# result.summary()

# # print(result.test_whiteness())
# print(result.is_stable())

# residuals = result.sigma_u
# resid_chol_decomp = np.linalg.cholesky(residuals)
# np.linalg.eigvals(resid_chol_decomp)



# # IRFs
# irfs_us = result.irf(20)
# irfs_us.plot(orth=True, signif=0.16)
# plt.show()
