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
import yfinance as yf
from fredapi import Fred
fred = Fred(api_key="ef7244731efdde9698fef5d547b7094f")
import filterpy # for Kalman Filter



# Functions
# Plot Function
def plot_data(data):
    for col in data.columns:
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes()
        ax.plot(data[col])
        ax.set_title(col)
        ax.legend()
        plt.show()

    

# ADF Test Function
adf_dict = {}        

def get_adf(data):
    for col in data.columns:
        result = adfuller(data[col])
        p_val = result[1]
        adf_dict[col] = p_val
        print(str(col) + ': ' + str(p_val))



# Get Data
# Data
# GDP
gdp_us = fred.get_series('GDPC1')
gdp_us.rename('GDP_US', inplace=True)


# Industrial Production
ind_pro_us = fred.get_series('INDPRO')
ind_pro_us.rename('INDPRO', inplace=True)


# CPI
cpi_us = fred.get_series('CPIAUCSL') # convert to YoY rate
cpi_us.rename('Infl_US', inplace=True)
infl_us = cpi_us.pct_change(periods=12).dropna() *100

infl_us.plot()


# Investment
inv_us = fred.get_series('GPDIC96')
inv_us.rename('Inv_US', inplace=True)


# Cpaital Utilization
cap_util_us = fred.get_series('TCU')
cap_util_us.rename('CU_US', inplace=True)


# FFR
ffr = fred.get_series('DFF') 
ffr.rename('FFR', inplace=True)
ffr_m = ffr.resample('M', loffset='1d').mean() # convert daily to monthly!!!


# T-Bill 3M
tb_3m = fred.get_series('TB3MS')
tb_3m.rename('TB_3M', inplace=True)


# T-Note 1Y
tb_1y = fred.get_series('DGS1')
tb_1y.rename('DGS1', inplace=True)
tb_1y_m = tb_1y.resample('M', loffset='1d').mean()


# T-Bond 10Y
tb_10y = fred.get_series('DGS10')
tb_10y.rename('DGS10', inplace=True)
tb_10y_m = tb_10y.resample('M', loffset='1d').mean()


# Term Spread
ts_10y3m_us = fred.get_series('T10Y3M')
ts_10y3m_us.rename('TS10Y3M_US', inplace=True)

ts_10y2y_us = fred.get_series('T10Y2Y')
ts_10y2y_us.rename('TS10Y2Y_US', inplace=True)

# y_12m = fred.get_series('TY12MCD')


# S&P 500
sp_500_tckr = yf.Ticker('^GSPC')
sp_500 = sp_500_tckr.history(period='max')
sp_500_1 = yf.download('^GSPC', 
                       start='1970-01-01',
                       end='2023-12-01')
sp_500_1_m = sp_500_1.resample('M', loffset='1d').mean()


# Financial Stress Variables
# Excess Bond Premium (EBP)


# Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
corp_spread = fred.get_series('BAA10Y')
corp_spread.rename('BAA10Y', inplace=True)
corp_spread_m = corp_spread.resample('M', loffset='1d').mean()


# Get Period
start_us = max(min(gdp_us.index),
               min(infl_us.index),
               min(ffr_m.index),
               min(cap_util_us.index),
               min(ts_10y2y_us.index))



end_us = min(max(gdp_us.index),
               max(infl_us.index),
               max(ffr_m.index),
               max(cap_util_us.index),
               max(ts_10y2y_us.index))




# Merge Data
df_us = [gdp_us[start_us:end_us],
         infl_us[start_us:end_us],
         ffr_m[start_us:end_us],
         cap_util_us[start_us:end_us],
         ts_10y2y_us[start_us:end_us]]

df_us = pd.concat(df_us, axis=1).dropna()



# Analysis

# Stationarity Check
get_adf(df_us)



# VAR
var_us_ordered = ['GDP_US',
                  'Infl_US',
                  'FFR',
                  'TS10Y2Y_US'
                  ]

df_var_us = df_us[var_us_ordered]


model_us = VAR(df_var_us)
print(model_us.select_order())

result = model_us.fit(ic='aic')
result.summary()

# print(result.test_whiteness())
print(result.is_stable())

residuals = result.sigma_u
resid_chol_decomp = np.linalg.cholesky(residuals)
np.linalg.eigvals(resid_chol_decomp)



# IRFs
irfs_us = result.irf(20)
irfs_us.plot(orth=True, signif=0.16)
plt.show()
