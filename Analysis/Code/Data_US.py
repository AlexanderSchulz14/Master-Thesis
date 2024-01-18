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
from nelson_siegel_svensson.calibrate import *
import rpy2



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
os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data')
ebp = pd.read_csv('ebp_csv.csv', 
                  index_col=[0], 
                  parse_dates=True)


# VIX? (von Bloomberg?)


# Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
corp_spread = fred.get_series('BAA10Y')
corp_spread.rename('BAA10Y', inplace=True)
corp_spread_m = corp_spread.resample('M', loffset='1d').mean()


# Yield Data
os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data')
yields_us = pd.read_excel('LW_monthly.xlsx',
                          skiprows=8)

yields_us.rename(columns={yields_us.columns[0]:'Date'}, inplace=True)

yields_us['Date'] = pd.to_datetime(yields_us['Date'],
                                   format='%Y%m')

yields_us.set_index([yields_us.columns[0]], inplace=True)

yields_us.columns = [col.replace(' ', '') for col in yields_us.columns]

# Subset Yield Data as not all maturities are available for each point in time
start_yields_us = '1978-01-01'
yields_us_sub = yields_us.loc[start_yields_us:]
yields_us_sub.to_csv("Yields_Data_Subset.csv")



# Beta Coefficients from R
yields_us_sub_r = pd.read_csv('Yields_US_R.csv',
                              index_col=[0],
                              parse_dates=True)



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



# # Plots
# x = ['3m', '6m', '12m', '24m', '36m','60m', '120m']
# y = yields_us_sub.index
# Z = yields_us_sub[x].to_numpy()

# # X, Y = np.meshgrid(x, y)
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(X, Y, Z, cmap='viridis')

# # # Set labels
# # ax.set_xticks(x)
# # ax.set_xticklabels(yields_us_sub.index.strftime('%Y-%m-%d'), rotation=45, ha='right')
# # ax.set_xlabel('Date')


# fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
# fig.update_layout(title='Yield Curve',
#                   scene = {"aspectratio": {"x": 1, "y": 1, "z": 0.75}})
# fig.show()



# Nelson Siegel Decomposition playing around
# maturities_to_use = ['1m', '3m', 
#                      '6m', '12m', 
#                      '24m', '60m',
#                      '90m', '120m']


maturities_to_use = ['3m', '6m', 
                     '9m', '12m', 
                     '15m', '18m',
                     '21m', '24m',
                     '30m', '36m',
                     '48m', '60m',
                     '72m', '84m',
                     '96m', '108m',
                     '120m']


maturities_float = [float(mat.replace('m', '')) for mat in maturities_to_use]
maturities_float_year = [mat/12 for mat in maturities_float]

date = '2019-01-01'
test_data = yields_us_sub.loc[date, maturities_to_use]




t = np.array(maturities_float_year)
y = test_data.values

curve, status = calibrate_ns_ols(t, y, tau0=1.0)
assert status.success
print(curve)


errors = []
beta0_ls = {}

for date in yields_us_sub.index:
    
    try:
        decomp_data = yields_us_sub.loc[date, maturities_to_use]
        
        t = np.array(maturities_float_year)
        y = decomp_data.values
        
        curve, status = calibrate_ns_ols(t, y, tau0=1.0)
        assert status.success
        print(curve.beta0)
        beta0_ls[date] = curve.beta0
        # yields_us_sub.loc[date, 'beta0'] = curve.beta0
        
    except:
        errors.append(date)
        # yields_us_sub.loc[date, 'beta0'] = 'NA'
    
    
    
    sorted(beta0_ls.items(), key=lambda x:x[1], reverse=True)




keys = list(beta0_ls.keys())
values = list(beta0_ls.values())

plt.plot(keys, values, linestyle='-')
plt.xlabel('Date')
plt.ylabel('Beta_0')
# plt.title('Dictionary Values Line Plot')
plt.grid(True)
plt.show()









