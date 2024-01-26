import pandas as pd
import numpy as np
import random
from datetime import date, datetime
from pandas.tseries.frequencies import to_offset
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
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
import io








# Auxiliary Functions
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


# Coefficient Approximation Function
def get_beta_0_approx(data, time, 
                  yield_1='3m', 
                  yield_2='24m', 
                  yield_3='120m'):
    calc_data = data.loc[time, [yield_1, yield_2, yield_3]]
    beta_0_approx = calc_data.mean()
    
    return beta_0_approx
    
    






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
# sp_500_tckr = yf.Ticker('^GSPC')
# sp_500 = sp_500_tckr.history(period='max')
sp_500_1 = yf.download('^GSPC', 
                       start='1970-01-01',
                       end='2023-01-01')
sp_500_1_m = sp_500_1.resample('M', loffset='1d').mean()


# Financial Stress Variables
# Excess Bond Premium (EBP)
os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data')
ebp = pd.read_csv('ebp_csv.csv', 
                  index_col=[0], 
                  parse_dates=True)


# VIX? (von Bloomberg?)
vix = yf.download('^VIX',
                  start='1990-01-02',
                  end='2024-01-01')
vix_m = vix.resample('M', loffset='1d').mean()



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
start_yields_us = '1972-01-01'
yields_us_sub = yields_us.loc[start_yields_us:]
yields_us_sub.to_csv("Yields_Data_Subset.csv")



# Beta Coefficients from R
yields_us_sub_r = pd.read_csv('Yields_US_R.csv',
                              index_col=[0],
                              parse_dates=True)

yields_us_sub_r.rename(columns={'beta_0' : 'Level Factor',
                                'beta_1' : 'Slope Factor',
                                'beta_2' : 'Curvature Factor'},
                       inplace=True)


# Get Approximations for Coefficients (see Diebold et al. (2006))
# Beta 0 - Level Factor
yields_us_sub_r['y(3) + y(24) + y(120)/3'] = np.nan
for t in yields_us_sub_r.index:
    result_beta_0 = get_beta_0_approx(yields_us_sub_r, t)
    yields_us_sub_r.loc[t, 'y(3) + y(24) + y(120)/3'] = result_beta_0
    

# Beta 1 - Slope Factor
yields_us_sub_r['y(3) - y(120)'] = yields_us_sub_r['3m'] - yields_us_sub_r['120m']


# Beta 2 - Curvature Factor
yields_us_sub_r['2 * y(24) - y(120) - y(3)'] =  2 * yields_us_sub_r['24m'] - yields_us_sub_r['120m'] - yields_us_sub_r['3m']


# Columns to use for merge
yield_cols_to_use = ['3m', '6m', 
                     '9m', '12m', 
                     '15m', '18m',
                     '21m', '24m',
                     '30m', '36m',
                     '48m', '60m',
                     '72m', '84m',
                     '96m', '108m',
                     '120m', 'Level Factor',
                     'Slope Factor', 'Curvature Factor',
                     'y(3) + y(24) + y(120)/3',
                     'y(3) - y(120)',
                     '2 * y(24) - y(120) - y(3)']



# Get Period
start_us = max(
    # min(gdp_us.index),
    min(ind_pro_us.index),
    min(infl_us.index),
    min(ffr_m.index),
    # min(cap_util_us.index),
    min(yields_us_sub_r.index),
    min(tb_3m.index),
    min(sp_500_1_m.index),
    min(ebp['ebp'].index)
#    min(vix_m.index),
#    min(ts_10y2y_us.index)
               )



end_us = min(
    # max(gdp_us.index),
    max(ind_pro_us.index),
    max(infl_us.index),
    max(ffr_m.index),
    # max(cap_util_us.index),
    max(yields_us_sub_r.index),
    max(tb_3m.index),
    max(sp_500_1_m.index),
    max(ebp['ebp'].index)
#    max(vix_m.index),
#    max(ts_10y2y_us.index)
               )



# Merge Data
df_us = [
    # gdp_us[start_us:end_us],
    ind_pro_us[start_us:end_us],
    infl_us[start_us:end_us],
    ffr_m[start_us:end_us],
    # cap_util_us[start_us:end_us],
    yields_us_sub_r.loc[start_us:end_us, yield_cols_to_use],
    tb_3m[start_us:end_us],
    sp_500_1_m.loc[start_us:end_us, 'Close'],
    ebp.loc[start_us:end_us, 'ebp']
#  ts_10y2y_us[start_us:end_us]
         ]

df_us = pd.concat(df_us, axis=1).dropna()








# Plots
# All Coefficients
os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis')

plt.figure(figsize=(15,10))
sns.lineplot(df_us[['Level Factor', 'Slope Factor', 'Curvature Factor']])
plt.legend(loc='lower right')

plt.savefig('Factor_Figure.pdf', dpi=1000)
plt.show()

# Beta 0 Level Factor & Inflation
plt.figure(figsize=(15,10))
sns.lineplot(df_us[['Level Factor', 'y(3) + y(24) + y(120)/3', 'Infl_US']],
             legend='auto',
             palette=['blue', 'orange', 'green'])
plt.legend(loc='lower right')

plt.savefig('Beta_0_Figure.pdf', dpi=1000)
plt.show()

# Correlation
df_us['Level Factor'].corr(df_us['y(3) + y(24) + y(120)/3'])

df_us['Level Factor'].corr(df_us['Infl_US'])



# Beta 1 Slope Factor
plt.figure(figsize=(15,10))
sns.lineplot(df_us[['Slope Factor', 'y(3) - y(120)']])
plt.legend(loc='lower right')

plt.savefig('Beta_1_Figure.pdf', dpi=1000)
plt.show()

# Correlation
df_us['Slope Factor'].corr(df_us['y(3) - y(120)'])



# Beta 2 - Curvature Factor
plt.figure(figsize=(15,10))
sns.lineplot(df_us[['Curvature Factor', '2 * y(24) - y(120) - y(3)']],
             palette=['c', 'r'])
plt.legend(loc='lower right')

plt.savefig('Beta_2_Figure.pdf', dpi=1000)
plt.show()

# Correlation
df_us['Curvature Factor'].corr(df_us['2 * y(24) - y(120) - y(3)'])


