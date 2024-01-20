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
import sdmx


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



# Short Term Interest Rate
ea_rate_3m = fred.get_series('IR3TIB01EZM156N')
ea_rate_3m.rename('EA_Rate_3M', inplace=True)


# Stock Market
eustx_50 = yf.download('^GDAXI',
                       start='1990-01-01',
                       end='2024-01-01')

eustx_50_m = eustx_50.resample('M', loffset='1d').mean()


# Financial Stress (VSTOXX) -> geht nur bis 2016!
# -> Alternative: VIX fuer US & EA?
# os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data')
# vstoxx = pd.read_csv('VSTOXX.csv',
#                      skiprows=2, 
#                      index_col=[0],
#                      parse_dates=True,
#                      dayfirst=True)

# vstoxx_m = vstoxx.resample('M', loffset='1d').mean()

# vstoxx.to_excel('VSTOXX_D_Excel.xlsx')
# vstoxx_m.to_excel('VSTOXX_M_Excel.xlsx')



# Yields

ecb = sdmx.Client('ECB')
def get_column(series_key):
    column = series_key.attrib.TITLE.value
    return column if len(column) <= 90 else column[:90] + '...'

def get_unit(data_message):
    unit_codelist = data_message.structure.attributes.get('UNIT').local_representation.enumerated
    series_key = next(iter(data_message.data[0].series))
    return unit_codelist[series_key.attrib.UNIT.value].name.localized_default()

def get_datasets(client, dataflow_id, keys=None, startPeriod=None, endPeriod=None, data_structure_definition=None):
    data_message = client.data(dataflow_id, key=keys, params={'startPeriod': startPeriod, 'endPeriod': endPeriod})

    df = sdmx.to_pandas(data_message, datetime={'dim': 'TIME_PERIOD'})
    columns = [get_column(series_key) for series_key in data_message.data[0].series]

    return df, columns


yc_df, yc_columns, yc_unit = get_datasets(
    ecb,
    'YC',
    keys={'DATA_TYPE_FM': ['SR_10Y', 'SR_1Y'], 'INSTRUMENT_FM': ['G_N_A'], 'FREQ': 'B'},
    startPeriod='2012',
)
yc_df


entrypoint = 'https://data-api.ecb.europa.eu/service'
resource = 'data'
flowRef = 'YC'
form = '?format=csvdata'


# Yield 3 M
key = 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M'

request_url = entrypoint + '/' + resource + '/'+ flowRef + '/' + key + form
response = req.get(request_url)

yield_3m_ea = pd.read_csv(io.StringIO(response.text),
                          usecols=['TIME_PERIOD', 'OBS_VALUE', 'DATA_TYPE_FM'],
                          index_col=['TIME_PERIOD'],
                          infer_datetime_format=True)

yield_3m_ea.index = pd.to_datetime(yield_3m_ea.index)

# Yield 2Y
key = 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y'

request_url = entrypoint + '/' + resource + '/'+ flowRef + '/' + key + form
response = req.get(request_url)

yield_2y_ea = pd.read_csv(io.StringIO(response.text),
                          usecols=['TIME_PERIOD', 'OBS_VALUE', 'DATA_TYPE_FM'],
                          index_col=['TIME_PERIOD'],
                          infer_datetime_format=True)

yield_2y_ea.index = pd.to_datetime(yield_2y_ea.index)

# Yield 5Y
key = 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y'

request_url = entrypoint + '/' + resource + '/'+ flowRef + '/' + key + form
response = req.get(request_url)

yield_5y_ea = pd.read_csv(io.StringIO(response.text),
                          usecols=['TIME_PERIOD', 'OBS_VALUE', 'DATA_TYPE_FM'],
                          index_col=['TIME_PERIOD'],
                          infer_datetime_format=True)

yield_5y_ea.index = pd.to_datetime(yield_5y_ea.index)



# Yield 10Y
key = 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y'

request_url = entrypoint + '/' + resource + '/'+ flowRef + '/' + key + form
response = req.get(request_url)

yield_10y_ea = pd.read_csv(io.StringIO(response.text),
                          usecols=['TIME_PERIOD', 'OBS_VALUE', 'DATA_TYPE_FM'],
                          index_col=['TIME_PERIOD'],
                          infer_datetime_format=True)

yield_10y_ea.index = pd.to_datetime(yield_10y_ea.index)


# Yieldcurve Factors
os.chdir(r'C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data')
factors_ea = pd.read_csv('ECB_data_full.csv',
                         usecols=['DATA_TYPE_FM', 'TIME_PERIOD', 'OBS_VALUE'],
                         index_col=['TIME_PERIOD'],
                         infer_datetime_format=True)

factors_ea.index = pd.to_datetime(factors_ea.index)

factors_ea_sub = factors_ea.loc[factors_ea['DATA_TYPE_FM'].isin(['BETA0', 
                                                                 'BETA1', 
                                                                 'BETA2', 
                                                                 'TAU1'])]


for coeff in factors_ea_sub['DATA_TYPE_FM'].unique():
    factors_ea_sub.loc[factors_ea_sub['DATA_TYPE_FM'] == str(coeff)].plot()

