import pandas as pd
import numpy as np
import random
from datetime import date, datetime
from pandas.tseries.frequencies import to_offset
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
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

import filterpy  # for Kalman Filter
from nelson_siegel_svensson.calibrate import *
import rpy2
import io
import sdmx


##### Data
# Industrial Production
ind_pro_ea = fred.get_series("EA19PRINTO01GYSAM")
ind_pro_ea.rename("INDPRO_EA", inplace=True)


# CPI
cpi_ea = fred.get_series("CP0000EZ19M086NEST")
cpi_ea.rename("CPI_EA", inplace=True)

# Inflation
infl_ea = fred.get_series("CPHPTT01EZM659N")
infl_ea.rename("Infl_EA", inplace=True)


# Short Term Interest Rate
ea_rate_3m = fred.get_series("IR3TIB01EZM156N")
ea_rate_3m.rename("EA_Rate_3M", inplace=True)


# Stock Market
eustx_50 = yf.download("^GDAXI", start="1990-01-01", end="2024-01-01")

eustx_50_m = eustx_50["Close"].resample("M", loffset="1d").mean()
eustx_50_m.name = "EUSTX_50"

eustx_50_m_ret = eustx_50_m.pct_change(periods=12).dropna() * 100
eustx_50_m_ret.name = "EUSTX_50_YoY"


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

entrypoint = "https://data-api.ecb.europa.eu/service"
resource = "data"
flowRef = "YC"
form = "?format=csvdata"


# Yield 3 M
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_3m_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_3m_ea.index = pd.to_datetime(yield_3m_ea.index)

yield_3m_ea.rename(columns={"OBS_VALUE": "Y3M"}, inplace=True)

yield_3m_ea_m = yield_3m_ea.resample("M", loffset="1d").mean()

# Yield 1Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_1y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_1y_ea.index = pd.to_datetime(yield_1y_ea.index)

yield_1y_ea.rename(columns={"OBS_VALUE": "Y1Y"}, inplace=True)


yield_1y_ea_m = yield_1y_ea.resample("M", loffset="1d").mean()


# Yield 2Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_2y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_2y_ea.index = pd.to_datetime(yield_2y_ea.index)

yield_2y_ea.rename(columns={"OBS_VALUE": "Y2Y"}, inplace=True)


yield_2y_ea_m = yield_2y_ea.resample("M", loffset="1d").mean()

# Yield 5Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_5y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_5y_ea.index = pd.to_datetime(yield_5y_ea.index)

yield_5y_ea.rename(columns={"OBS_VALUE": "Y5Y"}, inplace=True)


yield_5y_ea_m = yield_5y_ea.resample("M", loffset="1d").mean()

# Yield 7Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_7Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_7y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_7y_ea.index = pd.to_datetime(yield_7y_ea.index)

yield_7y_ea.rename(columns={"OBS_VALUE": "Y7Y"}, inplace=True)


yield_7y_ea_m = yield_7y_ea.resample("M", loffset="1d").mean()

# Yield 10Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_10y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_10y_ea.index = pd.to_datetime(yield_10y_ea.index)

yield_10y_ea.rename(columns={"OBS_VALUE": "Y10Y"}, inplace=True)


yield_10y_ea_m = yield_10y_ea.resample("M", loffset="1d").mean()


# Merge Yield Data for Nelson-Siegel decomposition in R
start_yields_ea = max(
    min(yield_3m_ea_m.index),
    min(yield_1y_ea_m.index),
    min(yield_2y_ea_m.index),
    min(yield_5y_ea_m.index),
    min(yield_7y_ea_m.index),
    min(yield_10y_ea_m.index),
)


end_yields_ea = min(
    max(yield_3m_ea_m.index),
    max(yield_1y_ea_m.index),
    max(yield_2y_ea_m.index),
    max(yield_5y_ea_m.index),
    max(yield_7y_ea_m.index),
    max(yield_10y_ea_m.index),
)


yields_ea_m = [
    yield_3m_ea_m.loc[start_yields_ea:end_yields_ea, "Y3M"],
    yield_1y_ea_m.loc[start_yields_ea:end_yields_ea, "Y1Y"],
    yield_2y_ea_m.loc[start_yields_ea:end_yields_ea, "Y2Y"],
    yield_5y_ea_m.loc[start_yields_ea:end_yields_ea, "Y5Y"],
    yield_7y_ea_m.loc[start_yields_ea:end_yields_ea, "Y7Y"],
    yield_10y_ea_m.loc[start_yields_ea:end_yields_ea, "Y10Y"],
]

yields_ea_m = pd.concat(yields_ea_m, axis=1)

os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data")
yields_ea_m.to_csv("Yields_EA.csv")


# Yieldcurve Factors ECB
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data")
factors_ea = pd.read_csv(
    "ECB_data_full.csv",
    usecols=["DATA_TYPE_FM", "TIME_PERIOD", "OBS_VALUE"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

factors_ea.index = pd.to_datetime(factors_ea.index)

factors_ea_sub = factors_ea.loc[
    factors_ea["DATA_TYPE_FM"].isin(["BETA0", "BETA1", "BETA2", "TAU1"])
]


# Beta 0 - Level Factor
beta_0 = pd.DataFrame(
    factors_ea.loc[factors_ea["DATA_TYPE_FM"] == "BETA0", "OBS_VALUE"]
)
beta_0.rename(columns={"OBS_VALUE": "Level Factor"}, inplace=True)

beta_0_m = beta_0.resample("M", loffset="1d").mean()

# Beta 0 Approximation
beta_0_m["y(3) + y(24) + y(120)/3"] = np.nan
for t in beta_0_m.index:
    yield_3m = yield_3m_ea_m.loc[t, "Y3M"]
    yield_2y = yield_2y_ea_m.loc[t, "Y2Y"]
    yield_10y = yield_10y_ea_m.loc[t, "Y10Y"]

    beta_0_m.loc[t, "y(3) + y(24) + y(120)/3"] = (yield_3m + yield_2y + yield_10y) / 3


# Beta 1 - Slope Factor
beta_1 = pd.DataFrame(
    factors_ea.loc[factors_ea["DATA_TYPE_FM"] == "BETA1", "OBS_VALUE"]
)
beta_1.rename(columns={"OBS_VALUE": "Slope Factor"}, inplace=True)

beta_1_m = beta_1.resample("M", loffset="1d").mean()


# Beta 1 Approximation
beta_1_m["y(3) - y(120)"] = np.nan

for t in beta_1_m.index:
    spread = yield_3m_ea_m.loc[t, "Y3M"] - yield_10y_ea_m.loc[t, "Y10Y"]
    beta_1_m.loc[t, "y(3) - y(120)"] = spread


# Beta 2 - Curvature Factor
beta_2 = pd.DataFrame(
    factors_ea.loc[factors_ea["DATA_TYPE_FM"] == "BETA2", "OBS_VALUE"]
)
beta_2.rename(columns={"OBS_VALUE": "Curvature Factor"}, inplace=True)

beta_2_m = beta_2.resample("M", loffset="1d").mean()


# Beta 2 Approximation
beta_2_m["Curvature_Approx"] = np.nan

for t in beta_2_m.index:
    curvat_approx = (
        2 * yield_2y_ea_m.loc[t, "Y2Y"]
        - yield_10y_ea_m.loc[t, "Y10Y"]
        - yield_3m_ea_m.loc[t, "Y3M"]
    )
    beta_2_m.loc[t, "Curvature_Approx"] = curvat_approx


# Yieldcurve Factors own Calculation (in R using ECB Spot Yields)
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data")

yields_ea_m_r = pd.read_csv(
    "Yields_EA_R.csv", index_col=[0], parse_dates=True, infer_datetime_format=True
)


# Merge EA Data
start_ea = max(
    min(ind_pro_ea.index),
    min(infl_ea.index),
    min(ea_rate_3m.index),
    min(eustx_50_m_ret.index),
    min(yields_ea_m_r.index),
)


end_ea = min(
    max(ind_pro_ea.index),
    max(infl_ea.index),
    max(ea_rate_3m.index),
    max(eustx_50_m_ret.index),
    max(yields_ea_m_r.index),
)


df_ea = [
    ind_pro_ea[start_ea:end_ea],
    infl_ea[start_ea:end_ea],
    ea_rate_3m[start_ea:end_ea],
    eustx_50_m_ret[start_ea:end_ea],
    yields_ea_m_r[start_ea:end_ea],
]

df_ea = pd.concat(df_ea, axis=1).dropna()

##### Plots
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis")

# Factors ECB
plt.figure(figsize=(15, 10))
plt.plot(beta_0_m.loc[:, "Level Factor"], label="Level Factor", color="b")

plt.plot(
    beta_1_m.loc[:, "Slope Factor"],
    label="Slope Factor",
    color="orange",
    linestyle="--",
)

plt.plot(
    beta_2_m.loc[:, "Curvature Factor"],
    label="Curvature Factor",
    color="g",
    linestyle=":",
)

plt.legend()
plt.savefig("Factors_Figure_EA.pdf", dpi=1000)
plt.show()


# Beta 0 & Approximation & Inflation
plt.figure(figsize=(15, 10))
plt.plot(beta_0_m.loc[:, "Level Factor"], label="Level Factor", color="b")

plt.plot(
    beta_0_m.loc[:, "y(3) + y(24) + y(120)/3"],
    label="y(3) + y(24) + y(120)/3",
    linestyle="--",
    color="orange",
)

plt.plot(infl_ea.loc["2004-10-01":], label="Inflation EA", linestyle=":", color="g")

plt.legend()
plt.savefig("Beta_0_Figure_EA.pdf", dpi=1000)
plt.show()


# Beta 1 & Approximation
plt.figure(figsize=(15, 10))
plt.plot(beta_1_m.loc[:, "Slope Factor"], label="Slope Factor", color="b")

plt.plot(
    beta_1_m.loc[:, "y(3) - y(120)"],
    label="y(3) - y(120)",
    color="orange",
    linestyle="--",
)

plt.legend()
plt.savefig("Beta_1_Figure_EA.pdf", dpi=1000)
plt.show()


# Beta 2 & Approximation
plt.figure(figsize=(15, 10))
plt.plot(beta_2_m.loc[:, "Curvature Factor"], label="Curvature Factor", color="c")

plt.plot(
    beta_2_m.loc[:, "Curvature_Approx"],
    label="2 * y(24) - y(120) - y(3)",
    color="red",
    linestyle="--",
)

plt.legend()
plt.savefig("Beta_2_Figure_EA.pdf", dpi=1000)
plt.show()


for coeff in factors_ea_sub["DATA_TYPE_FM"].unique():
    factors_ea_sub.loc[factors_ea_sub["DATA_TYPE_FM"] == str(coeff)].plot()


# Plots Own Calculation
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis")
# Factors
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r["beta_0"], label="Level Factor", color="b")
plt.plot(yields_ea_m_r["beta_1"], label="Slope Factor", color="orange", linestyle="--")
plt.plot(
    yields_ea_m_r["beta_2"], label="Curvature Factor", color="green", linestyle=":"
)

plt.legend()
plt.savefig("Factors_Figure_EA_1.pdf", dpi=1000)
plt.show()

# Beta 0
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r["beta_0"], label="Level Factor", color="b")
plt.plot(
    beta_0_m.loc[:, "y(3) + y(24) + y(120)/3"],
    label="y(3) + y(24) + y(120)/3",
    linestyle="--",
    color="orange",
)
plt.plot(infl_ea.loc["2004-10-01":], label="Inflation EA", linestyle=":", color="g")

plt.legend()
plt.savefig("Beta_0_Figure_EA_1.pdf", dpi=1000)
plt.show()

# Beta 1
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r["beta_1"], label="Slope Factor", color="b")

plt.plot(
    beta_1_m.loc[:, "y(3) - y(120)"],
    label="y(3) - y(120)",
    color="orange",
    linestyle="--",
)

plt.legend()
plt.savefig("Beta_1_Figure_EA_1.pdf", dpi=1000)
plt.show()


# Beta 2
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r["beta_2"], label="Curvature Factor", color="c")

plt.plot(
    beta_2_m.loc[:, "Curvature_Approx"],
    label="2 * y(24) - y(120) - y(3)",
    color="red",
    linestyle="--",
)

plt.legend()
plt.savefig("Beta_2_Figure_EA_1.pdf", dpi=1000)
plt.show()
