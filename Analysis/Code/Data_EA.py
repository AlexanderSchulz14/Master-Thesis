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
from googlefinance.get import get_code

from fredapi import Fred

fred = Fred(api_key="ef7244731efdde9698fef5d547b7094f")

from MA_functions import *

import filterpy  # for Kalman Filter
from nelson_siegel_svensson.calibrate import *
import rpy2
import io
import sdmx

# Paths
data_path_ma = r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data"


##### Data
# Industrial Production
# FRED: EA19PRINTO01GYSAM
# OECD: OECDPRINTO01GYSAM
ind_pro_ea = fred.get_series("OECDPRINTO01GYSAM")
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

# Recession Indicator
ea_rec = fred.get_series("EUROREC").dropna()
ea_rec.rename("EA_REC", inplace=True)


# Stock Market
# eustx_50 = yf.download("^STOXX50E", start="2000-01-01", end="2024-01-01")

# eustx_50_m = eustx_50["Close"].resample("M", loffset="1d").mean()
# eustx_50_m.name = "EUSTX_50"

# eustx_50_m_ret = eustx_50_m.pct_change(periods=12).dropna() * 100
# eustx_50_m_ret.name = "EUSTX_50_YoY"

eustx_50_m = pd.read_csv(data_path_ma + "\\" + "EUSTX_50_Data.csv", index_col=[0])
eustx_50_m.index = pd.to_datetime(eustx_50_m.index)
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

# Bloomberg BDH Abfrage
# vix_data = pd.read_excel(data_path_ma + "\\" "VIX_Data.xlsx", skiprows=5, index_col=0)

# vix_data.rename(
#     columns={vix_data.columns[0]: "VIX", vix_data.columns[1]: "VSTOXX"}, inplace=True
# )

# vix_data.to_csv(data_path_ma + "\\" + "VIX_VSTOXX_Data.csv")

# vix_data = vix_data["2004":]


# vix_data.index = vix_data.index + pd.offsets.MonthBegin(1)

# Data ohne Bloomber BDH Formel
vix_data = pd.read_csv(data_path_ma + "\\" + "VIX_VSTOXX_Data.csv", index_col=0)

vix_data.index = pd.to_datetime(vix_data.index)

vix_data.index = vix_data.index + pd.offsets.MonthBegin(1)


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

# Yield 6M
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_6M"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_6m_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_6m_ea.index = pd.to_datetime(yield_6m_ea.index)

yield_6m_ea.rename(columns={"OBS_VALUE": "Y6M"}, inplace=True)

yield_6m_ea_m = yield_6m_ea.resample("M", loffset="1d").mean()

# Yield 9M
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_9M"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_9m_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_9m_ea.index = pd.to_datetime(yield_9m_ea.index)

yield_9m_ea.rename(columns={"OBS_VALUE": "Y9M"}, inplace=True)

yield_9m_ea_m = yield_9m_ea.resample("M", loffset="1d").mean()

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

# Yield 3Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_3y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_3y_ea.index = pd.to_datetime(yield_3y_ea.index)

yield_3y_ea.rename(columns={"OBS_VALUE": "Y3Y"}, inplace=True)


yield_3y_ea_m = yield_3y_ea.resample("M", loffset="1d").mean()

# Yield 4Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_4Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_4y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_4y_ea.index = pd.to_datetime(yield_4y_ea.index)

yield_4y_ea.rename(columns={"OBS_VALUE": "Y4Y"}, inplace=True)


yield_4y_ea_m = yield_4y_ea.resample("M", loffset="1d").mean()

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

# Yield 6Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_6Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_6y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_6y_ea.index = pd.to_datetime(yield_6y_ea.index)

yield_6y_ea.rename(columns={"OBS_VALUE": "Y6Y"}, inplace=True)


yield_6y_ea_m = yield_6y_ea.resample("M", loffset="1d").mean()

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

# Yield 8Y
key = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_8Y"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

yield_8y_ea = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

yield_8y_ea.index = pd.to_datetime(yield_8y_ea.index)

yield_8y_ea.rename(columns={"OBS_VALUE": "Y8Y"}, inplace=True)


yield_8y_ea_m = yield_8y_ea.resample("M", loffset="1d").mean()

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


########## Financial Stress ##########
entrypoint = "https://data-api.ecb.europa.eu/service"
resource = "data"
flowRef = "CISS"
form = "?format=csvdata"

key = "M.U2.Z0Z.4F.EC.SOV_GDPW.IDX"

request_url = entrypoint + "/" + resource + "/" + flowRef + "/" + key + form
response = req.get(request_url)

ciss_idx = pd.read_csv(
    io.StringIO(response.text),
    usecols=["TIME_PERIOD", "OBS_VALUE"],
    index_col=["TIME_PERIOD"],
    infer_datetime_format=True,
)

ciss_idx.index = pd.to_datetime(ciss_idx.index)

ciss_idx.rename(columns={"OBS_VALUE": "CISS"}, inplace=True)


# Extract the Series from the DataFrames
ciss_series = ciss_idx.loc[:"2024-01-01", "CISS"]
vix_series = vix_data.loc["2000-09-01":, "VSTOXX"]

# Compute the Pearson correlation
correlation, p_value = pearsonr(ciss_series, vix_series)

fig, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()

ax1.plot(vix_series, color="#69b3a2", label="VSTOXX")

ax2.plot(ciss_series, color="#3399e6", label="CISS")

# Adding labels
ax1.set_xlabel("Date")
ax1.set_ylabel("VSTOXX", color="#69b3a2")
ax2.set_ylabel("CISS", color="#3399e6")

# Adding legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.show()


# plt.figure(figsize=(15, 10))
# plt.plot(ciss_idx.loc[:"2024-01-01", ["CISS"]] * 100, label="CISS")
# plt.plot(vix_data.loc["2000-09-01":, ["VSTOXX"]], label="VSTOXX")

# plt.legend()
# plt.show()

# Merge Yield Data for Nelson-Siegel decomposition in R
start_yields_ea = max(
    min(yield_3m_ea_m.index),
    min(yield_6m_ea_m.index),
    min(yield_9m_ea_m.index),
    min(yield_1y_ea_m.index),
    min(yield_2y_ea_m.index),
    min(yield_3y_ea_m.index),
    min(yield_4y_ea_m.index),
    min(yield_5y_ea_m.index),
    min(yield_6y_ea_m.index),
    min(yield_7y_ea_m.index),
    min(yield_8y_ea_m.index),
    min(yield_10y_ea_m.index),
)


end_yields_ea = min(
    max(yield_3m_ea_m.index),
    max(yield_6m_ea_m.index),
    max(yield_9m_ea_m.index),
    max(yield_1y_ea_m.index),
    max(yield_2y_ea_m.index),
    max(yield_3y_ea_m.index),
    max(yield_4y_ea_m.index),
    max(yield_5y_ea_m.index),
    max(yield_6y_ea_m.index),
    max(yield_7y_ea_m.index),
    max(yield_8y_ea_m.index),
    max(yield_10y_ea_m.index),
)


yields_ea_m = [
    yield_3m_ea_m.loc[start_yields_ea:end_yields_ea, "Y3M"],
    yield_6m_ea_m.loc[start_yields_ea:end_yields_ea, "Y6M"],
    yield_9m_ea_m.loc[start_yields_ea:end_yields_ea, "Y9M"],
    yield_1y_ea_m.loc[start_yields_ea:end_yields_ea, "Y1Y"],
    yield_2y_ea_m.loc[start_yields_ea:end_yields_ea, "Y2Y"],
    yield_3y_ea_m.loc[start_yields_ea:end_yields_ea, "Y3Y"],
    yield_4y_ea_m.loc[start_yields_ea:end_yields_ea, "Y4Y"],
    yield_5y_ea_m.loc[start_yields_ea:end_yields_ea, "Y5Y"],
    yield_6y_ea_m.loc[start_yields_ea:end_yields_ea, "Y6Y"],
    yield_7y_ea_m.loc[start_yields_ea:end_yields_ea, "Y7Y"],
    yield_8y_ea_m.loc[start_yields_ea:end_yields_ea, "Y8Y"],
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
beta_0_m["(y(3) + y(24) + y(120))/3"] = np.nan
for t in beta_0_m.index:
    yield_3m = yield_3m_ea_m.loc[t, "Y3M"]
    yield_2y = yield_2y_ea_m.loc[t, "Y2Y"]
    yield_10y = yield_10y_ea_m.loc[t, "Y10Y"]

    beta_0_m.loc[t, "(y(3) + y(24) + y(120))/3"] = (yield_3m + yield_2y + yield_10y) / 3


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
    min(vix_data.index),
    min(ciss_idx.index),
    min(yields_ea_m_r.index),
    min(ea_rec.index),
)


end_ea = min(
    max(ind_pro_ea.index),
    max(infl_ea.index),
    max(ea_rate_3m.index),
    max(eustx_50_m_ret.index),
    max(vix_data.index),
    max(ciss_idx.index),
    max(yields_ea_m_r.index),
    max(ea_rec.index),
)


df_ea = [
    ind_pro_ea[start_ea:end_ea],
    infl_ea[start_ea:end_ea],
    ea_rate_3m[start_ea:end_ea],
    eustx_50_m_ret[start_ea:end_ea],
    vix_data.loc[start_ea:end_ea, "VSTOXX"],
    ciss_idx.loc[start_ea:end_ea],
    yields_ea_m_r[start_ea:end_ea],
    ea_rec.loc[start_ea:end_ea],
]

df_ea = pd.concat(df_ea, axis=1).dropna()

##### Plots
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis")

# VSTOXX
plt.figure(figsize=(15, 10))
plt.plot(df_ea["INDPRO_EA"], label="INDPRO_EA")
plt.plot(df_ea["VSTOXX"], label="VSTOXX")
plt.plot(df_ea["EUSTX_50_YoY"], label="EUSTX_50_YoY")

plt.legend()
plt.show()


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
    beta_0_m.loc[:, "(y(3) + y(24) + y(120))/3"],
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

# EA_Recession Subset
ea_rec = ea_rec.loc[start_ea:end_ea]
# Factors
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r.loc[start_ea:end_ea, "beta_0"], label="Level Factor", color="b")
plt.plot(
    yields_ea_m_r.loc[start_ea:end_ea, "beta_1"],
    label="Slope Factor",
    color="orange",
    linestyle="--",
)
plt.plot(
    yields_ea_m_r.loc[start_ea:end_ea, "beta_2"],
    label="Curvature Factor",
    color="green",
    linestyle=":",
)

# Adding recession bars
start_date = None  # Initialize start_date to None
for i in range(1, len(ea_rec)):
    if ea_rec.iloc[i] == 1 and ea_rec.iloc[i - 1] == 0:
        start_date = ea_rec.index[i]
    if ea_rec.iloc[i] == 0 and ea_rec.iloc[i - 1] == 1:
        end_date = ea_rec.index[i]
        plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
        start_date = None  # Reset start_date after plotting

# Handle the case where the series ends in a recession
if start_date is not None:
    plt.axvspan(start_date, yields_ea_m_r.index[-1], color="lightgray", alpha=0.8)

plt.legend()
# plt.savefig("Factors_Figure_EA_1.pdf", dpi=1000)
plt.show()

# Beta 0
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r.loc[start_ea:end_ea, "beta_0"], label="Level Factor", color="b")
plt.plot(
    beta_0_m.loc[start_ea:end_ea, "(y(3) + y(24) + y(120))/3"],
    label="(y(3) + y(24) + y(120))/3",
    linestyle="--",
    color="orange",
)
plt.plot(infl_ea.loc[start_ea:end_ea], label="Inflation EA", linestyle=":", color="g")

plt.legend()
plt.savefig("Beta_0_Figure_EA_1.pdf", dpi=1000)
plt.show()

# Beta 1
plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r.loc[start_ea:end_ea, "beta_1"], label="Slope Factor", color="b")

plt.plot(
    beta_1_m.loc[start_ea:end_ea, "y(3) - y(120)"],
    label="y(3) - y(120)",
    color="orange",
    linestyle="--",
)

plt.plot(df_ea.loc[start_ea:end_ea, "INDPRO_EA"], label="INDRPO_EA_YoY", linestyle=":")

plt.legend()
plt.savefig("Beta_1_Figure_EA_1.pdf", dpi=1000)
plt.show()


# Beta 2
plt.figure(figsize=(15, 10))
plt.plot(
    yields_ea_m_r.loc[start_ea:end_ea, "beta_2"], label="Curvature Factor", color="c"
)

plt.plot(
    beta_2_m.loc[start_ea:end_ea, "Curvature_Approx"],
    label="2 * y(24) - y(120) - y(3)",
    color="red",
    linestyle="--",
)

plt.legend()
plt.savefig("Beta_2_Figure_EA_1.pdf", dpi=1000)
plt.show()


########## Analysis ##########
# Correlations
# Factor Plots
# Level Factor
pearsonr(
    yields_ea_m_r.loc[start_ea:end_ea, "beta_0"],
    beta_0_m.loc[start_ea:end_ea, "(y(3) + y(24) + y(120))/3"],
)

pearsonr(df_ea.loc[start_ea:end_ea, "beta_0"], df_ea.loc[start_ea:end_ea, "Infl_EA"])


plt.figure(figsize=(15, 10))
plt.plot(df_ea.loc[start_ea:end_ea, "Infl_EA"], label="Inflation EA", color="c")

plt.plot(
    df_ea.loc[start_ea:end_ea, "beta_0"],
    label="Level",
    color="red",
    linestyle="--",
)

plt.legend()
plt.show()


# SLope Factor
pearsonr(
    yields_ea_m_r.loc[start_ea:end_ea, "beta_1"],
    beta_1_m.loc[start_ea:end_ea, "y(3) - y(120)"],
)

pearsonr(df_ea.beta_1, df_ea.beta_2)

pearsonr(yields_ea_m_r.loc[:"2022", "beta_1"], yields_ea_m_r.loc[:"2022", "beta_2"])


yields_ea_m_r.loc[:"2023-01-01", "beta_1"].tail(20)

pearsonr(df_ea["INDPRO_EA"], df_ea["beta_1"])


# Curvature Factor
pearsonr(
    yields_ea_m_r.loc[start_ea:end_ea, "beta_2"],
    beta_2_m.loc[start_ea:end_ea, "Curvature_Approx"],
)

plt.figure(figsize=(15, 10))
plt.plot(yields_ea_m_r.loc[start_ea:end_ea, "beta_2"], label="Curvature", color="c")

plt.plot(
    beta_2_m.loc[start_ea:end_ea, "Curvature_Approx"],
    label="Curvature_Approx",
    color="red",
    linestyle="--",
)

plt.legend()
plt.show()


##############################
########## Analysis ##########
##############################

##########sVAR ##########
df_analysis_ea = [
    df_ea["INDPRO_EA"],
    df_ea["Infl_EA"],
    df_ea["EA_Rate_3M"],
    df_ea["VSTOXX"],
    df_ea["beta_0"],
    df_ea["beta_1"],
    df_ea["beta_2"],
    df_ea["EUSTX_50_YoY"],
]

df_analysis_ea = pd.concat(df_analysis_ea, axis=1)

df_analysis_ea.rename(
    columns={
        "INDPRO_EA": "IP_EA",
        "beta_0": "L_EA",
        "beta_1": "S_EA",
        "beta_2": "C_EA",
        "EUSTX_50_YoY": "EUSTX_50",
    },
    inplace=True,
)

# Estimate sVAR
model_ea = VAR(df_analysis_ea)
print(model_ea.select_order())

result = model_ea.fit(maxlags=4, ic="bic")
print(result.is_stable())

# Stationarity Check (with Latex output)
adf_test_ea = get_adf(df_analysis_ea)

col_names_adf = ["t-Statistic", "Critical value", "p-value"]

df_adf_ea = pd.DataFrame.from_dict(adf_test_ea, orient="index", columns=col_names_adf)

df_adf_ea.index = [
    "$IP^{EA}_{t}$",
    "$\\pi^{EA}_{t}$",
    "$i^{EA}_{t}$",
    "$FS^{EA}_{t}$",
    "$L^{EA}_{t}$",
    "$S^{EA}_{t}$",
    "$C^{EA}_{t}$",
    "$M^{EA}_{t}$",
]


print(df_adf_ea.round(4).to_latex(escape=False))


# Estimation Results (with Latex output)
result.summary()
result.params.round(4)
print(result.params.round(4).to_latex())

result.bse.round(4)

result.pvalues.round(4)

# Output Table
estimates_ea = result.params.round(4)
estimates_ea.index = (
    estimates_ea.index[:1].tolist() + (estimates_ea.index[1:].str[3:] + "-1").tolist()
)
# estimates_ea.reset_index(inplace=True)
# estimates_ea = estimates_ea.iloc[:, 1:]
std_errors_ea = result.bse.round(4)
std_errors_ea.index = ("se_" + std_errors_ea.index[:1]).tolist() + (
    "se_" + std_errors_ea.index[1:].str[3:] + "-1"
).tolist()
# std_errors_ea.reset_index(inplace=True)


for i in range(estimates_ea.shape[0]):
    print(estimates_ea.iloc[i, :])


test = pd.concat([estimates_ea, std_errors_ea])


index_sort = []
for i in range(estimates_ea.shape[0]):
    index_sort.append(estimates_ea.index[i])
    index_sort.append(std_errors_ea.index[i])


test = test.reindex(index_sort)

print(test.to_latex(float_format="%.4f"))


# Information Criteria
llf_ea = {"Log-Likelihood": result.llf}
aic_ea = {"AIC": result.aic}
bic_ea = {"BIC": result.bic}
hqic_ea = {"HQIC": result.hqic}

dict_ic_ea = {**llf_ea, **aic_ea, **bic_ea, **hqic_ea}
print(pd.DataFrame.from_dict(dict_ic_ea, orient="index").round(4).to_latex())


# IRFs
irfs_ea = result.irf(36)
# plt.figure(figsize=(30, 15))
irfs_ea.plot(
    orth=True,
    signif=0.1,
    figsize=(30, 15),
    plot_params={
        "legend_fontsize": 20
        # "tick_params": {"axis": "y", "pad": 10}
    },
    subplot_params={
        "fontsize": 15,
        #  "wspace" : 0.8,
        # "hspace": 0.8,
        #  "left" : 0.01,
        #  "right" : 1,
        # "tick_params": {"axis": "y", "pad": 10},
    },
)
plt.savefig("IRF_EA_30_15_v3.pdf", dpi=1000)
plt.show()


# Block Granger Causality
# Macro to Yield Curve
granger_result = result.test_causality(
    ["L_EA", "S_EA", "C_EA"], ["IP_EA", "Infl_EA", "EA_Rate_3M"], kind="F"
)

# print(granger_result.summary())

result_macro_us = granger_result.summary()

df_result_macro_us = pd.DataFrame(result_macro_us[1:], columns=result_macro_us[0])

df_result_macro_us.iloc[0, 1]

print(df_result_macro_us.to_latex())


# Yield Curve to Macro
granger_result = result.test_causality(
    ["IP_EA", "Infl_EA", "EA_Rate_3M"], ["L_EA", "S_EA", "C_EA"], kind="F"
)

# print(granger_result.summary())

result_yc_us = granger_result.summary()

df_result_yc_us = pd.DataFrame(result_yc_us[1:], columns=result_yc_us[0])

print(df_result_yc_us.to_latex())
