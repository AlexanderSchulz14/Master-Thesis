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
from tabulate import tabulate
from stargazer.stargazer import Stargazer


# import plotly.graph_objects as go
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

from MA_functions import *

# import filterpy  # for Kalman Filter
# from nelson_siegel_svensson.calibrate import *
# import rpy2
import io


# Get Data
# Data
# GDP
gdp_us = fred.get_series("GDPC1")
gdp_us.rename("GDP_US", inplace=True)


# Industrial Production
ind_pro_us = fred.get_series("INDPRO")
ind_pro_us.rename("INDPRO", inplace=True)

ind_pro_us_diff = ind_pro_us.pct_change(periods=12).dropna() * 100
ind_pro_us_diff.name = "INDPRO_YoY"


# CPI
cpi_us = fred.get_series("CPIAUCSL")  # convert to YoY rate
cpi_us.rename("Infl_US", inplace=True)

infl_us = cpi_us.pct_change(periods=12).dropna() * 100


# Investment
inv_us = fred.get_series("GPDIC96")
inv_us.rename("Inv_US", inplace=True)


# Cpaital Utilization
cap_util_us = fred.get_series("TCU")
cap_util_us.rename("CU_US", inplace=True)

cap_util_us_diff = cap_util_us.pct_change(periods=12).dropna() * 100
cap_util_us_diff.name = "CU_US_YoY"


# FFR
ffr = fred.get_series("DFF")
ffr.rename("FFR", inplace=True)
ffr_m = ffr.resample("M", loffset="1d").mean()  # convert daily to monthly!!!


# T-Bill 3M
tb_3m = fred.get_series("TB3MS")
tb_3m.rename("TB_3M", inplace=True)


# T-Note 1Y
tb_1y = fred.get_series("DGS1")
tb_1y.rename("DGS1", inplace=True)
tb_1y_m = tb_1y.resample("M", loffset="1d").mean()


# T-Bond 10Y
tb_10y = fred.get_series("DGS10")
tb_10y.rename("DGS10", inplace=True)
tb_10y_m = tb_10y.resample("M", loffset="1d").mean()


# Term Spread
ts_10y3m_us = fred.get_series("T10Y3M")
ts_10y3m_us.rename("TS10Y3M_US", inplace=True)

ts_10y2y_us = fred.get_series("T10Y2Y")
ts_10y2y_us.rename("TS10Y2Y_US", inplace=True)

# y_12m = fred.get_series('TY12MCD')


# S&P 500
# sp_500_tckr = yf.Ticker('^GSPC')
# sp_500 = sp_500_tckr.history(period='max')
sp_500_1 = yf.download("^GSPC", start="1970-01-01", end="2023-01-01")
sp_500_1_m = sp_500_1["Close"].resample("M", loffset="1d").mean()
sp_500_1_m.name = "S&P_500"

sp_500_1_m_ret = sp_500_1_m.pct_change(periods=12).dropna() * 100
sp_500_1_m_ret.name = "S&P_500_YoY"

# Financial Stress Variables
# Excess Bond Premium (EBP)
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data")
ebp = pd.read_csv("ebp_csv.csv", index_col=[0], parse_dates=True)


# VIX? (von Bloomberg?)
vix = yf.download("^VIX", start="1990-01-02", end="2024-01-01")
vix_m = vix.resample("M", loffset="1d").mean()


# Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
corp_spread = fred.get_series("BAA10Y")
corp_spread.rename("BAA10Y", inplace=True)
corp_spread_m = corp_spread.resample("M", loffset="1d").mean()


# Yield Data
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data")
yields_us = pd.read_excel("LW_monthly.xlsx", skiprows=8)

yields_us.rename(columns={yields_us.columns[0]: "Date"}, inplace=True)

yields_us["Date"] = pd.to_datetime(yields_us["Date"], format="%Y%m")

yields_us.set_index([yields_us.columns[0]], inplace=True)

yields_us.columns = [col.replace(" ", "") for col in yields_us.columns]

# Subset Yield Data as not all maturities are available for each point in time -> used in R for Nelson-Siegel decomposition
start_yields_us = "1972-01-01"
yields_us_sub = yields_us.loc[start_yields_us:]
yields_us_sub.to_csv("Yields_Data_Subset.csv")


# Beta Coefficients from R
yields_us_sub_r = pd.read_csv("Yields_US_R.csv", index_col=[0], parse_dates=True)

yields_us_sub_r.rename(
    columns={
        "beta_0": "Level Factor",
        "beta_1": "Slope Factor",
        "beta_2": "Curvature Factor",
    },
    inplace=True,
)


# Get Approximations for Coefficients (see Diebold et al. (2006))
# Beta 0 - Level Factor
yields_us_sub_r["(y(3) + y(24) + y(120))/3"] = np.nan
for t in yields_us_sub_r.index:
    result_beta_0 = get_beta_0_approx(yields_us_sub_r, t)
    yields_us_sub_r.loc[t, "(y(3) + y(24) + y(120))/3"] = result_beta_0


# Beta 1 - Slope Factor
yields_us_sub_r["y(3) - y(120)"] = yields_us_sub_r["3m"] - yields_us_sub_r["120m"]


# Beta 2 - Curvature Factor
yields_us_sub_r["2 * y(24) - y(120) - y(3)"] = (
    2 * yields_us_sub_r["24m"] - yields_us_sub_r["120m"] - yields_us_sub_r["3m"]
)


# Columns to use for merge
yield_cols_to_use = [
    "3m",
    "6m",
    "9m",
    "12m",
    "15m",
    "18m",
    "21m",
    "24m",
    "30m",
    "36m",
    "48m",
    "60m",
    "72m",
    "84m",
    "96m",
    "108m",
    "120m",
    "Level Factor",
    "Slope Factor",
    "Curvature Factor",
    "(y(3) + y(24) + y(120))/3",
    "y(3) - y(120)",
    "2 * y(24) - y(120) - y(3)",
]


# Get Period
start_us = "1973-01-01"
# start_us = max(
#     # min(gdp_us.index),
#     min(ind_pro_us.index),
#     min(ind_pro_us_diff.index),
#     min(infl_us.index),
#     min(ffr_m.index),
#     min(cap_util_us_diff.index),
#     min(yields_us_sub_r.index),
#     min(tb_3m.index),
#     min(sp_500_1_m.index),
#     min(sp_500_1_m_ret.index),
#     min(ebp["ebp"].index),
#     #    min(vix_m.index),
#     #    min(ts_10y2y_us.index)
# )

# end_us = "2000-12-01"
end_us = min(
    # max(gdp_us.index),
    max(ind_pro_us.index),
    max(ind_pro_us_diff.index),
    max(infl_us.index),
    max(ffr_m.index),
    max(cap_util_us_diff.index),
    max(yields_us_sub_r.index),
    max(tb_3m.index),
    max(sp_500_1_m.index),
    max(sp_500_1_m_ret.index),
    max(ebp["ebp"].index),
    #    max(vix_m.index),
    #    max(ts_10y2y_us.index)
)


# Merge Data
df_us = [
    # gdp_us[start_us:end_us],
    ind_pro_us[start_us:end_us],
    ind_pro_us_diff[start_us:end_us],
    infl_us[start_us:end_us],
    ffr_m[start_us:end_us],
    cap_util_us_diff[start_us:end_us],
    yields_us_sub_r.loc[start_us:end_us, yield_cols_to_use],
    tb_3m[start_us:end_us],
    sp_500_1_m.loc[start_us:end_us],
    sp_500_1_m_ret.loc[start_us:end_us],
    ebp.loc[start_us:end_us, "ebp"],
    #  ts_10y2y_us[start_us:end_us]
]

df_us = pd.concat(df_us, axis=1).dropna()


##############################
########## Analysis ##########
##############################
# Plots
# All Coefficients
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis")

plt.figure(figsize=(15, 10))
sns.lineplot(df_us[["Level Factor", "Slope Factor", "Curvature Factor"]])
plt.legend(loc="lower right")

plt.savefig("Factor_Figure.pdf", dpi=1000)
plt.show()

# Beta 0 Level Factor & Inflation
plt.figure(figsize=(15, 10))
sns.lineplot(
    df_us[["Level Factor", "(y(3) + y(24) + y(120))/3", "Infl_US"]],
    legend="auto",
    palette=["blue", "orange", "green"],
)
plt.legend(loc="lower right")

plt.savefig("Beta_0_Figure.pdf", dpi=1000)
plt.show()

# Correlation
df_us["Level Factor"].corr(df_us["(y(3) + y(24) + y(120))/3"])

df_us["Level Factor"].corr(df_us["Infl_US"])


# Beta 1 Slope Factor
plt.figure(figsize=(15, 10))
sns.lineplot(df_us[["Slope Factor", "y(3) - y(120)", "INDPRO_YoY"]])
plt.legend(loc="lower right")

plt.savefig("Beta_1_Figure.pdf", dpi=1000)
plt.show()

# Correlation
df_us["Slope Factor"].corr(df_us["y(3) - y(120)"])
df_us["Slope Factor"].corr(df_us["INDPRO_YoY"])


# Beta 2 - Curvature Factor
plt.figure(figsize=(15, 10))
sns.lineplot(
    df_us[["Curvature Factor", "2 * y(24) - y(120) - y(3)"]], palette=["c", "r"]
)
plt.legend(loc="lower right")

plt.savefig("Beta_2_Figure.pdf", dpi=1000)
plt.show()

# Correlation
df_us["Curvature Factor"].corr(df_us["2 * y(24) - y(120) - y(3)"])


# Tables
yields_us_summary = df_us[yield_cols_to_use[0:17]].describe()
yields_us_summary = yields_us_summary.loc[
    ["mean", "std", "min", "max"]
].T  # .T -> use transpose of summary statistics df

factors_us_summaries = df_us[yield_cols_to_use[17:20]].describe()
factors_us_summaries = factors_us_summaries.loc[
    ["mean", "std", "min", "max"]
].T  # .T -> use transpose of summary statistics df


# HP Filter
# INDPRO
cycle, trend = hpfilter(df_us["INDPRO"])
ip_hp = df_us["INDPRO"] - trend

plt.figure(figsize=(15, 10))
plt.plot(df_us["INDPRO_YoY"], label="INDPRO")
plt.plot(ip_hp, label="INDPRO_HP")
plt.legend()
plt.show()

pearsonr(df_us["Slope Factor"], ip_hp)

# Inflation
cycle, trend = hpfilter(cpi_us)
cpi_hp = cpi_us - trend

plt.figure(figsize=(15, 10))
plt.plot(df_us["Infl_US"], label="Inflation")
plt.plot(cpi_hp[start_us:end_us], label="CPI_HP")
plt.legend()
plt.show()

##########sVAR ##########
# Differenced Data
df_analysis_us = [
    df_us["Level Factor"],
    df_us["Slope Factor"],
    df_us["Curvature Factor"],
    df_us["INDPRO_YoY"],
    df_us["Infl_US"],
    df_us["FFR"],
    df_us["ebp"],
    df_us["S&P_500_YoY"],
]

df_analysis_us = [
    df_us["INDPRO_YoY"],
    df_us["Infl_US"],
    df_us["FFR"],
    df_us["ebp"],
    df_us["Level Factor"],
    df_us["Slope Factor"],
    df_us["Curvature Factor"],
    df_us["S&P_500_YoY"],
]


df_analysis_us = pd.concat(df_analysis_us, axis=1)

df_analysis_us.rename(
    columns={
        "INDPRO_YoY": "IP",
        "Level Factor": "L",
        "Slope Factor": "S",
        "Curvature Factor": "C",
        "S&P_500_YoY": "S&P_500",
    },
    inplace=True,
)

# Plot Data
plot_data(df_analysis_us)


# Stationarity Check
get_adf(df_analysis_us)


# Estimate sVAR
model_us = VAR(df_analysis_us)
print(model_us.select_order())

result = model_us.fit(maxlags=1, ic="aic")
result.summary()

# stargazer = Stargazer(result)

# print(result.test_whiteness())
print(result.is_stable())

residuals = result.sigma_u
resid_chol_decomp = np.linalg.cholesky(residuals)
np.linalg.eigvals(resid_chol_decomp)


# IRFs
irfs_us = result.irf(36)
# plt.figure(figsize=(30, 15))
irfs_us.plot(
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
plt.savefig("IRF_US_30_15_v2.pdf", dpi=1000)
plt.show()

irfs_us = result.irf(36)
plt.figure(figsize=(10, 5))
irfs_us.plot(orth=True, impulse="L", signif=0.1)
plt.savefig("IRF_L.pdf")
plt.show()

plt.figure(figsize=(10, 5))
irfs_us.plot(orth=True, impulse="S", signif=0.1)
plt.savefig("IRF_S.pdf")
plt.show()

plt.figure(figsize=(10, 5))
irfs_us.plot(orth=True, impulse="C", signif=0.1)
plt.savefig("IRF_C.pdf")
plt.show()

plt.figure(figsize=(10, 5))
irfs_us.plot(orth=True, impulse="IP", signif=0.1)
plt.savefig("IRF_IP.pdf")
plt.show()

plt.figure(figsize=(10, 5))
irfs_us.plot(orth=True, impulse="Infl_US", signif=0.1)
plt.savefig("IRF_Infl_US.pdf")
plt.show()

plt.figure(figsize=(10, 5))
irfs_us.plot(orth=True, impulse="FFR", signif=0.1)
plt.savefig("IRF_FFR.pdf")
plt.show()


########## Granger Causality ##########
grangercausalitytests(df_us[["Infl_US", "Level Factor"]], 4)

grangercausalitytests(df_us[["Level Factor", "Infl_US"]], 4)

grangercausalitytests(df_us[["Slope Factor", "INDPRO_YoY"]], 4)

grangercausalitytests(df_us[["INDPRO_YoY", "Slope Factor"]], 4)

grangercausalitytests(df_us[["TB_3M", "Level Factor"]], 4)

grangercausalitytests(df_us[["Level Factor", "TB_3M"]], 4)

grangercausalitytests(df_us[["Slope Factor", "TB_3M"]], 4)

grangercausalitytests(df_us[["Curvature Factor", "TB_3M"]], 4)


# VAR
df_analysis_us = [
    df_us["INDPRO_YoY"],
    df_us["Infl_US"],
    df_us["TB_3M"],
    df_us["ebp"],
    df_us["Level Factor"],
    df_us["Slope Factor"],
    df_us["Curvature Factor"],
    df_us["S&P_500_YoY"],
]


df_analysis_us = pd.concat(df_analysis_us, axis=1)

df_analysis_us.rename(
    columns={
        "INDPRO_YoY": "IP",
        "Level Factor": "L",
        "Slope Factor": "S",
        "Curvature Factor": "C",
        "S&P_500_YoY": "S&P_500",
    },
    inplace=True,
)

path_ma_data = r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data"
df_analysis_us.to_csv(path_ma_data + "\\" + "VAR_Data_US.csv")

# Granger Causality
# Macro to Yield Curve
granger_result = result.test_causality(
    ["L", "S", "C"],
    ["IP", "Infl_US", "FFR"],
)

print(granger_result.summary())

result_macro_us = granger_result.summary()

df_result_macro_us = pd.DataFrame(result_macro_us[1:], columns=result_macro_us[0])

df_result_macro_us.to_latex()


# Yield Curve to Macro
granger_result = result.test_causality(
    ["IP", "Infl_US", "FFR"],
    ["L", "S", "C"],
)

print(granger_result.summary())

result_yc_us = granger_result.summary()

df_result_yc_us = pd.DataFrame(result_yc_us[1:], columns=result_yc_us[0])

df_result_yc_us.to_latex()


# Create an empty 8x8 matrix filled with False
lower_triangular = np.zeros((8, 8), dtype=bool)

# Set TRUE values below or on the main diagonal
lower_triangular[np.tril_indices_from(lower_triangular)] = True

# Print the matrix
print(lower_triangular)


########## Playing Around ##########
print(result.summary())


plt.figure(figsize=(30, 15))
acorr_plot = result.plot_acorr()
plt.show()


print(result.test_whiteness(nlags=5))


result.params
print(result.params.to_latex())
