import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import random
from datetime import date, datetime
from pandas.tseries.frequencies import to_offset
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sns.set_theme(style="darkgrid")
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
from stargazer.stargazer import Stargazer


# import plotly.graph_objects as go
import copy
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, SVAR
import statsmodels.tsa.vector_ar.svar_model as svar
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import pearsonr
import requests as req

from fredapi import Fred

fred = Fred(api_key="ef7244731efdde9698fef5d547b7094f")

from MA_functions import *

# import filterpy  # for Kalman Filter
# from nelson_siegel_svensson.calibrate import *
# import rpy2
import io

data_path_ma = r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data"
figures_path_ma = r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis"

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
cpi_us.rename("CPI_US", inplace=True)

infl_us = cpi_us.pct_change(periods=12).dropna() * 100
infl_us.rename("Infl_US", inplace=True)


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

# NBER Recession Dates
us_rec = fred.get_series("USREC")

us_rec.loc["1970":].plot.area(color="lightgray", alpha=0.5)


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
ebp = pd.read_csv(data_path_ma + "\\" + "ebp_csv.csv", index_col=[0], parse_dates=True)

# CISS Index
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


# VIX? (von Bloomberg?)
# vix = yf.download("^VIX", start="1990-01-02", end="2024-01-01")
# vix_m = vix.resample("M", loffset="1d").mean()

# Data ohne Bloomber BDH Formel
vix_data = pd.read_csv(data_path_ma + "\\" + "VIX_VSTOXX_Data.csv", index_col=0)

vix_data.index = pd.to_datetime(vix_data.index)

vix_data.index = vix_data.index + pd.offsets.MonthBegin(1)


# Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
# corp_spread = fred.get_series("BAA10Y")
# corp_spread.rename("BAA10Y", inplace=True)
# corp_spread_m = corp_spread.resample("M", loffset="1d").mean()

###############################################################################
# Yields Data
###############################################################################
yields_us = pd.read_excel(data_path_ma + "\\" + "LW_monthly.xlsx", skiprows=8)

yields_us.rename(columns={yields_us.columns[0]: "Date"}, inplace=True)

yields_us["Date"] = pd.to_datetime(yields_us["Date"], format="%Y%m")

yields_us.set_index([yields_us.columns[0]], inplace=True)

yields_us.columns = [col.replace(" ", "") for col in yields_us.columns]

yields_us = yields_us

# Subset Yield Data as not all maturities are available for each point in time -> used in R for Nelson-Siegel decomposition
start_yields_us = "1975-01-01"
yields_us_sub = yields_us.loc[start_yields_us:]
yields_us_sub.to_csv(data_path_ma + "\\" + "Yields_Data_Subset.csv")


# Beta Coefficients from R
yields_us_sub_r = pd.read_csv(data_path_ma + "\\" + "Yields_US_R.csv", index_col=[0], parse_dates=True)

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
start_us = "1978-01-01"
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
#     # min(ciss_idx.index),
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
    # max(ciss_idx.index),
    #    max(vix_m.index),
    #    max(ts_10y2y_us.index)
)


# Merge Data
df_us = [
    # gdp_us[start_us:end_us],
    # ind_pro_us[start_us:end_us],
    ind_pro_us_diff[start_us:end_us],
    cpi_us[start_us:end_us],
    infl_us[start_us:end_us],
    ffr_m[start_us:end_us],
    cap_util_us[start_us:end_us],
    cap_util_us_diff[start_us:end_us],
    yields_us_sub_r.loc[start_us:end_us, yield_cols_to_use],
    tb_3m[start_us:end_us],
    sp_500_1_m.loc[start_us:end_us],
    sp_500_1_m_ret.loc[start_us:end_us],
    ebp.loc[start_us:end_us, "ebp"],
    # ciss_idx.loc[start_us:end_us],
    #  ts_10y2y_us[start_us:end_us]
]

df_us = pd.concat(df_us, axis=1).dropna()


##############################
########## Analysis ##########
##############################
# Plots
# Add shadded Recession Areas in Figures for Sample Period
us_rec = us_rec.loc[start_us:end_us]

# # mal 100 damit alles in Prozentpunkte ist
# df_us[
#     [
#         "Level Factor",
#         "Slope Factor",
#         "Curvature Factor",
#         "(y(3) + y(24) + y(120))/3",
#         "y(3) - y(120)",
#         "2 * y(24) - y(120) - y(3)",
#     ]
# ] = (
#     df_us[
#         [
#             "Level Factor",
#             "Slope Factor",
#             "Curvature Factor",
#             "(y(3) + y(24) + y(120))/3",
#             "y(3) - y(120)",
#             "2 * y(24) - y(120) - y(3)",
#         ]
#     ]
# * 100
# )

# All Coefficients
os.chdir(r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis")

plt.figure(figsize=(15, 10))
sns.lineplot(df_us[["Level Factor", "Slope Factor", "Curvature Factor"]])
plt.legend(loc="lower right")

# Adding recession bars
start_date = None  # Initialize start_date to None
for i in range(1, len(us_rec)):
    if us_rec.iloc[i] == 1 and us_rec.iloc[i - 1] == 0:
        start_date = us_rec.index[i]
    if us_rec.iloc[i] == 0 and us_rec.iloc[i - 1] == 1:
        end_date = us_rec.index[i]
        plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
        start_date = None  # Reset start_date after plotting

# Handle the case where the series ends in a recession
if start_date is not None:
    plt.axvspan(start_date, ind_pro_us_diff.index[-1], color="lightgray", alpha=0.8)

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

# Adding recession bars
start_date = None  # Initialize start_date to None
for i in range(1, len(us_rec)):
    if us_rec.iloc[i] == 1 and us_rec.iloc[i - 1] == 0:
        start_date = us_rec.index[i]
    if us_rec.iloc[i] == 0 and us_rec.iloc[i - 1] == 1:
        end_date = us_rec.index[i]
        plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
        start_date = None  # Reset start_date after plotting

# Handle the case where the series ends in a recession
if start_date is not None:
    plt.axvspan(start_date, ind_pro_us_diff.index[-1], color="lightgray", alpha=0.8)

plt.savefig("Beta_0_Figure.pdf", dpi=1000)
plt.show()

# Correlation
df_us["Level Factor"].corr(df_us["(y(3) + y(24) + y(120))/3"])

df_us["Level Factor"].corr(df_us["Infl_US"])


# Beta 1 Slope Factor
plt.figure(figsize=(15, 10))
sns.lineplot(df_us[["Slope Factor", "y(3) - y(120)", "INDPRO_YoY"]])
plt.legend(loc="lower right")

# Adding recession bars
start_date = None  # Initialize start_date to None
for i in range(1, len(us_rec)):
    if us_rec.iloc[i] == 1 and us_rec.iloc[i - 1] == 0:
        start_date = us_rec.index[i]
    if us_rec.iloc[i] == 0 and us_rec.iloc[i - 1] == 1:
        end_date = us_rec.index[i]
        plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
        start_date = None  # Reset start_date after plotting

# Handle the case where the series ends in a recession
if start_date is not None:
    plt.axvspan(start_date, ind_pro_us_diff.index[-1], color="lightgray", alpha=0.8)

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

# Adding recession bars
start_date = None  # Initialize start_date to None
for i in range(1, len(us_rec)):
    if us_rec.iloc[i] == 1 and us_rec.iloc[i - 1] == 0:
        start_date = us_rec.index[i]
    if us_rec.iloc[i] == 0 and us_rec.iloc[i - 1] == 1:
        end_date = us_rec.index[i]
        plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
        start_date = None  # Reset start_date after plotting

# Handle the case where the series ends in a recession
if start_date is not None:
    plt.axvspan(start_date, ind_pro_us_diff.index[-1], color="lightgray", alpha=0.8)

plt.savefig("Beta_2_Figure.pdf", dpi=1000)
plt.show()


# Interest Rate and Yield
plt.figure(figsize=(15, 10))
plt.plot(df_us["FFR"], label="FFR")
plt.plot(df_us["3m"], label="3m")
plt.plot(df_us["120m"], label="120m")
plt.plot(df_us["Level Factor"], label="Level Factor")

plt.legend()
plt.show()


# IP, Inflation, FFR
plt.figure(figsize=(15, 10))
plt.plot(df_us["FFR"], label="FFR")
plt.plot(df_us["INDPRO_YoY"], label="INDPRO_YoY")
plt.plot(df_us["Infl_US"], label="Infl_US")


plt.legend()
plt.show()


# IP, Slope
plt.figure(figsize=(15, 10))
plt.plot(df_us["INDPRO_YoY"], label="INDPRO_YoY")
plt.plot(df_us["Slope Factor"], label="Slope Factor")


plt.legend()
plt.show()

pearsonr(df_us["INDPRO_YoY"], df_us["Slope Factor"])

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
# cycle, trend = hpfilter(df_us["INDPRO"])
# ip_hp = df_us["INDPRO"] - trend

# plt.figure(figsize=(15, 10))
# plt.plot(df_us["INDPRO_YoY"], label="INDPRO")
# plt.plot(ip_hp, label="INDPRO_HP")
# plt.legend()
# plt.show()

# pearsonr(df_us["Slope Factor"], ip_hp)

# # Inflation
# cycle, trend = hpfilter(cpi_us)
# cpi_hp = cpi_us - trend

# plt.figure(figsize=(15, 10))
# plt.plot(df_us["Infl_US"], label="Inflation")
# plt.plot(cpi_hp[start_us:end_us], label="CPI_HP")
# plt.legend()
# plt.show()


##########sVAR ##########
# Differenced Data
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
        "INDPRO_YoY": "IP_US",
        "ebp": "FS_US",
        "Level Factor": "L_US",
        "Slope Factor": "S_US",
        "Curvature Factor": "C_US",
        "S&P_500_YoY": "S&P_500",
    },
    inplace=True,
)


# start_date = ""

# end_date = "2019-12-01"

# df_analysis_us = df_analysis_us[:end_date]

# df_analysis_us.to_csv("Analysis_US.csv")

# Estimate sVAR
model_us = VAR(df_analysis_us)
print(model_us.select_order())

result = model_us.fit(maxlags=6, ic="aic")

# print(result.test_whiteness())
# print(result.is_stable())

residuals = result.sigma_u
resid_chol_decomp = np.linalg.cholesky(residuals).astype(np.float64)
# np.linalg.eigvals(resid_chol_decomp)

# # Define the size of the matrix
# n = 8

# # Create a lower triangular matrix with "E" on the main diagonal and below, and 0 elsewhere
# lower_triangular_matrix = np.zeros((n, n), dtype=object)
# for i in range(n):
#     for j in range(i+1):
#         lower_triangular_matrix[i, j] = "E"

# print(lower_triangular_matrix)

# # Fit SVAR model with Cholesky decomposition
# svar_model = SVAR(df_analysis_us, svar_type='A', A=lower_triangular_matrix)
# svar_results = svar_model.fit()
# print(svar_results.summary())

# svar_results_irf = svar_results.irf(20)
# svar_results_irf.plot(figsize=(30,15), signif=0.1)


# Stationarity Check (with Latex output)
adf_test_us = get_adf(df_analysis_us)

col_names_adf = ["t-Statistic", "Critical value", "p-value"]

df_adf_us = pd.DataFrame.from_dict(adf_test_us, orient="index", columns=col_names_adf)

df_adf_us.index = [
    "$IP^{US}_{t}$",
    "$\\pi^{US}_{t}$",
    "$i^{US}_{t}$",
    "$FS^{US}_{t}$",
    "$L^{US}_{t}$",
    "$S^{US}_{t}$",
    "$C^{US}_{t}$",
    "$M^{US}_{t}$",
]


print(df_adf_us.round(4).to_latex(escape=False))


# Estimation Results (with Latex output)
result.summary()
result.params.round(2)
print(result.params.round(2).to_latex())

result.bse.round(4)

result.pvalues.round(4)


# # Output Table
# estimates_us = result.params.round(4)
# estimates_us.index = (
#     estimates_us.index[:1].tolist() + (estimates_us.index[1:].str[3:] + "-1").tolist()
# )
# # estimates_us.reset_index(inplace=True)
# # estimates_us = estimates_us.iloc[:, 1:]
# std_errors_us = result.bse.round(4)
# std_errors_us.index = ("se_" + std_errors_us.index[:1]).tolist() + (
#     "se_" + std_errors_us.index[1:].str[3:] + "-1"
# ).tolist()
# # std_errors_us.reset_index(inplace=True)


# for i in range(estimates_us.shape[0]):
#     print(estimates_us.iloc[i, :])


# test = pd.concat([estimates_us, std_errors_us])


# index_sort = []
# for i in range(estimates_us.shape[0]):
#     index_sort.append(estimates_us.index[i])
#     index_sort.append(std_errors_us.index[i])


# test = test.reindex(index_sort)

# print(test.to_latex(float_format="%.4f"))


# # Information Criteria
# llf_us = {"Log-Likelihood": result.llf}
# aic_us = {"AIC": result.aic}
# bic_us = {"BIC": result.bic}
# hqic_us = {"HQIC": result.hqic}

# dict_ic_us = {**llf_us, **aic_us, **bic_us, **hqic_us}
# print(pd.DataFrame.from_dict(dict_ic_us, orient="index").round(4).to_latex())


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

# irfs_us = result.irf(36)
# plt.figure(figsize=(10, 5))
# irfs_us.plot(orth=True, impulse="L", signif=0.1)
# plt.savefig("IRF_L.pdf")
# plt.show()

# plt.figure(figsize=(10, 5))
# irfs_us.plot(orth=True, impulse="S", signif=0.1)
# plt.savefig("IRF_S.pdf")
# plt.show()

# plt.figure(figsize=(10, 5))
# irfs_us.plot(orth=True, impulse="C", signif=0.1)
# plt.savefig("IRF_C.pdf")
# plt.show()

# plt.figure(figsize=(10, 5))
# irfs_us.plot(orth=True, impulse="IP", signif=0.1)
# plt.savefig("IRF_IP.pdf")
# plt.show()

# plt.figure(figsize=(10, 5))
# irfs_us.plot(orth=True, impulse="Infl_US", signif=0.1)
# plt.savefig("IRF_Infl_US.pdf")
# plt.show()

# plt.figure(figsize=(10, 5))
# irfs_us.plot(orth=True, impulse="FFR", signif=0.1)
# plt.savefig("IRF_FFR.pdf")
# plt.show()


########## Granger Causality ##########
# grangercausalitytests(df_us[["Infl_US", "Level Factor"]], 4)

# grangercausalitytests(df_us[["Level Factor", "Infl_US"]], 4)

# grangercausalitytests(df_us[["Slope Factor", "INDPRO_YoY"]], 4)

# grangercausalitytests(df_us[["INDPRO_YoY", "Slope Factor"]], 4)

# grangercausalitytests(df_us[["TB_3M", "Level Factor"]], 4)

# grangercausalitytests(df_us[["Level Factor", "TB_3M"]], 4)

# grangercausalitytests(df_us[["Slope Factor", "TB_3M"]], 4)

# grangercausalitytests(df_us[["Curvature Factor", "TB_3M"]], 4)


# Block Granger Causality
# Macro to Yield Curve
granger_result = result.test_causality(
    ["L_US", "S_US", "C_US"], ["IP_US", "Infl_US", "FFR"], kind="wald"
)

# print(granger_result.summary())

result_macro_us = granger_result.summary()

df_result_macro_us = pd.DataFrame(result_macro_us[1:], columns=result_macro_us[0])

df_result_macro_us.iloc[0, 1]

print(df_result_macro_us.to_latex())


# Yield Curve to Macro
granger_result = result.test_causality(
    ["IP_US", "Infl_US", "FFR"], ["L_US", "S_US", "C_US"], kind="wald"
)

# print(granger_result.summary())

result_yc_us = granger_result.summary()

df_result_yc_us = pd.DataFrame(result_yc_us[1:], columns=result_yc_us[0])

print(df_result_yc_us.to_latex())


# # Level Data
# df_analysis_us = [
#     df_us["INDPRO"],
#     df_us["CPI_US"],
#     df_us["FFR"],
#     df_us["ebp"],
#     df_us["Level Factor"],
#     df_us["Slope Factor"],
#     df_us["Curvature Factor"],
#     df_us["S&P_500_YoY"],
# ]


# df_analysis_us = pd.concat(df_analysis_us, axis=1)

# df_analysis_us.rename(
#     columns={
#         "INDPRO": "IP",
#         "Level Factor": "L",
#         "Slope Factor": "S",
#         "Curvature Factor": "C",
#         "S&P_500_YoY": "S&P_500",
#     },
#     inplace=True,
# )

# # Stationarity Check
# adf_test_us = get_adf(df_analysis_us)


# # Estimate sVAR
# model_us = VAR(df_analysis_us)
# print(model_us.select_order())

# result = model_us.fit(maxlags=4, ic="aic")
# result.summary()


# llf_us = {"Log-Likelihood": result.llf}
# aic_us = {"AIC": result.aic}
# bic_us = {"BIC": result.bic}
# hqic_us = {"HQIC": result.hqic}

# dict_ic_us = {**llf_us, **aic_us, **bic_us, **hqic_us}
# print(pd.DataFrame.from_dict(dict_ic_us, orient="index").round(4).to_latex())


# result.params
# print(result.params.to_latex())


# # print(result.test_whiteness())
# print(result.is_stable())


# # IRFs
# irfs_us = result.irf(36)
# # plt.figure(figsize=(30, 15))
# irfs_us.plot(
#     orth=True,
#     signif=0.1,
#     figsize=(30, 15),
#     plot_params={
#         "legend_fontsize": 20
#         # "tick_params": {"axis": "y", "pad": 10}
#     },
#     subplot_params={
#         "fontsize": 15,
#         #  "wspace" : 0.8,
#         # "hspace": 0.8,
#         #  "left" : 0.01,
#         #  "right" : 1,
#         # "tick_params": {"axis": "y", "pad": 10},
#     },
# )
# plt.savefig("IRF_US_30_15_Level_Data.pdf", dpi=1000)
# plt.show()


# # Capacity Utilization
# df_analysis_us = [
#     df_us["CU_US"],
#     df_us["Infl_US"],
#     df_us["FFR"],
#     df_us["ebp"],
#     df_us["Level Factor"],
#     df_us["Slope Factor"],
#     df_us["Curvature Factor"],
#     df_us["S&P_500_YoY"],
# ]


# df_analysis_us = pd.concat(df_analysis_us, axis=1)

# df_analysis_us.rename(
#     columns={
#         "Level Factor": "L",
#         "Slope Factor": "S",
#         "Curvature Factor": "C",
#         "S&P_500_YoY": "S&P_500",
#     },
#     inplace=True,
# )


# # Stationarity Check
# adf_test_us = get_adf(df_analysis_us)


# # Estimate sVAR
# model_us = VAR(df_analysis_us)
# print(model_us.select_order())

# result = model_us.fit(maxlags=4, ic="aic")
# result.summary()


# llf_us = {"Log-Likelihood": result.llf}
# aic_us = {"AIC": result.aic}
# bic_us = {"BIC": result.bic}
# hqic_us = {"HQIC": result.hqic}

# dict_ic_us = {**llf_us, **aic_us, **bic_us, **hqic_us}
# print(pd.DataFrame.from_dict(dict_ic_us, orient="index").round(4).to_latex())


# result.params
# print(result.params.to_latex())


# # print(result.test_whiteness())
# print(result.is_stable())


# # IRFs
# irfs_us = result.irf(36)
# # plt.figure(figsize=(30, 15))
# irfs_us.plot(
#     orth=True,
#     signif=0.1,
#     figsize=(30, 15),
#     plot_params={
#         "legend_fontsize": 20
#         # "tick_params": {"axis": "y", "pad": 10}
#     },
#     subplot_params={
#         "fontsize": 15,
#         #  "wspace" : 0.8,
#         # "hspace": 0.8,
#         #  "left" : 0.01,
#         #  "right" : 1,
#         # "tick_params": {"axis": "y", "pad": 10},
#     },
# )
# plt.savefig("IRF_US_30_15_CU.pdf", dpi=1000)
# plt.show()


# # Diebold et al (2006) sample
# df_analysis_us = [
#     df_us["Level Factor"],
#     df_us["Slope Factor"],
#     df_us["Curvature Factor"],
#     df_us["CU_US"],
#     df_us["Infl_US"],
#     df_us["FFR"],
# ]


# df_analysis_us = pd.concat(df_analysis_us, axis=1)

# # start_date = ""

# end_date = "2000-12-01"

# df_analysis_us = df_analysis_us[:end_date]

# df_analysis_us.rename(
#     columns={
#         "Level Factor": "L",
#         "Slope Factor": "S",
#         "Curvature Factor": "C",
#     },
#     inplace=True,
# )


# # Plot Data
# plot_data(df_analysis_us)


# # Stationarity Check
# adf_test_us = get_adf(df_analysis_us)


# # Estimate sVAR
# model_us = VAR(df_analysis_us)
# print(model_us.select_order())

# result = model_us.fit(maxlags=4, ic="aic")
# result.summary()


# llf_us = {"Log-Likelihood": result.llf}
# aic_us = {"AIC": result.aic}
# bic_us = {"BIC": result.bic}
# hqic_us = {"HQIC": result.hqic}

# dict_ic_us = {**llf_us, **aic_us, **bic_us, **hqic_us}
# print(pd.DataFrame.from_dict(dict_ic_us, orient="index").round(4).to_latex())


# result.params
# print(result.params.to_latex())


# # print(result.test_whiteness())
# print(result.is_stable())

# # IRFs
# irfs_us = result.irf(36)
# # plt.figure(figsize=(30, 15))
# irfs_us.plot(
#     orth=True,
#     signif=0.1,
#     figsize=(30, 15),
#     plot_params={
#         "legend_fontsize": 20
#         # "tick_params": {"axis": "y", "pad": 10}
#     },
#     subplot_params={
#         "fontsize": 15,
#         #  "wspace" : 0.8,
#         # "hspace": 0.8,
#         #  "left" : 0.01,
#         #  "right" : 1,
#         # "tick_params": {"axis": "y", "pad": 10},
#     },
# )
# plt.savefig("IRF_US_30_15_Diebold_2006.pdf", dpi=1000)
# plt.show()

###############################################################################
# Playing Around
###############################################################################
# Create a plot
# ind_pro_us_diff = ind_pro_us_diff.loc[start_us:end_us]
# us_rec = us_rec.loc[start_us:end_us]

# plt.figure(figsize=(15, 10))
# plt.plot(
#     ind_pro_us_diff.index,
#     ind_pro_us_diff,
#     label="Industrial Production",
#     color="#69b3a2",
# )

# # Adding recession bars
# start_date = None  # Initialize start_date to None
# for i in range(1, len(us_rec)):
#     if us_rec.iloc[i] == 1 and us_rec.iloc[i - 1] == 0:
#         start_date = us_rec.index[i]
#     if us_rec.iloc[i] == 0 and us_rec.iloc[i - 1] == 1:
#         end_date = us_rec.index[i]
#         plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
#         start_date = None  # Reset start_date after plotting

# # Handle the case where the series ends in a recession
# if start_date is not None:
#     plt.axvspan(start_date, ind_pro_us_diff.index[-1], color="lightgray", alpha=0.8)

# # Adding labels and legend
# plt.xlabel("Date")
# plt.ylabel("Industrial Production Index")
# plt.title("Industrial Production with NBER Recessions")
# plt.legend()
# plt.show()


# # Fetch the data
# ind_pro_us = fred.get_series("INDPRO")
# us_rec = fred.get_series("USREC")

# # Ensure the series are in DataFrame format
# ind_pro_df = pd.DataFrame(ind_pro_us, columns=["INDPRO"])
# us_rec_df = pd.DataFrame(us_rec, columns=["USREC"])

# # Create a plot
# plt.figure(figsize=(15, 10))
# plt.plot(
#     ind_pro_df.index,
#     ind_pro_df["INDPRO"],
#     label="Industrial Production",
#     color="#69b3a2",
# )

# # Adding recession bars
# start_date = None  # Initialize start_date to None
# for i in range(1, len(us_rec_df)):
#     if us_rec_df["USREC"].iloc[i] == 1 and us_rec_df["USREC"].iloc[i - 1] == 0:
#         start_date = us_rec_df.index[i]
#     if us_rec_df["USREC"].iloc[i] == 0 and us_rec_df["USREC"].iloc[i - 1] == 1:
#         if start_date is not None:
#             end_date = us_rec_df.index[i]
#             plt.axvspan(start_date, end_date, color="lightgray", alpha=0.8)
#             start_date = None  # Reset start_date after plotting

# # Handle the case where the series ends in a recession
# if start_date is not None:
#     plt.axvspan(start_date, ind_pro_df.index[-1], color="lightgray", alpha=0.8)

# # Adding labels and legend
# plt.xlabel("Date")
# plt.ylabel("Industrial Production Index")
# plt.title("Industrial Production with NBER Recessions")
# plt.legend()
# plt.show()


# path_ma_data = r"C:\Users\alexa\Documents\Studium\MSc (WU)\Master Thesis\Analysis\Data"
# df_analysis_us.to_csv(path_ma_data + "\\" + "VAR_Data_US.csv")


# # Create an empty 8x8 matrix filled with False
# lower_triangular = np.zeros((8, 8), dtype=bool)

# # Set TRUE values below or on the main diagonal
# lower_triangular[np.tril_indices_from(lower_triangular)] = True

# # Print the matrix
# print(lower_triangular)

# print(result.summary())


# plt.figure(figsize=(30, 15))
# acorr_plot = result.plot_acorr()
# plt.show()


# print(result.test_whiteness(nlags=5))


########## Factor Loadings Figure ###########
dates_factor_plot = [
    # "1987-02-01",
    # "1986-07-01",
    # "1985-07-01",
    # "1984-02-01",
    # "1983-06-01",
    # "1981-04-01",
    # "1979-10-01",
    # "1979-05-01",
    # "1990-01-01",
    # "1990-08-01",
    # "1991-08-01",
    "2019-12-01",
]

# 1987-02-01

for date in dates_factor_plot:
    print(date)
    yc_us_date = date

    yc_us = yields_us_sub_r.loc[yc_us_date].iloc[0:120]
    # lmda = yields_us_sub_r.loc[yc_us_date]["lambda"]
    lmda = 0.7308
    # beta_0 = yields_us_sub_r.loc[yc_us_date]["Level Factor"]
    # beta_1 = yields_us_sub_r.loc[yc_us_date]["Slope Factor"]
    # beta_2 = yields_us_sub_r.loc[yc_us_date]["Curvature Factor"]

    yc_us_mat_idx = list(range(len(yc_us)))
    yc_us_mat = [i + 1 for i in yc_us_mat_idx]
    yc_us_mat.insert(0, 0)  # damit 0 maturity auch inkludiert ist

    ls_beta_1_loading = {}
    for maturity in yc_us_mat:
        # print(beta_1_loading(lmda=lmda, maturity=maturity))
        result = beta_1_loading(lmda=lmda, maturity=maturity)
        ls_beta_1_loading[maturity] = result

    # ls_beta_1_loading[0] = 1 # set first value to 1

    # list(ls_beta_1_loading.keys())
    # list(ls_beta_1_loading.values())

    # plt.plot(list(ls_beta_1_loading.keys()), list(ls_beta_1_loading.values()))

    ls_beta_2_loading = {}
    for maturity in yc_us_mat:
        # print(beta_2_loading(lmda=lmda, maturity=maturity))
        result = beta_2_loading(lmda=lmda, maturity=maturity)
        ls_beta_2_loading[maturity] = result

    # ls_beta_2_loading[0] = 1 # set first value to 1

    # list(ls_beta_2_loading.keys())
    # list(ls_beta_2_loading.values())

    # plt.plot(list(ls_beta_2_loading.keys()), list(ls_beta_2_loading.values()))

    # # Plot Loadings
    # plt.figure(figsize=(10, 8))

    # plt.axhline(y=1, color="black", label="Level Factor Loading")
    # plt.plot(
    #     list(ls_beta_1_loading.keys()),
    #     list(ls_beta_1_loading.values()),
    #     label="Slope Factor Loading",
    # )
    # plt.plot(
    #     list(ls_beta_2_loading.keys()),
    #     list(ls_beta_2_loading.values()),
    #     label="Curvature Factor Loading",
    # )

    # plt.xlim(left=0)

    # plt.legend()
    # plt.show()

    # for date in yields_us_sub_r.index:
    #     yc_us_date = date
    #     print(date)

    #     yc_us = yields_us_sub_r.loc[yc_us_date].iloc[0:120]
    #     lmda = yields_us_sub_r.loc[yc_us_date]["lambda"]
    #     # beta_0 = yields_us_sub_r.loc[yc_us_date]["Level Factor"]
    #     # beta_1 = yields_us_sub_r.loc[yc_us_date]["Slope Factor"]
    #     # beta_2 = yields_us_sub_r.loc[yc_us_date]["Curvature Factor"]

    #     yc_us_mat_idx = list(range(len(yc_us)))
    #     yc_us_mat = [i + 1 for i in yc_us_mat_idx]
    #     yc_us_mat.insert(0, 0)  # damit 0 maturity auch inkludiert ist

    #     ls_beta_1_loading = {}
    #     for maturity in yc_us_mat:
    #         # print(beta_1_loading(lmda=lmda, maturity=maturity))
    #         result = beta_1_loading(lmda=lmda, maturity=maturity)
    #         ls_beta_1_loading[maturity] = result

    #     # ls_beta_1_loading[0] = 1 # set first value to 1

    #     # list(ls_beta_1_loading.keys())
    #     # list(ls_beta_1_loading.values())

    #     # plt.plot(list(ls_beta_1_loading.keys()), list(ls_beta_1_loading.values()))

    #     ls_beta_2_loading = {}
    #     for maturity in yc_us_mat:
    #         # print(beta_2_loading(lmda=lmda, maturity=maturity))
    #         result = beta_2_loading(lmda=lmda, maturity=maturity)
    #         ls_beta_2_loading[maturity] = result

    #     # ls_beta_2_loading[0] = 1 # set first value to 1

    #     # list(ls_beta_2_loading.keys())
    #     # list(ls_beta_2_loading.values())

    #     # plt.plot(list(ls_beta_2_loading.keys()), list(ls_beta_2_loading.values()))

    #     # Plot Loadings
    #     plt.figure(figsize=(10, 8))

    #     plt.axhline(y=1, color="black", label="Level")
    #     plt.plot(
    #         list(ls_beta_1_loading.keys()), list(ls_beta_1_loading.values()), label="Slope"
    #     )
    #     plt.plot(
    #         list(ls_beta_2_loading.keys()),
    #         list(ls_beta_2_loading.values()),
    #         label="Curvature",
    #     )

    #     plt.legend()
    #     plt.show()

    # # YC plot
    # # maturities_plot = list(range(11, 120, 12))
    # maturities_plot = [0, 11, 23, 35, 47, 59, 71, 83, 95, 107, 119]
    # x_labels = [yc_us.index[i] for i in maturities_plot]

    # plt.figure(figsize=(15, 10))
    # plt.plot(yc_us.index, yc_us.values, linestyle="dashdot", color="darkred")

    # plt.xticks(ticks=maturities_plot, labels=x_labels)

    # plt.show()

    # Plot YC & Factor Loadings gesamt
    fig, ax1 = plt.subplots(figsize=(15, 10))

    plt.axhline(y=1, color="black", label="Level Factor Loading")

    ax1.plot(
        list(ls_beta_1_loading.keys()),
        list(ls_beta_1_loading.values()),
        label="Slope Factor Loading",
    )

    ax1.plot(
        list(ls_beta_2_loading.keys()),
        list(ls_beta_2_loading.values()),
        label="Curvature Factor Loading",
    )

    ax2 = ax1.twinx()
    ax2.plot(yc_us.values, linestyle="dashdot", color="darkred", label="Yield Curve")

    plt.xlim(left=0)

    # plt.xticks(ticks=maturities_plot, labels=x_labels)

    ax1.set_xlabel("Maturity (in months)")
    ax1.set_ylabel("Loadings")
    ax2.set_ylabel("Yield (in %)")

    # Combine legends from both axes
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line_handles, line_labels = ax.get_legend_handles_labels()
        lines.extend(line_handles)
        labels.extend(line_labels)
    ax1.legend(lines, labels, loc="lower right", bbox_to_anchor=(0.85, -0.125), ncol=4)

    # ax1.legend(loc="lower right", bbox_to_anchor=(0.685, -0.125), ncol=3)
    # ax2.legend(loc="lower right", bbox_to_anchor=(0.8, -0.125), ncol=1)
    plt.savefig(figures_path_ma + "\\" + "Factor_Loadings_Plot.pdf", dpi=1000)
    plt.show()


########## sVAR Function ##########
df_ic_test = get_svars(df_analysis_us, lag_start=1, lag_end=6, geography="US")
print(df_ic_test.to_latex(index=False, escape=False, float_format="%.2f"))

###############################################################################
# Block Comment
###############################################################################
