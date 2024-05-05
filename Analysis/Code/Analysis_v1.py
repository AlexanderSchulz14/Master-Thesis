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
import filterpy  # for Kalman Filter
import nelson_siegel_svensson  # Nelson-Siegel YC decomposition
import io

# Get Data
from Data_US import plot_data, get_adf, df_us

########## OLS sVAR ##########
# Levels
df_analysis_us = [
    df_us["Level Factor"],
    df_us["Slope Factor"],
    df_us["Curvature Factor"],
    df_us["INDPRO"],
    df_us["Infl_US"],
    df_us["TB_3M"],
    df_us["ebp"],
    df_us["S&P_500"],
]

df_analysis_us = pd.concat(df_analysis_us, axis=1)


# Plot Data
plot_data(df_analysis_us)


# Stationarity Check
get_adf(df_analysis_us)


# Estimate sVAR
model_us = VAR(df_analysis_us)
print(model_us.select_order())

result = model_us.fit(maxlags=5, ic="aic")
result.summary()

# print(result.test_whiteness())
print(result.is_stable())

residuals = result.sigma_u
resid_chol_decomp = np.linalg.cholesky(residuals)
np.linalg.eigvals(resid_chol_decomp)


# IRFs
irfs_us = result.irf(20)
plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Level Factor", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Slope Factor", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Curvature Factor", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="INDPRO", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Infl_US", signif=0.16)
plt.show()


# Differenced Data
df_analysis_us = [
    df_us["Level Factor"],
    df_us["Slope Factor"],
    df_us["Curvature Factor"],
    df_us["INDPRO_YoY"],
    df_us["Infl_US"],
    df_us["TB_3M"],
    df_us["ebp"],
    df_us["S&P_500_YoY"],
]

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


# Plot Data
plot_data(df_analysis_us)


# Stationarity Check
get_adf(df_analysis_us)


# Estimate sVAR
model_us = VAR(df_analysis_us)
print(model_us.select_order())

result = model_us.fit(maxlags=5, ic="aic")
print(result.summary())

# print(result.test_whiteness())
print(result.is_stable())

residuals = result.sigma_u
resid_chol_decomp = np.linalg.cholesky(residuals)
np.linalg.eigvals(resid_chol_decomp)


# IRFs
irfs_us = result.irf(20)
plt.figure(figsize=(30, 15))
irfs_us.plot(
    orth=True,
    signif=0.16,
    subplot_params={
        "fontsize": 8,
        #  "wspace" : 0.8,
        #  "hspace" : 0.8,
        #  "left" : 0.01
    },
)
plt.savefig("IRF_US.pdf", dpi=1000)
plt.show()

irfs_us = result.irf(20)
plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Level Factor", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Slope Factor", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Curvature Factor", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="INDPRO_YoY", signif=0.16)
plt.show()

plt.figure(figsize=(15, 5))
irfs_us.plot(orth=True, impulse="Infl_US", signif=0.16)
plt.show()

# FEVD
result.fevd(10).plot()
display(result.fevd(10).summary())


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


# maturities_to_use = ['3m', '6m',
#                      '9m', '12m',
#                      '15m', '18m',
#                      '21m', '24m',
#                      '30m', '36m',
#                      '48m', '60m',
#                      '72m', '84m',
#                      '96m', '108m',
#                      '120m']


# maturities_float = [float(mat.replace('m', '')) for mat in maturities_to_use]
# maturities_float_year = [mat/12 for mat in maturities_float]

# date = '2019-01-01'
# test_data = yields_us_sub.loc[date, maturities_to_use]


# t = np.array(maturities_float_year)
# y = test_data.values

# curve, status = calibrate_ns_ols(t, y, tau0=1.0)
# assert status.success
# print(curve)


# errors = []
# beta0_ls = {}

# for date in yields_us_sub.index:

#     try:
#         decomp_data = yields_us_sub.loc[date, maturities_to_use]

#         t = np.array(maturities_float_year)
#         y = decomp_data.values

#         curve, status = calibrate_ns_ols(t, y, tau0=1.0)
#         assert status.success
#         print(curve.beta0)
#         beta0_ls[date] = curve.beta0
#         # yields_us_sub.loc[date, 'beta0'] = curve.beta0

#     except:
#         errors.append(date)
#         # yields_us_sub.loc[date, 'beta0'] = 'NA'


#     sorted(beta0_ls.items(), key=lambda x:x[1], reverse=True)


# keys = list(beta0_ls.keys())
# values = list(beta0_ls.values())

# plt.plot(keys, values, linestyle='-')
# plt.xlabel('Date')
# plt.ylabel('Beta_0')
# # plt.title('Dictionary Values Line Plot')
# plt.grid(True)
# plt.show()


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


# Correlation
# Factor Plots
pearsonr(df_us.loc[:"1999", "Slope Factor"], df_us.loc[:"1999", "Curvature Factor"])

pearsonr(df_us.loc["2000":, "Slope Factor"], df_us.loc["2000":, "Curvature Factor"])


pearsonr(df_us["Level Factor"], df_us["(y(3) + y(24) + y(120))/3"])
pearsonr(df_us["Level Factor"], df_us["Infl_US"])

pearsonr(df_us["Slope Factor"], df_us["y(3) - y(120)"])
pearsonr(df_us["Slope Factor"], df_us["INDPRO"])
pearsonr(df_us["Slope Factor"], df_us["INDPRO_YoY"])
pearsonr(df_us["Slope Factor"], df_us["CU_US_YoY"])

plt.figure(figsize=(15, 10))
plt.plot(df_us["INDPRO_YoY"], label="INDPRO")
plt.plot(df_us["Slope Factor"] * -1, label="Slope Factor")
plt.legend()
plt.show()

pearsonr(df_us["Slope Factor"], ip_hp)


pearsonr(df_us["Curvature Factor"], df_us["2 * y(24) - y(120) - y(3)"])
