# Packages
import pandas as pd
import numpy as np
import random
from datetime import date, datetime
from pandas.tseries.frequencies import to_offset
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
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

# Path
path_ma = r"\\oenbnt\daten\benutzer\SCHULZA2\Daten\Persönlich\Master Thesis"

# Data
df_analysis_us = pd.read_csv(path_ma + "\\" + "Analysis_US.csv", index_col=[0])

df_analysis_us.index = pd.to_datetime(df_analysis_us.index)

model_us = VAR(df_analysis_us)
print(model_us.select_order())

result = model_us.fit(maxlags=4, ic="bic")

print(result.is_stable())


# Estimation Results (with Latex output)
result.summary()
estimates_us = result.params.round(4)
print(estimates_us.to_latex(float_format="%.4f"))

std_errors_us = result.bse.round(4)

p_vals_us = result.pvalues.round(4)

# Information Criteria
llf_us = {"Log-Likelihood": result.llf}
aic_us = {"AIC": result.aic}
bic_us = {"BIC": result.bic}
hqic_us = {"HQIC": result.hqic}

dict_ic_us = {**llf_us, **aic_us, **bic_us, **hqic_us}
print(pd.DataFrame.from_dict(dict_ic_us, orient="index").round(4).to_latex())

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
# plt.savefig("IRF_US_30_15_v2.pdf", dpi=1000)
plt.show()


# Playing Around


# Insert rows from df2 into df1
estimates_us = result.params.round(4)
estimates_us.index = (
    estimates_us.index[:1].tolist() + (estimates_us.index[1:].str[3:] + "-1").tolist()
)
# estimates_us.reset_index(inplace=True)
# estimates_us = estimates_us.iloc[:, 1:]
std_errors_us = result.bse.round(4)
std_errors_us.index = ("se_" + std_errors_us.index[:1]).tolist() + (
    "se_" + std_errors_us.index[1:].str[3:] + "-1"
).tolist()
# std_errors_us.reset_index(inplace=True)


for i in range(estimates_us.shape[0]):
    print(estimates_us.iloc[i, :])


test = pd.concat([estimates_us, std_errors_us])


index_sort = []
for i in range(estimates_us.shape[0]):
    index_sort.append(estimates_us.index[i])
    index_sort.append(std_errors_us.index[i])


test = test.reindex(index_sort)

print(test.to_latex(float_format="%.4f"))

