import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, SVAR
import statsmodels.tsa.vector_ar.svar_model as svar

sns.set_theme(style="darkgrid")
from statsmodels.tsa.stattools import adfuller


# Auxiliary Functions
# Plot Function
def plot_data(data):
    for col in data.columns:
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes()
        ax.plot(data[col])
        ax.set_title(col)
        ax.legend()
        plt.show()


# ADF Test Function
# adf_dict = {}


# def get_adf(data):
#     for col in data.columns:
#         result = adfuller(data[col])
#         p_val = result[1]
#         adf_dict[col] = p_val
#         print(str(col) + ": " + str(p_val))


def get_adf(data):

    adf_dict = {}

    if not adf_dict:
        for col in data.columns:
            result = adfuller(data[col])
            t_stat = result[0]
            critical_val = result[4]["5%"]
            p_val = result[1]

            adf_dict[col] = [t_stat, critical_val, p_val]
            # print(str(col) + ": " + str(p_val))

        return adf_dict
    else:
        adf_dict.clear()
        print("Non-empty dictionary. Please try again!")


# Coefficient Approximation Function
def get_beta_0_approx(data, time, yield_1="3m", yield_2="24m", yield_3="120m"):
    calc_data = data.loc[time, [yield_1, yield_2, yield_3]]
    beta_0_approx = calc_data.mean()

    return beta_0_approx


# Beta 1 Loading
def beta_1_loading(lmda, maturity):
    result = (1 - np.exp(-lmda * maturity)) / (lmda * maturity)
    return result


# Beta 2 Loading
def beta_2_loading(lmda, maturity):
    result = (1 - np.exp(-lmda * maturity)) / (lmda * maturity) - np.exp(
        -lmda * maturity
    )
    return result


# sVAR Function yielding IRFs and IC  for multiple lag cases
# Empty Information Criteria Dataframe for storing IC
# df_ic = pd.DataFrame(
#     columns=[
#         "Lag",
#         "Log-Likelihood",
#         "AIC",
#         "BIC",
#         "HQIC",
#     ]
# )

# ls_ic_row = []


def get_svars(data, lag_start: int, lag_end: int, geography: str):
    df_ic = pd.DataFrame(
        columns=[
            "Lag",
            "Log-Likelihood",
            "AIC",
            "BIC",
            "HQIC",
        ]
    )
    ls_ic_row = []

    model = VAR(data)

    for lag in range(lag_start, lag_end + 1):
        # Estimation
        result = model.fit(maxlags=lag, ic="aic")

        # Information Criteria
        ic_row = {
            "Lag": f"$p={lag}$",
            "Log-Likelihood": result.llf,
            "AIC": result.aic,
            "BIC": result.bic,
            "HQIC": result.hqic,
        }

        ls_ic_row.append(ic_row)

        if lag == lag_end:
            df_ic = pd.concat([df_ic, pd.DataFrame(ls_ic_row)], ignore_index=True)
        else:
            pass

        # IRF
        irfs_us = result.irf(36)
        irfs_us.plot(
            orth=True,
            signif=0.1,
            figsize=(30, 15),
            plot_params={"legend_fontsize": 20},
            subplot_params={
                "fontsize": 15,
            },
        )
        plt.savefig(f"IRF_{geography}_lag_{lag}.pdf", dpi=1000)
        print(f"Figure IRF_{geography}_lag_{lag}.pdf has been saved!")
        plt.show()

    print(df_ic.to_latex(index=False, escape=False, float_format="%.2f"))
    return df_ic
