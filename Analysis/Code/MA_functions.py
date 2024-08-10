import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
