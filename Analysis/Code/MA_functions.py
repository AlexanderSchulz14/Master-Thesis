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
adf_dict = {}


def get_adf(data):
    for col in data.columns:
        result = adfuller(data[col])
        p_val = result[1]
        adf_dict[col] = p_val
        print(str(col) + ": " + str(p_val))


# Coefficient Approximation Function
def get_beta_0_approx(data, time, yield_1="3m", yield_2="24m", yield_3="120m"):
    calc_data = data.loc[time, [yield_1, yield_2, yield_3]]
    beta_0_approx = calc_data.mean()

    return beta_0_approx
