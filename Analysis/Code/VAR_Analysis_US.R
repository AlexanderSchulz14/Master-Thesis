rm(list=ls())

# Packages
library(tidyverse)
library(YieldCurve)
library(readxl)
library(bsvars)
library(xts)
library(zoo)
library(xlsx)

# WD
setwd('C:/Users/alexa/Documents/Studium/MSc (WU)/Master Thesis/Analysis/Data')

# Data
data_us <- read_csv("VAR_Data_US.csv")
# data_us_ts <- read.zoo(data_us)

#data_us <- data_us[c(-1)]

data_us$...1 <- as.Date(data_us$...1)

# data_us_ts <- ts(data_us, order.by = data_us$...1)

# Analysis





# Playing around
data(us_fiscal_lsuw)
data("us_fiscal_ex")

series <- ts(data_us[c(-1)], frequency = 12, start = c(1973, 1))

# Create an empty 8x8 matrix
lower_triangular <- matrix(FALSE, nrow = 8, ncol = 8)

# Set TRUE values below the main diagonal
lower_triangular[row(lower_triangular) >= col(lower_triangular)] <- TRUE

B <- lower_triangular

test_bvar = specify_bsvar$new(
  data = series,
  p=4,
  B=B
)

burn_in = estimate(test_bvar, 1000)

out = estimate(burn_in, 1000)

irf = compute_impulse_responses(out, horizon = 20, standardise = TRUE)

dim(irf)

plot(irf[1, 1, 1:20, 100], type="l")

for (i in 1:8){
  plot(irf[5, i, 1:20, 100], type="l")
  print(i)
}












