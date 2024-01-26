rm(list=ls())

# Packages
library(tidyverse)
library(YieldCurve)
library(readxl)
library(bsvars)


# WD
setwd('C:/Users/alexa/Documents/Studium/MSc (WU)/Master Thesis/Analysis/Data')

# Data
yields_ea <- read_csv('Yields_EA.csv')

# Yields to use
cols_ea <- c('Y3M', 'Y2Y', 'Y5Y', 'Y10Y')

maturities_ea <- c(0.25, 2, 5, 10)

yields_ea$beta_0 <- NA
yields_ea$beta_1 <- NA
yields_ea$beta_2 <- NA


for (i in 1:nrow(yields_ea)){
  yields_ea_ls <- yields_ea[i, cols_ea]
  NSParameters <- Nelson.Siegel(rate = yields_ea_ls, maturity = maturities_ea)
  yields_ea[i, c('beta_0')] <- NSParameters[,1]
  yields_ea[i, c('beta_1')] <- NSParameters[,2]
  yields_ea[i, c('beta_2')] <- NSParameters[,3]
  
}

plot(yields_ea$beta_0, type='l')
plot(yields_ea$beta_1, type='l')
plot(yields_ea$beta_2, type='l')









