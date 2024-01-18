rm(list=ls())

# Packages
library(tidyverse)
library(YieldCurve)
library(readxl)
library(bsvars)


# WD
setwd('C:/Users/alexa/Documents/Studium/MSc (WU)/Master Thesis/Analysis/Data')

# Analysis
maturities_str <- c('3m',
                     '6m',
                     '9m',
                     '12m',
                     '15m',
                     '18m',
                     '21m',
                     '24m',
                     '30m',
                     '36m',
                     '48m',
                     '60m',
                     '72m',
                     '84m',
                     '96m',
                     '108m',
                     '120m')

maturities_yrs <- c(0.25,
                  0.5,
                  0.75,
                  1.0,
                  1.25,
                  1.5,
                  1.75,
                  2.0,
                  2.5,
                  3.0,
                  4.0,
                  5.0,
                  6.0,
                  7.0,
                  8.0,
                  9.0,
                  10.0)


####################
yields_sub_us <- read_csv('Yields_Data_Subset.csv')

yields_sub_us$beta_0 <- NA
yields_sub_us$beta_1 <- NA
yields_sub_us$beta_2 <- NA



for (i in 1:nrow(yields_sub_us)) {
  yields_ls <- yields_sub_us[i, maturities_str]
  NSParameters <- Nelson.Siegel(rate = yields_ls, maturity = maturities_yrs)
  yields_sub_us[i, c('beta_0')] <- NSParameters[,1]
  yields_sub_us[i, c('beta_1')] <- NSParameters[,2]
  yields_sub_us[i, c('beta_2')] <- NSParameters[,3]
  
}


which.max(yields_sub_us$beta_0)
which.min(yields_sub_us$beta_0)

yields_sub_us[442,1]

write.csv(yields_sub_us, file = 'Yields_US_R.csv', row.names = FALSE)










