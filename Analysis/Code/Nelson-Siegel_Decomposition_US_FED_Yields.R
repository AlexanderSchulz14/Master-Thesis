rm(list=ls())

# Packages
library(tidyverse)
library(YieldCurve)
library(readxl)
library(bsvars)


# WD
setwd('C:/Users/alexa/Documents/Studium/MSc (WU)/Master Thesis/Analysis/Data')

# Analysis
maturities_str <- c('SVENY01', 
                    'SVENY03', 
                    'SVENY05', 
                    'SVENY07', 
                    'SVENY10',
                    'SVENY15')

maturities_yrs <- c(1, 3, 5, 7, 10, 15)


####################
yields_sub_us <- read_csv('FED_Yields.csv')

yields_sub_us$beta_0 <- NA
yields_sub_us$beta_1 <- NA
yields_sub_us$beta_2 <- NA



for (i in 1:nrow(yields_sub_us)) {
  yields_ls <- yields_sub_us[i, maturities_str]
  NSParameters <- Nelson.Siegel(rate = yields_ls, maturity = maturities_yrs)
  yields_sub_us[i, c('beta_0')] <- NSParameters[,1]
  yields_sub_us[i, c('beta_1')] <- NSParameters[,2]
  yields_sub_us[i, c('beta_2')] <- NSParameters[,3]
  yields_sub_us[i, c('lambda')] <- NSParameters[,4]
  
}


which.max(yields_sub_us$beta_0)
which.min(yields_sub_us$beta_0)

yields_sub_us[442,1]

write.csv(yields_sub_us, file = 'FED_Yields_US_R.csv', row.names = FALSE)










