# Clear environment
rm(list=ls())

# Packages
library(tidyverse)
library(readxl)
library(bsvars)

# WD
setwd('C:/Users/alexa/Documents/Studium/MSc (WU)/Master Thesis/Analysis/Data')

# Data
yields_ea <- read_csv('Yields_EA.csv')

# Yields to use
cols_ea <- c('Y3M', 
             'Y6M',
             'Y9M',
             'Y1Y', 
             'Y2Y',
             'Y3Y',
             'Y4Y',
             'Y5Y',
             'Y6Y',
             'Y7Y', 
             'Y8Y', 
             'Y10Y')

maturities_ea <- c(0.25, 
                   0.5,
                   0.75,
                   1, 
                   2, 
                   3,
                   4,
                   5,
                   6,
                   7,
                   8,
                   10)

# Fixed lambda value
lambda_fixed <- 0.7308

# Nelson-Siegel function with fixed lambda
nelson_siegel_function <- function(maturity, beta1, beta2, beta3, lambda) {
  return(beta1 + beta2 * ((1 - exp(-maturity / lambda)) / (maturity / lambda)) + 
           beta3 * (((1 - exp(-maturity / lambda)) / (maturity / lambda)) - exp(-maturity / lambda)))
}

# Add columns for the estimated beta parameters
yields_ea$beta_0 <- NA
yields_ea$beta_1 <- NA
yields_ea$beta_2 <- NA
yields_ea$lambda <- lambda_fixed  # Store the fixed lambda value

# Loop through each row to fit the Nelson-Siegel model with fixed lambda
for (i in 1:nrow(yields_ea)){
  # Extract the yields for the given maturities
  yields_ea_ls <- as.numeric(yields_ea[i, cols_ea])
  
  # Use nls to fit the Nelson-Siegel function with the fixed lambda
  fit <- tryCatch({
    nls(yields_ea_ls ~ nelson_siegel_function(maturities_ea, beta1, beta2, beta3, lambda_fixed),
        start = list(beta1 = 0.02, beta2 = -0.01, beta3 = 0.01), control = nls.control(maxiter = 500))
  }, error = function(e) NULL)  # Return NULL on error
  
  # If fit is successful, extract the coefficients
  if (!is.null(fit) && inherits(fit, "nls")) {
    coef_fit <- coef(fit)
    yields_ea[i, 'beta_0'] <- coef_fit['beta1']
    yields_ea[i, 'beta_1'] <- coef_fit['beta2']
    yields_ea[i, 'beta_2'] <- coef_fit['beta3']
  }
}



# Write the result to CSV
write.csv(yields_ea, file = 'Yields_EA_R_v2.csv', row.names = FALSE)
