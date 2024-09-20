# Clear environment
rm(list=ls())

# Packages
library(tidyverse)
library(readxl)
library(bsvars)

# WD
setwd('C:/Users/alexa/Documents/Studium/MSc (WU)/Master Thesis/Analysis/Data')

# Define maturities and yields columns
maturities_str <- c('SVENY01', 
                    'SVENY03', 
                    'SVENY05', 
                    'SVENY07', 
                    'SVENY10',
                    'SVENY15')

maturities_yrs <- c(1, 3, 5, 7, 10, 15)

# Define the Nelson-Siegel function with fixed lambda
nelson_siegel_function <- function(maturity, beta1, beta2, beta3, lambda) {
  return(beta1 + beta2 * ((1 - exp(-maturity / lambda)) / (maturity / lambda)) + 
           beta3 * (((1 - exp(-maturity / lambda)) / (maturity / lambda)) - exp(-maturity / lambda)))
}

# Fixed lambda value
lambda_fixed <- 0.7308  # You can change this value as needed

####################
# Load data
yields_sub_us <- read_csv('FED_Yields.csv')

# Add columns for the estimated beta parameters
yields_sub_us$beta_0 <- NA
yields_sub_us$beta_1 <- NA
yields_sub_us$beta_2 <- NA
yields_sub_us$lambda <- lambda_fixed  # Store the fixed lambda value

# Loop through each row to fit the Nelson-Siegel model with fixed lambda
for (i in 1:nrow(yields_sub_us)) {
  # Extract the yields for the given maturities
  yields_ls <- as.numeric(yields_sub_us[i, maturities_str])
  
  # Use nls to fit the Nelson-Siegel function with the fixed lambda
  fit <- tryCatch({
    nls(yields_ls ~ nelson_siegel_function(maturities_yrs, beta1, beta2, beta3, lambda_fixed),
        start = list(beta1 = 0.02, beta2 = -0.01, beta3 = 0.01), control = nls.control(maxiter = 500))
  }, error = function(e) NULL)  # Return NULL on error
  
  # If fit is successful (i.e., not NULL), extract the coefficients
  if (!is.null(fit) && inherits(fit, "nls")) {
    coef_fit <- coef(fit)
    yields_sub_us[i, 'beta_0'] <- coef_fit['beta1']
    yields_sub_us[i, 'beta_1'] <- coef_fit['beta2']
    yields_sub_us[i, 'beta_2'] <- coef_fit['beta3']
  }
}

# Analyze results
which.max(yields_sub_us$beta_0)
which.min(yields_sub_us$beta_0)

# View a specific row
yields_sub_us[442, 1]

# Save the results to CSV
write.csv(yields_sub_us, file = 'FED_Yields_US_R.csv', row.names = FALSE)

