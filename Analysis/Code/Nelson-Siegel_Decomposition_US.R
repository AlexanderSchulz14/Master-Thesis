install.packages("YieldCurve")
library(YieldCurve)

maturities <- c(0.08333333333333333, 0.25, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0)
yields <- c(2.36180471, 2.39571285, 
            2.44879254, 2.51586724, 
            2.45099558, 2.43416199, 
            2.53235959, 2.64622369)

NSParameters <- Nelson.Siegel(rate = yields, maturity = maturities)
NSParameters[,1]


data(FedYieldCurve)
maturity.Fed <- c(3/12, 0.5, 1,2,3,5,7,10)
NSParameters <- Nelson.Siegel( rate=first(FedYieldCurve,'10 month'),	maturity=maturity.Fed)
y <- NSrates(NSParameters[5,], maturity.Fed)
plot(maturity.Fed,FedYieldCurve[5,],main="Fitting Nelson-Siegel yield curve",
     xlab=c("Pillars in months"), type="o")
lines(maturity.Fed,y, col=2)
legend("topleft",legend=c("observed yield curve","fitted yield curve"),
       col=c(1,2),lty=1)
grid()