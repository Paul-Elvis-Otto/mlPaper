# Full Regression Model Summary

```r

Call:
lm(formula = v2x_corr ~ v2x_libdem, data = df)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.49530 -0.20223 -0.01509  0.20058  0.61807 

Coefficients:
             Estimate Std. Error t value
(Intercept)  0.563006   0.002144  262.63
v2x_libdem  -0.596047   0.006689  -89.11
            Pr(>|t|)    
(Intercept)   <2e-16 ***
v2x_libdem    <2e-16 ***
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.247 on 25393 degrees of freedom
  (2518 observations deleted due to missingness)
Multiple R-squared:  0.2382,	Adjusted R-squared:  0.2382 
F-statistic:  7940 on 1 and 25393 DF,  p-value: < 2.2e-16

```
