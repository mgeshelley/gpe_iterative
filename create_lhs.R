library(lhs)
library(MASS)

set.seed(123)

n_points = 200
# lhs_samp = maximinLHS(n_points,2,10,method='build')
lhs_samp = improvedLHS(n_points,2,10)

write.matrix(lhs_samp,'lhs_samp.dat')