library(rstan)
library(plot3D)
rstan_options(auto_write = TRUE)
library(shinystan)

library(readxl)
heart <- read_excel("~/notebooks/BDA Project/heart.xlsx")
View(heart)

binomial_model <- stan_model('Hier.stan')

#Running chains
options(mc.cores=4)

#Diagnosis of age
#####
fit_cont <- sampling(binomial_model, ContStan(heart$age,M=8), iter = 2000,chains =4)

print(fit_cont)
params = rstan::extract(fit_cont)

means <- rep(0,8)
for (j in 1:8) means[j] = mean(params$theta[,j]) 
plot(c(1,2,3,4,5,6,7,8),means, "type"="b")

hist(params$theta[,3], breaks =20, main="Posterior",
          xlab="Theta[1]")


quantile(params$theta[,1], probs=c(0.05, 0.95))

traceplot(fit_cont, "theta")
#####

#Data preparation
discobs <- vector(mode="list",length=8)
discobs[[1]] <- heart[2] %>% pull(sex)
discobs[[2]] <- heart[3] %>% pull(cp)
discobs[[3]] <- heart[6] %>% pull(fbs)
discobs[[4]] <- heart[7] %>% pull(restecg)
discobs[[5]] <- heart[9] %>% pull(exang)
discobs[[6]] <- heart[11] %>% pull(slope)
discobs[[7]] <- heart[12] %>% pull(ca)
discobs[[8]] <- heart[13] %>% pull(thal)

contobs <- vector(mode="list",length=5)
contobs[[1]] <- heart[1] %>% pull(age)
contobs[[2]] <- heart[4] %>% pull(trestbps)
contobs[[3]] <- heart[5] %>% pull(chol)
contobs[[4]] <- heart[8] %>% pull(thalach)
contobs[[5]] <- heart[10] %>% pull(oldpeak)

#Tryout for discreteStan
#####
fitmodel <- sampling(binomial_model, DiscreteStan(discobs[[1]]), iter = 2000,chains =4)

print(fitmodel )
params = rstan::extract(fitmodel )

means <- rep(0,ncol(params$theta))
for (j in 1:length(means)) means[j] = mean(params$theta[,j]) 
plot(seq(1,length(means)),means, "type"="b")

hist(params$theta[,2], breaks =20, main="Posterior",
     xlab="Theta[1]")

#90% credible interval
quantile(params$theta[,1], probs=c(0.05, 0.95))
#Traceplot of chains
traceplot(fitmodel , "theta")

#predictive distribution for probability of having disiease if ca=0
 hist(params$ypred[,1])


# #3D histogram - ppl with slope=0 and ppl with age in first bin
# image2D(z= table(cut(larams$theta[,1],20),cut(params$theta[,1],20)))
launch_shinystan(fitmodel)
#####

#Tryout for continuousStan
#####
fitmodel <- sampling(binomial_model, ContStan(contobs[[1]],M=4), iter = 2000,chains =4)

print(fitmodel )
params = rstan::extract(fitmodel )

means <- rep(0,ncol(params$theta))
for (j in 1:length(means)) means[j] = mean(params$theta[,j]) 
plot(seq(1,length(means)),means, "type"="b")

hist(params$theta[,2], breaks =20, main="Posterior",
     xlab="Theta[1]")

#90% credible interval
quantile(params$theta[,1], probs=c(0.05, 0.95))
#Traceplot of chains
traceplot(fitmodel , "theta")

#predictive distribution for probability of having disiease if ca=0
hist(params$ypred[,1])


# #3D histogram - ppl with slope=0 and ppl with age in first bin
# image2D(z= table(cut(larams$theta[,1],20),cut(params$theta[,1],20)))
launch_shinystan(fitmodel)
#####

