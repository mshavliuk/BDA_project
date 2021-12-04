#Outputs are lists, with N, y, n

#DISCRETE: sex, cp, fbs, restecg, exang, slope, ca, thal
#CONTINUOUS: age, schol, thalach, oldpeak


#N is number of unique values in covariate 
#y is vector of recorded successes(heart disease=yes) for each unique value
#n is vector of total trials for each unique value
DiscreteStan <- function(covariate){
  N <- length(unique(covariate))
  
  y <- rep(0,N)
  n <- rep(0,N)
  for (j in 1:N){
    y[j] = sum(heart$target[covariate == (j-1)])
    n[j] = length(heart$target[covariate == (j-1)])
  }
  
  list(N = N, y = y, n = n)
}

#M = number of bins, y and n are the same as above
ContStan <- function(covariate,M = 5){
  step = (max(covariate) - min(covariate))/5
  
  y <- rep(0,M)
  n <- rep(0,M)
  edge <- rep(0,M+1)
  edge[1] <- min(covariate)
  for (j in 1:M){
    edge[j+1] = min(covariate) +(j)*step 
    y[j] = sum(heart$target[edge[j] <= covariate & covariate< edge[j+1]])
    n[j] = length(heart$target[edge[j] <= covariate & covariate < edge[j+1]])
  }
  
  list(N = M, y = y, n = n)
  
}