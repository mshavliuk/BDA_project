

data {
  int<lower=0> N; //number of bins
  int<lower=0> y[N]; //number of succesess for j in 1:N
  int<lower=0> n[N]; //number of trials for j
}

parameters {
    real<lower=0,upper=1> theta[N]; //prob of succes for j
    real<lower=0,upper=1> lambda;
    real<lower=0.1> kappa;
    
}

transformed parameters{
  real<lower=0> alpha;
  real<lower=0> beta;
  alpha = lambda * kappa;
  beta = (1 - lambda) * kappa;
  
}

model {
  //hyperprios
  lambda ~ uniform(0,1);
  kappa ~ pareto(0.1,1.5);
  //prior
  theta ~ beta(alpha,beta);
  //likelihood
  y ~ binomial(n,theta); 
}

generated quantities {
  vector[N] ypred;
  // Compute predictive distribution 
  for (j in 1:N) {
    ypred[j] = normal_rng(theta[j], 0.1);
  }
}
