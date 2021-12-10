data {
  int<lower=1> N;                           // number of data points
  int<lower=1> J;                           // number of dimensions
  matrix[N, J] x;                           // data
  real<lower=0, upper=1> y[N];              // outcomes

  real prior_alpha_mu;                      // prior mean for alpha
  real<lower=0> prior_alpha_sigma;          // prior std for alpha

  int<lower=0, upper=2> beta_prior_type;    // 0. normal; 1. double_exponential; 2. uniform

  // beta prior parameters. Required for beta_prior_type == 0
  // should be passes as 1-element array. Otherwise as empty array.
  real prior_beta_mu[beta_prior_type == 0];
  real<lower=0> prior_beta_sigma[beta_prior_type == 0];
}

transformed data {
  vector[J] min_x;
  vector[J] scale_x;
  matrix<lower=0, upper=1>[N, J] x_std;
  for (j in 1:J) {
    min_x[j] = min(x[,j]);
    scale_x[j] = (max(x[,j]) - min_x[j]);
    x_std[,j] = (x[,j] - min_x[j]) / scale_x[j];
  }
}

parameters {
  real<lower=0, upper=1> alpha;
  vector[J] beta;
  real<lower=0> sigma;
}

model {
  sigma ~ cauchy(0,10);
  alpha ~ normal(prior_alpha_mu, prior_alpha_sigma);
  if (beta_prior_type == 0) {
    beta ~ normal(prior_beta_mu[1], prior_beta_sigma[1]);
  } else if (beta_prior_type == 1) {
    beta ~ double_exponential(0, 1);
  } else {
    // beta is uniform
  }
  y ~ normal(alpha + x_std * beta, sigma);
}

generated quantities {
  vector[N * J] log_lik;
  vector[N] probs;

  for (i in 1:N) {
    for(j in 1: J) {
      log_lik[(i - 1) * J + j] = normal_lpdf(y[i] | alpha + x_std[i,j] * beta[j], sigma);
    }
    probs[i] = alpha + x_std[i] * beta;
  }
}
