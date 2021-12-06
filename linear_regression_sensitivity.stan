data {
  int<lower=1> N;                           // number of data points
  int<lower=1> J;                           // number of dimensions
  matrix[N, J] x;                           // data
  real<lower=0, upper=1> y[N];              // outcomes
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
  vector<lower=0, upper=1>[J] alpha;
  vector[J] beta;
  real<lower=0> sigma;
}

model {
  sigma ~ cauchy(0,10);
  //alpha ~ beta(1, 1);
  //beta ~ normal(prior_beta_mu, prior_beta_sigma);
  y ~ normal(alpha + x_std * beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] probs;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y | alpha + x_std[i] * beta, sigma);
    probs[i] = alpha + x_std[i] * beta;
  }
}
