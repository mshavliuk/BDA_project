data {
  int<lower=1> N; // number of data points
  int<lower=1> J; // number of dimensions
  matrix[N, J] x;                           // data
  real<lower=0, upper=1> y[N];              // outcomes
  real prior_alpha_mu;                      // prior mean for alpha
  real<lower=0> prior_alpha_sigma;          // prior std for alpha
  real prior_beta_mu;                       // prior mean for beta
  real<lower=0> prior_beta_sigma;           // prior std for beta
}
transformed data {
  vector[J] min_x;
  vector[J] scale_x;
  matrix[N, J] x_std;
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
  for (j in 1:J) {
    sigma ~ cauchy(0,10);
    alpha[j] ~ beta(1, 1);
    beta[j] ~ normal(prior_beta_mu, prior_beta_sigma);
    y ~ normal(alpha[j] + beta[j] * x_std[,j], sigma);
  }
}
generated quantities {
  vector[N] log_lik = rep_vector(0, N);
  vector<lower=0, upper=1>[N] probs = rep_vector(0, N);

  for (i in 1:N) {
    for (j in 1:J) {
      log_lik[i] += normal_lpdf(y | alpha[j] + beta[j] * x_std[i,j], sigma);
      probs[i] += alpha[j] + beta[j] * x_std[i,j];
    }
    probs[i] /= J;
  }
}
