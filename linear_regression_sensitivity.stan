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
  for (j in 1:J) {
    alpha[j] ~ beta(1, 1);
    beta[j] ~ double_exponential(0, 1);
    y ~ normal(alpha[j] + beta[j] * x_std[,j], sigma);
  }
}
generated quantities {
  vector[N * J] log_lik;
  vector<lower=0, upper=1>[N] probs = rep_vector(0, N);

  for (i in 1:N) {
    for (j in 1:J) {
      log_lik[(i - 1) * J + j] = normal_lpdf(y | alpha[j] + beta[j] * x_std[i,j], sigma);
      probs[i] += alpha[j] + beta[j] * x_std[i,j];
    }
    probs[i] /= J;
  }
}
