data {
  int<lower=1> N_groups;
  int<lower=1> N_samples[N_groups];                   // number of training data points
  int<lower=1> J;                                     // number of dimensions
  matrix[sum(N_samples), J] x_samples;                // data
  int<lower=0, upper=1> y_samples[sum(N_samples)];    // outcomes
}

parameters {
  real beta_mu;
  real<lower=0> beta_sigma;
  real alpha[N_groups];        // intercept
  vector[J] beta[N_groups];    // regression coefficient
}

model {
  // HyperPriors
  beta_mu ~ normal(0, 1);
  beta_sigma ~ cauchy(0,10);
  int group_start = 1;

  for (i in 1:N_groups) {
    // Prior
    alpha[i] ~ student_t(2, 0, 10);
    beta[i] ~ normal(beta_mu, beta_sigma);

    // Likelihood
    for(j in group_start:(group_start + N_samples[i] - 1)) {
      y_samples[j] ~ bernoulli_logit(alpha[i] + x_samples[j] * beta[i]);
    }

    group_start += N_samples[i];
  }
}

generated quantities {         
  real<lower=0, upper=1> probs[sum(N_samples)];
  vector[sum(N_samples) * J] log_lik;
  int group_start = 1;

  for (i in 1:N_groups) {
    for(j in group_start:(group_start + N_samples[i] - 1)) {
      for (k in 1:J) {
        log_lik[(j - 1) * J + k] = bernoulli_logit_lpmf(y_samples[j] | alpha[i] + x_samples[j, k] * beta[i][k]);
      }
      probs[j] = inv_logit(alpha[i] + x_samples[j] * beta[i]); // model
    }

    group_start += N_samples[i];
  }
}
