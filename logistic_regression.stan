data {
  int<lower=1> N_samples;                             // number of training data points
  int<lower=1> N_test;                                // number of test data points
  int<lower=1> J;                                     // number of dimensions
  matrix[N_samples, J] x_samples;                     // trainning data
  int<lower=0, upper=1> y_samples[N_samples];         // trainning outcomes
  matrix[N_test, J] x_test;                           // test data

  int<lower=0, upper=2> beta_prior_type;    // 0. normal; 1. double_exponential; 2. uniform

  // beta prior parameters. Required for beta_prior_type == 0
  // should be passes as 1-element array. Otherwise as empty array.
  real prior_beta_mu[beta_prior_type == 0];
  real<lower=0> prior_beta_sigma[beta_prior_type == 0];
}

parameters {
  real alpha;        // intercept
  vector[J] beta;    // regression coefficient
}

model {
  // Prior
  alpha ~ student_t(2, 0, 10);

  if (beta_prior_type == 0) {
    beta ~ normal(prior_beta_mu[1], prior_beta_sigma[1]);
  } else if (beta_prior_type == 1) {
    beta ~ double_exponential(0, 1);
  } else {
    // beta is uniform
  }

  // Likelihood / distribution of y
  y_samples ~ bernoulli_logit(alpha + x_samples * beta);
}

generated quantities {         
  real<lower=0, upper=1> y_prob_pred[N_test];
  real<lower=0, upper=1> probs[N_samples];
  vector[N_samples] log_lik;

  // Calculate LOO
  for (i in 1:N_samples) {
    log_lik[i] = bernoulli_logit_lpmf(y_samples[i] | alpha + x_samples[i] * beta);
    probs[i] = inv_logit(alpha + x_samples[i] * beta); // model
  }

  // Calculate the prediction probability
  for (i in 1:N_test) {
      y_prob_pred[i] = inv_logit(alpha + x_test[i] * beta); // model
  }
}
