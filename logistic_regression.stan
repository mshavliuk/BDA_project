data {
    int<lower=1> N;                                     // number of data points
    int<lower=1> J;                                     // number of dimensions
    matrix[N, J] X;                                     // model input data    
    int<lower=0, upper=1> y[N];                         // outcomes
}

parameters {
    real alpha;        // intercept
    vector[J] beta;    // regression coefficient
}

model {
    // Prior
    alpha ~ student_t(2, 0, 10);
    beta ~ normal(0, 1);

    // Likelihood / distribution of y
    y ~ bernoulli_logit(alpha + X * beta);
}
generated quantities {     
    real<lower=0, upper=1> probs[N];    
    vector[N * J] log_lik;

    // Calculate LOO
    for (i in 1:N) {
      for(j in 1:J) {
        log_lik[(i - 1) * J + j] = bernoulli_logit_lpmf(y[i] | alpha + X[i, j] * beta[j]);
      }
      probs[i] = inv_logit(alpha + X[i] * beta); // model
    }         
}
