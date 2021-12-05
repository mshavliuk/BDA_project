data {
    int<lower=1> N_train;                               // number of training data points
    int<lower=1> N_test;                                // number of test data points
    int<lower=1> J;                                     // number of dimensions
    matrix[N_train, J] X_train;                         // trainning data    
    int<lower=0, upper=1> y_train[N_train];             // trainning outcomes
    matrix[N_test, J] X_test;                           // test data
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
    y_train ~ bernoulli_logit(alpha + X_train * beta);
}
generated quantities {         
    real<lower=0, upper=1> y_prob_pred[N_test];
    vector[N_train] log_lik = rep_vector(0, N_train);    
    real tmp;

    // Calculate LOO
    for (i in 1:N_train)
    {
        tmp = 0;
        for (j in 1:J)
        {
            tmp += beta[j] * X_train[i,j];
        }                
        log_lik[i] = bernoulli_logit_lpmf(y_train|alpha + tmp);
    }   

    // Calculate the prediction probability
    for (i in 1:N_test)
    {
        tmp = 0;
        for (j in 1:J)
        {
            tmp += beta[j] * X_test[i,j];
        }                
        y_prob_pred[i] = inv_logit(alpha + tmp); // model
    }       
}