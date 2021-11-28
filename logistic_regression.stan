data {
    int<lower=1> N; // number of data points
    int<lower=1> J; // number of dimensions
    matrix[N, J] x;                           // data
    int<lower=0, upper=1> y[N];              // outcomes
}

transformed data {
    vector[J] min_x;
    vector[J] scale_x;
    matrix[N, J] x_std;
    for (j in 1:J) 
    {
        min_x[j] = min(x[,j]);
        scale_x[j] = (max(x[,j]) - min_x[j]);
        x_std[,j] = (x[,j] - min_x[j]) / scale_x[j];
    }
}

parameters {
    real alpha;        //intercept
    vector[J] beta;    // regression coefficient
}

model {
    // Prior
    alpha ~ beta(1, 1);
    beta ~ cauchy(0, 1);

    // Likelihood / distribution of y
    y ~ bernoulli_logit(alpha + x * beta);
}
generated quantities {         
    real<lower=0, upper=1> probs[N];
    vector[N] log_lik = rep_vector(0, N);
    
    real tmp;
    for (i in 1:N)
    {
        tmp = 0;
        for (j in 1:J)
        {
            tmp += beta[j] * x_std[i,j];
        }        
        probs[i] = inv_logit(alpha + tmp); // model
        log_lik[i] = bernoulli_logit_lpmf(y|alpha + tmp);
    }   
}