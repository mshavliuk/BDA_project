import contextlib

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import stan
from IPython.core.display import display
from sklearn.preprocessing import StandardScaler

from utils import suppress_stdout_stderr
from diagnostics import psis_loo_summary, k_fold_cv


def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False) -> stan.model.Model:
    with open('logistic_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'X': samples.values, 'y': outcomes.values}
    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return stan.build(stan_code, data=stan_data, random_seed=0)


def build_for_accuracy_check(samples, outcomes, test_samples, verbose=False) -> stan.model.Model:
    with open('logistic_regression_accuracy.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N_train': len(outcomes),
                 'N_test': len(test_samples),
                 'J': samples.shape[1],
                 'X_train': samples,
                 'y_train': outcomes,
                 'X_test': test_samples}

    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return stan.build(stan_code, data=stan_data, random_seed=0)


def sample(model, verbose=False):
    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return model.sample(num_chains=4, num_samples=500, num_warmup=200)


def get_disease_prob(fit):
    return fit['y_prob_pred'].mean(axis=1)


def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    SS = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[col_to_scale] = SS.fit_transform(data[col_to_scale])

    samples = data[['cp', 'trestbps', 'thalach', 'ca', 'oldpeak']]
    outcomes = data['target']
    model = build(samples, outcomes, True)
    fit = sample(model)
    k_fold_cv(samples, outcomes, 50)
    psis_loo_summary(fit, 'Logistic')
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    display(summary)


if __name__ == '__main__':
    main()
