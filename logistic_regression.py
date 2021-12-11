import contextlib
from typing import Union

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import stan
from IPython.core.display import display
from sklearn.preprocessing import StandardScaler

import diagnostics
from utils import suppress_stdout_stderr


class BetaPriorType:
    Normal = 0
    DoubleExponential = 1
    Uniform = 2


def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False) -> stan.model.Model:
    with open('logistic_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'X': samples.values, 'y': outcomes.values}
    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return stan.build(stan_code, data=stan_data, random_seed=0)


def build_for_accuracy_check(samples: pd.DataFrame, outcomes: pd.DataFrame,
                             test_samples: Union[pd.DataFrame, None] = None, verbose=False,
                             kw_priors=None,
                             **kwargs) -> stan.model.Model:
    kw_priors = kw_priors if kw_priors else dict()
    test_samples = test_samples if test_samples is not None else samples.head(1)
    with open('logistic_regression_accuracy.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N_samples': len(samples),
                 'N_test': len(test_samples),
                 'J': samples.shape[1],
                 'x_samples': samples.values,
                 'y_samples': outcomes.values,
                 'x_test': test_samples.values,
                 'prior_beta_mu': [], 'prior_beta_sigma': [],
                 **kw_priors}

    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return stan.build(stan_code, data=stan_data, random_seed=0)


def sample(model, verbose=False, **kwargs):
    defaults = dict(num_chains=4, num_samples=500, num_warmup=200)
    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return model.sample(**{**defaults, **kwargs})


def get_disease_prob(fit, i, *args):
    return fit['y_prob_pred'][i].mean()


def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    SS = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[col_to_scale] = SS.fit_transform(data[col_to_scale])

    samples = data[['cp', 'trestbps', 'thalach', 'ca', 'oldpeak']]
    outcomes = data['target']
    diagnostics.k_fold_cv(build_for_accuracy_check, get_disease_prob, samples, outcomes, 5)
    model = build(samples, outcomes, True)
    fit = sample(model)
    diagnostics.loo_within_sample(fit, outcomes)
    diagnostics.psis_loo_summary(fit, 'Logistic')

    # fit with saving warmup steps
    fit = sample(model, num_samples=0, num_warmup=200, save_warmup=True)
    diagnostics.plot_chains(fit, samples)
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    plt.show()
    display(summary)


if __name__ == '__main__':
    main()
