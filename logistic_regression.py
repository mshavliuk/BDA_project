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


def build(samples: pd.DataFrame, outcomes: pd.DataFrame,
          test_samples: Union[pd.DataFrame, None] = None, verbose=False,
          kw_priors=None,
          **kwargs) -> stan.model.Model:
    kw_priors = kw_priors if kw_priors else dict()
    test_samples = test_samples if test_samples is not None else samples.head(1)
    with open('logistic_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N_samples': len(samples),
                 'N_test': len(test_samples),
                 'J': samples.shape[1],
                 'x_samples': samples.values,
                 'y_samples': outcomes.values,
                 'x_test': test_samples.values,
                 'beta_prior_type': BetaPriorType.Uniform,
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


def normalize_samples(samples):
    SS = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    normalized = samples.copy()
    normalized.loc[:, col_to_scale] = SS.fit_transform(samples[col_to_scale])

    return normalized

def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    samples = normalize_samples(data[data.columns.difference(['target'])])
    outcomes = data['target']
    model = build(samples, outcomes)
    fit = sample(model)
    diagnostics.k_fold_cv(build, get_disease_prob, samples, outcomes, 5)
    diagnostics.loo_within_sample(fit, outcomes)
    diagnostics.psis_loo_summary(fit, 'Logistic')

    # fit with saving warmup steps
    fit = sample(model, num_samples=0, num_warmup=200, save_warmup=True)
    diagnostics.plot_chains(fit, samples)
    plt.show()


if __name__ == '__main__':
    main()
