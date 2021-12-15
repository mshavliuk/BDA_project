
import itertools
import contextlib
from typing import Union, List

import matplotlib.pyplot as plt
import pandas as pd
import stan
import numpy as np
from sklearn.preprocessing import StandardScaler

import diagnostics
from utils import suppress_stdout_stderr


class BetaPriorType:
    Normal = 0
    DoubleExponential = 1
    Uniform = 2


def build(samples: pd.DataFrame, outcomes: pd.Series, n_groups: int, n_samples: List[int],
          test_samples: Union[pd.DataFrame, None] = None, verbose=False,
          kw_priors=None,
          **kwargs) -> stan.model.Model:
    kw_priors = kw_priors if kw_priors else dict()
    with open('hierarchical.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {
        'N_groups': n_groups,
        'N_samples': n_samples,
        'J': samples.shape[1],
        'x_samples': samples.values,
        'y_samples': outcomes.values,
        **kw_priors
    }

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

    groups = samples.groupby(samples['thal'])
    samples.loc[:, :] = list(itertools.chain(*(group.values for i, group in groups)))

    # samples = data[data.columns.difference(['target'])]
    outcomes = pd.Series(itertools.chain(*([data['target'][g].values for g in groups.groups.values()])))

    model = build(samples, outcomes, kw_priors={
                 'beta_prior_type': BetaPriorType.Normal,
                 'prior_beta_mu': [0], 'prior_beta_sigma': [1]
    }, n_groups=len(groups), n_samples=[len(group) for i, group in groups], verbose=True)
    fit = sample(model, verbose=True)

    diagnostics.plot_posterior(fit)
    diagnostics.loo_within_sample(fit, outcomes)
    diagnostics.convergence(fit, var_names=['alpha', 'beta'])
    diagnostics.psis_loo_summary(fit, 'Logistic')
    #
    # diagnostics.k_fold_cv(build, get_disease_prob, samples, outcomes, 5)
    # # fit with saving warmup steps
    # fit = sample(model, num_samples=0, num_warmup=200, save_warmup=True)
    # diagnostics.plot_chains(fit, samples)
    # ws accuracy is 86.8% for grouping by cp
    #
    plt.show()


if __name__ == '__main__':
    main()
