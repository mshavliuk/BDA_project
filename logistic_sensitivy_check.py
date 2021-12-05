import contextlib
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from IPython.display import display
from stan.fit import Fit
from sklearn.model_selection import KFold

def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False) -> stan.model.Model:
    with open('logistic_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'X': samples.values, 'y': outcomes.values}

    return stan.build(stan_code, data=stan_data, random_seed=0)


def sample(model, verbose=False):
    return model.sample(num_chains=4, num_samples=500, num_warmup=200)


def jitter(array):
    return array + np.random.rand(*array.shape)


def psis_loo_summary(fit: Fit):
    loo = az.loo(fit, pointwise=True)
    display(loo)
    fig, ax = plt.subplots(figsize=(8, 3))
    az.plot_khat(loo, show_bins=True, ax=ax)
    ax.set_title('Loo Linear model')
    print(f"Mean Pareto K: {loo['pareto_k'].values.mean():.2f}")
    plt.show()

def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    SS = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[col_to_scale] = SS.fit_transform(data[col_to_scale])

    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']
    model = build(samples, outcomes)
    fit = sample(model)

    psis_loo_summary(fit)
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    display(summary)


if __name__ == '__main__':
    main()