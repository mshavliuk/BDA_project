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


def loo_within_sample(fit: Fit, outcomes: pd.DataFrame):
    false_pos, false_neg, predicted = 0, 0, 0
    
    for i, actual_outcome in enumerate(outcomes):
        prediction = fit['probs'][i].mean() >= .5
        false_pos += not actual_outcome and prediction
        false_neg += actual_outcome and not prediction
        predicted += actual_outcome == prediction

    print(f"LOO-WS score: {predicted / len(outcomes) * 100:3.2f}%", end= " | ")
    print(f"Predicted: {predicted:3d} / {len(outcomes):3d}", end= " | ")
    print(f"False positives: {false_pos} | False negatives: {false_neg}")


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
    print("-------------------")
    loo_within_sample(fit, outcomes)
    psis_loo_summary(fit)
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    display(summary)


if __name__ == '__main__':
    main()