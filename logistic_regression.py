import contextlib
import time
from datetime import datetime

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from IPython.display import display
from stan.fit import Fit



def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False) -> stan.model.Model:
    with open('logistic_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'x': samples.values, 'y': outcomes.values}

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


def get_disease_prob(fit, data: pd.DataFrame, sample):
    return fit['probs'][1999:].mean(axis=0)


def leave_one_out(samples: pd.DataFrame, outcomes: pd.DataFrame):
    num = samples.shape[0]
    false_pos, false_neg, predicted = 0, 0, 0
    start_time = time.time()
    for i in range(num):
        sample_query = samples.index.isin([i])
        test_sample = samples[sample_query]
        loo_samples = samples[~sample_query]
        outcome_query = outcomes.index.isin([i])
        test_outcome = outcomes[outcome_query].values[0]
        loo_outcomes = outcomes[~outcome_query]

        model = build(loo_samples, loo_outcomes)
        fit = model.sample(num_chains=4, num_samples=200, num_warmup=200)

        prob = get_disease_prob(fit, loo_samples, test_sample)
        false_pos += not test_outcome and (prob >= .5)
        false_neg += test_outcome and (prob < .5)
        predicted += test_outcome == (prob >= .5)
        eta = datetime.fromtimestamp(time.time() + (time.time() - start_time) / (i + 1) * num)
        print(f'ETA: {eta.strftime("%H:%M:%S")}', end=' | ')
        print(f"LOO score: {predicted / (i + 1) * 100:3.2f}%", end=' | ')
        print(f"Predicted: {predicted:3d} / {i + 1:3d}", end=' | ')
        print(f"False positives: {false_pos} | False negatives: {false_neg}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    return predicted

def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    #samples = data[data.columns.difference(['target'])]
    samples = data[['cp', 'trestbps', 'thalach', 'ca', 'oldpeak', 'thal']]
    outcomes = data['target']
    model = build(samples, outcomes)
    fit = sample(model)
    # leave_one_out(samples, outcomes) # takes an hour to execute
    psis_loo_summary(fit)
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    display(summary)


if __name__ == '__main__':
    main()