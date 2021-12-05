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

from utils import suppress_stdout_stderr


def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False) -> stan.model.Model:
    with open('linear_regression_sensitivity.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'x': samples.values, 'y': outcomes.values}

    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return stan.build(stan_code, data=stan_data, random_seed=0)


def sample(model, verbose=False):
    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
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


def get_prob(fit, data: pd.DataFrame, sample):
    prob = np.zeros(fit.num_chains * fit.num_samples)
    scale = (data.max() - data.min())
    sample_std = (sample - data.min()) / scale
    for i, param_name in enumerate(data.columns):
        alpha_i = fit['alpha'][i]
        beta_i = fit['beta'][i]
        x_i = sample_std[param_name].iloc[0]
        prob += alpha_i + beta_i * x_i

    prob /= sample.shape[1]
    return prob


def loo_cv(samples: pd.DataFrame, outcomes: pd.DataFrame):
    num = samples.shape[0]
    false_pos, false_neg, predicted = 0, 0, 0
    start_time = time.time()
    for i in range(num):
        sample_query = samples.index.isin([i])
        test_sample = samples[sample_query]
        outcome_query = outcomes.index.isin([i])
        test_outcome = outcomes[outcome_query].values[0]

        with suppress_stdout_stderr():
            loo_outcomes = outcomes[~outcome_query]
            loo_samples = samples[~sample_query]
            model = build(loo_samples, loo_outcomes)
            fit = model.sample(num_chains=4, num_samples=200, num_warmup=200)

        prob_mean = get_prob(fit, samples, test_sample).mean()
        false_pos += not test_outcome and (prob_mean >= .5)
        false_neg += test_outcome and (prob_mean < .5)
        predicted += test_outcome == (prob_mean >= .5)
        eta = datetime.fromtimestamp(time.time() + (time.time() - start_time) / (i + 1) * num)
        print(f'ETA: {eta.strftime("%H:%M:%S")}', end=' | ')
        print(f"LOO-CV score: {predicted / (i + 1) * 100:3.2f}%", end=' | ')
        print(f"Predicted: {predicted:3d} / {i + 1:3d}", end=' | ')
        print(f"False positives: {false_pos} | False negatives: {false_neg}", end='\r', flush=True)

    print(f"\nTotal time: {time.time() - start_time}")


def loo_within_sample(fit: Fit, outcomes: pd.DataFrame):
    false_pos, false_neg, predicted = 0, 0, 0
    for i, actual_outcome in enumerate(outcomes):
        prediction = fit['probs'][i].mean() >= .5
        false_pos += not actual_outcome and prediction
        false_neg += actual_outcome and not prediction
        predicted += actual_outcome == prediction
    print(f"LOO-WS score: {predicted / len(outcomes) * 100:3.2f}%", end=' | ')
    print(f"Predicted: {predicted:3d} / {len(outcomes):3d}", end=' | ')
    print(f"False positives: {false_pos} | False negatives: {false_neg}")


def plot_draws(fit, samples):
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), sharey=True,
                             gridspec_kw=dict(left=0.05, right=0.98, bottom=0.05, top=0.98,
                                              wspace=0.1, hspace=0.3))
    axes = axes.ravel()
    line_style = dict(color='C3', linewidth=2, alpha=0.8)
    for i, param_name in enumerate(samples.columns):
        probs = fit['probs'][:, :200]
        values = samples[param_name].values
        beta = np.broadcast_to(values, probs.T.shape).T
        axes[i].scatter(y=probs.ravel(), x=jitter(beta).ravel(), s=4, color='#00f', alpha=0.01)
        axes[i].set_title(param_name)
        axes[i].set_ylabel(r"$P_{disease}$")

        xs = (values.min(), values.max() + 1)
        alpha = fit['alpha'][i].mean()
        beta = fit['beta'][i].mean()
        ys = (alpha, alpha + beta)
        axes[i].plot(xs, ys, **line_style)
        axes[i].set_ybound((0, 1))
        axes[i].annotate(f"$\\alpha = {alpha:.3f}$\n$\\beta = {beta:.3f}$", xy=(0.7, 0.1),
                         xycoords='axes fraction', fontsize=14)

    axes[samples.shape[1]].legend(
        (
            plt.Line2D([], [], **line_style),
            plt.Line2D([], [], linestyle='', marker='o', markersize=10, color='#00f', alpha=0.8)
        ),
        (
            r'linear correlation $\alpha + x \times \beta$',
            'draws'
        ),
        loc='upper left',
        fontsize=16
    )
    for ax in axes[samples.shape[1]:]:
        ax.set_axis_off()
    plt.show()


def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']
    model = build(samples, outcomes)
    fit = sample(model)
    #plot_draws(fit, samples)
    #loo_cv(samples, outcomes)
    #loo_within_sample(fit, outcomes)
    psis_loo_summary(fit)
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta', 'sigma'])
    display(summary)


if __name__ == '__main__':
    main()
