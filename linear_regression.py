import contextlib
import time
from datetime import datetime

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from IPython.display import display
from sklearn.model_selection import KFold
from stan.fit import Fit

from utils import suppress_stdout_stderr


class BetaPriorType:
    Normal = 0
    DoubleExponential = 1
    Uniform = 2


def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False,
          kw_priors=None) -> stan.model.Model:
    kw_priors = kw_priors if kw_priors else dict()
    with open('linear_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'x': samples.values, 'y': outcomes.values,
                 'prior_alpha_mu': 0.5, 'prior_alpha_sigma': 10,
                 'prior_beta_mu': [], 'prior_beta_sigma': [],
                 'beta_prior_type': BetaPriorType.Uniform, **kw_priors}

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


def get_prob(fit, data: pd.DataFrame, sample: pd.DataFrame):
    prob = fit['alpha']
    scale = (data.max() - data.min())
    sample_std = (sample - data.min()) / scale
    for i, param_name in enumerate(data.columns):
        beta_i = fit['beta'][i]
        x_i = sample_std[param_name]
        prob += beta_i * x_i

    return prob


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
    probs = fit['probs'][:, :250]
    for i, param_name in enumerate(samples.columns):
        values = samples[param_name].values
        beta = np.broadcast_to(values, probs.T.shape).T
        axes[i].scatter(y=probs.ravel(), x=jitter(beta).ravel(), s=4, color='#00f', alpha=0.01)
        axes[i].set_title(param_name)
        axes[i].set_ylabel(r"$P_{disease}$")

        xs = (values.min(), values.max() + 1)
        beta = fit['beta'][i].mean()
        ys = (probs.mean() - beta / 2, probs.mean() + beta / 2)
        axes[i].plot(xs, ys, **line_style)
        axes[i].annotate(f"$\\beta = {beta:.3f}$", xy=(0.7, 0.1),
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


def k_fold_cv(samples: pd.DataFrame, outcomes: pd.DataFrame, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    false_pos, false_neg, predicted = 0, 0, 0
    tot_test_len = 0
    start_time = time.time()
    for i, (fit_idx, _) in enumerate(kf.split(samples)):
        fit_query = samples.index.isin(fit_idx)
        samples_fit, samples_test = samples[fit_query], samples[~fit_query]
        outcomes_fit, outcomes_test = outcomes[fit_query], outcomes[~fit_query]

        with suppress_stdout_stderr():
            model = build(samples_fit, outcomes_fit)
            fit = model.sample(num_chains=4, num_samples=200, num_warmup=200)

        tot_test_len += len(outcomes_test)
        for test_sample, actual_outcome in zip(samples_test.iloc, outcomes_test):
            prob_mean = get_prob(fit, samples, test_sample).mean()
            false_pos += not actual_outcome and (prob_mean >= .5)
            false_neg += actual_outcome and (prob_mean < .5)
            predicted += actual_outcome == (prob_mean >= .5)

        eta = datetime.fromtimestamp(time.time() + (time.time() - start_time) / (i + 1) * n_splits)
        print(f'ETA: {eta.strftime("%H:%M:%S")}', end=' | ')
        print(f"LOO-CV score: {predicted / tot_test_len * 100:3.2f}%", end=' | ')
        print(f"Predicted: {predicted:3d} / {tot_test_len:3d}", end=' | ')
        print(f"False positives: {false_pos} | False negatives: {false_neg}", end='\r', flush=True)

    print(f"\nTotal time: {int(time.time() - start_time)} sec")


def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']
    #model = build(samples, outcomes)
    #fit = sample(model)
    #plot_draws(fit, samples)
    #loo_cv(samples, outcomes)
    k_fold_cv(samples, outcomes)
    #loo_within_sample(fit, outcomes)
    #psis_loo_summary(fit)
    #summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta', 'sigma'])
    #display(summary)


if __name__ == '__main__':
    main()
