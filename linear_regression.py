import contextlib

import arviz as az
import numpy as np
import pandas as pd
import stan
from matplotlib import pyplot as plt
from stan.fit import Fit

import diagnostics
from utils import suppress_stdout_stderr


class BetaPriorType:
    Normal = 0
    DoubleExponential = 1
    Uniform = 2


def build(samples: pd.DataFrame, outcomes: pd.DataFrame, verbose=False,
          kw_priors=None, **kwargs) -> stan.model.Model:
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


def sample(model, verbose=False, **kwargs):
    defaults = dict(num_chains=4, num_samples=500, num_warmup=200)
    context = contextlib.nullcontext if verbose else suppress_stdout_stderr
    with context():
        return model.sample(**{**defaults, **kwargs})


def jitter(array):
    return array + np.random.rand(*array.shape)


def get_disease_prob(fit, _, data: pd.DataFrame, sample: pd.DataFrame):
    prob = fit['alpha']
    scale = (data.max() - data.min())
    sample_std = (sample - data.min()) / scale
    for i, param_name in enumerate(data.columns):
        beta_i = fit['beta'][i]
        x_i = sample_std[param_name]
        prob += beta_i * x_i

    return prob.mean()


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
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), sharey=True,
                             gridspec_kw=dict(left=0.05, right=0.98, bottom=0.04, top=0.96,
                                              wspace=0.1, hspace=0.3))
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_{disease}$")
    axes = axes.ravel()
    line_style = dict(color='C3', linewidth=2, alpha=0.8)
    probs = fit['probs'][:, :250]
    for i, param_name in enumerate(samples.columns):
        values = samples[param_name].values
        beta = np.broadcast_to(values, probs.T.shape).T
        axes[i].scatter(y=probs.ravel(), x=jitter(beta).ravel(), s=4, color='#00f', alpha=0.01)
        axes[i].set_title(param_name)

        xs = (values.min(), values.max() + 1)
        beta_mean = fit['beta'][i].mean()
        ys = (probs.mean() - beta_mean, probs.mean() + beta_mean)
        axes[i].plot(xs, ys, **line_style)
        axes[i].annotate(f"$\\beta = {beta_mean:.3f}$", xy=(0.05, 0.05),
                         xycoords='axes fraction', fontsize=14)

    axes[samples.shape[1]].legend(
        (
            plt.Line2D([], [], **line_style),
            plt.Line2D([], [], linestyle='', marker='o', markersize=10, color='#00f', alpha=0.8)
        ),
        (
            r'linear correlation $\alpha + x \cdot \beta$',
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
    plot_draws(fit, samples)
    diagnostics.convergence(fit, var_names=['alpha', 'beta', 'sigma'])
    diagnostics.k_fold_cv(build, get_disease_prob, samples, outcomes, 5)
    diagnostics.loo_within_sample(fit, outcomes)
    diagnostics.psis_loo_summary(fit, 'Linear')

    # fit with saving warmup steps
    fit = sample(model, num_samples=0, num_warmup=200, save_warmup=True)
    diagnostics.plot_chains(fit, samples)

    plt.show()
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta', 'sigma'])
    display(summary)


if __name__ == '__main__':
    main()
