import time
from datetime import datetime
from typing import Callable

import arviz as az
import pandas as pd
from IPython.core.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from stan.fit import Fit

from utils import suppress_stdout_stderr


def psis_loo_summary(fit: Fit, name: str):
    loo = az.loo(fit, pointwise=True)
    display(loo)
    fig, ax = plt.subplots(figsize=(8, 3))
    az.plot_khat(loo, show_bins=True, ax=ax)
    ax.set_title(f'Loo {name} model')
    print(f"Mean Pareto K: {loo['pareto_k'].values.mean():.2f}")


def k_fold_cv(model_builder: Callable, predictor: Callable, samples: pd.DataFrame,
              outcomes: pd.DataFrame, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    false_pos, false_neg, predicted = 0, 0, 0
    start_time = time.time()
    tested_num = 0

    for i, (fit_idx, _) in enumerate(kf.split(samples)):
        fit_query = samples.index.isin(fit_idx)
        fit_samples, test_samples = samples[fit_query], samples[~fit_query]
        fit_outcomes, test_outcomes = outcomes[fit_query], outcomes[~fit_query]
        with suppress_stdout_stderr():
            model = model_builder(samples=fit_samples, outcomes=fit_outcomes,
                                  test_samples=test_samples)
            fit = model.sample(num_chains=4, num_samples=500, num_warmup=200)

        num = len(test_outcomes)
        tested_num += num
        for idx, test_sample, actual_outcome in zip(range(num), test_samples.values, test_outcomes):
            prob = predictor(fit, idx, samples, test_sample)
            false_pos += not actual_outcome and (prob >= .5)
            false_neg += actual_outcome and (prob < .5)
            predicted += actual_outcome == (prob >= .5)

        eta = datetime.fromtimestamp(
            time.time() + (time.time() - start_time) / (i + 1) * (n_splits - i - 1))
        print(f'ETA: {eta.strftime("%H:%M:%S")}', end=' | ')
        print(f"LOO score: {predicted / tested_num * 100:3.2f}%", end=' | ')
        print(f"Predicted: {predicted:3d} / {tested_num:3d}", end=' | ')
        print(f"False positives: {false_pos} | False negatives: {false_neg}", end='\r')

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time}")
    return predicted


def loo_within_sample(fit: Fit, outcomes: pd.Series):
    false_pos, false_neg, predicted = 0, 0, 0

    for i, actual_outcome in enumerate(outcomes):
        prediction = fit['probs'][i].mean() >= .5
        false_pos += not actual_outcome and prediction
        false_neg += actual_outcome and not prediction
        predicted += actual_outcome == prediction

    print(f"LOO-WS score: {predicted / len(outcomes) * 100:3.2f}%", end=" | ")
    print(f"Predicted: {predicted:3d} / {len(outcomes):3d}", end=" | ")
    print(f"False positives: {false_pos} | False negatives: {false_neg}")
    return predicted


def convergence(fit: Fit, var_names):
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=var_names)
    display(summary[['mcse_mean', 'ess_bulk', 'r_hat']])

    stats = az.from_pystan(fit, log_likelihood='log_lik')['sample_stats']
    display(pd.DataFrame({'max_tree_depth': stats.tree_depth.values.max(),
                          'mean_tree_depth': stats.tree_depth.values.mean(),
                          'divergences_num': stats.diverging.values.sum()}, index=['stat']))


def plot_chains(fit: Fit, samples: pd.DataFrame):
    posterior = az.from_pystan(fit)['posterior']
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(16, 9),
                             gridspec_kw=dict(left=0.05, right=0.98, bottom=0.04, top=0.96,
                                              wspace=0.3, hspace=0.3))

    axes = axes.ravel()
    for param_i, param_name in enumerate(samples.columns):
        axes[param_i].set_title(f"$\\beta_{{{param_name}}}$")
        for chain_i in range(posterior.num_chains):
            axes[param_i].plot(posterior['beta'].data[chain_i, :100, param_i], alpha=0.5)

    axes[-2].set_axis_off()
    gs = axes[-1].get_gridspec()
    axes[-1].remove()
    axbig = fig.add_subplot(gs[-1:, -2:])
    axbig.set_title(r"$\alpha$")
    for chain_i in range(posterior.num_chains):
        axbig.plot(posterior['alpha'].data[chain_i, :], alpha=0.5)



def plot_posterior(fit):
    fig, axes = plt.subplots(3, 5, figsize=(16, 9),
                             gridspec_kw=dict(left=0.02, right=0.98, bottom=0.04, top=0.94,
                                              wspace=0.2, hspace=0.5))
    az.plot_posterior(fit, var_names=['beta', 'alpha'], hdi_prob=.9, round_to=2, ax=axes)
    axes.ravel()[-1].set_axis_off()
