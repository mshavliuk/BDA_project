import time

import arviz as az
import pandas as pd
from IPython.core.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from stan.fit import Fit

from logistic_regression import build_for_accuracy_check, get_disease_prob
from utils import suppress_stdout_stderr


def psis_loo_summary(fit: Fit, name: str):
    loo = az.loo(fit, pointwise=True)
    display(loo)
    fig, ax = plt.subplots(figsize=(8, 3))
    az.plot_khat(loo, show_bins=True, ax=ax)
    ax.set_title(f'Loo {name} model')
    print(f"Mean Pareto K: {loo['pareto_k'].values.mean():.2f}")
    plt.show()


def k_fold_cv(samples: pd.DataFrame, outcomes: pd.DataFrame, n_splits=5):
    X = samples.values
    y = outcomes.values
    kf = KFold(n_splits=n_splits, shuffle=True)
    false_pos, false_neg, predicted = 0, 0, 0
    start_time = time.time()
    tested_num = 0

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        samples, test_samples = X[train_idx], X[test_idx]
        outcomes, test_outcomes = y[train_idx], y[test_idx]

        with suppress_stdout_stderr():
            model = build_for_accuracy_check(samples, outcomes,
                                             test_samples)
            fit = model.sample(num_chains=4, num_samples=200, num_warmup=200)

        tested_num += len(test_outcomes)
        probs = get_disease_prob(fit)
        for prob, actual_outcome in zip(probs, test_outcomes):
            false_pos += not actual_outcome and (prob >= .5)
            false_neg += actual_outcome and (prob < .5)
            predicted += actual_outcome == (prob >= .5)

        print(f"LOO score: {predicted / tested_num * 100:3.2f}%", end=' | ')
        print(f"Predicted: {predicted:3d} / {tested_num:3d}", end=' | ')
        print(f"False positives: {false_pos} | False negatives: {false_neg}", end='\r')

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time}")
    return predicted


def loo_within_sample(fit: Fit, outcomes: pd.DataFrame):
    false_pos, false_neg, predicted = 0, 0, 0

    for i, actual_outcome in enumerate(outcomes):
        prediction = fit['probs'][i].mean() >= .5
        false_pos += not actual_outcome and prediction
        false_neg += actual_outcome and not prediction
        predicted += actual_outcome == prediction

    print(f"LOO-WS score: {predicted / len(outcomes) * 100:3.2f}%", end=" | ")
    print(f"Predicted: {predicted:3d} / {len(outcomes):3d}", end=" | ")
    print(f"False positives: {false_pos} | False negatives: {false_neg}")
