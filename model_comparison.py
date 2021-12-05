import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

import linear_regression as lin
import logistic_regression as log
from utils import suppress_stdout_stderr


def main():
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']

    with suppress_stdout_stderr():
        model_lin = lin.build(samples[['cp', 'trestbps', 'thalach', 'ca', 'oldpeak']], outcomes)
        fit_lin = model_lin.sample(num_chains=4, num_samples=500, num_warmup=200)

        model_log = log.build(samples[['cp', 'trestbps', 'thalach', 'ca', 'oldpeak']], outcomes)
        fit_log = model_log.sample(num_chains=4, num_samples=500, num_warmup=200)

    inference_lin = az.convert_to_inference_data(fit_lin, log_likelihood='log_lik')
    inference_log = az.convert_to_inference_data(fit_log, log_likelihood='log_lik')

    models = dict([
        ("linear_model", inference_lin),
        ("logistic_model", inference_log)
    ])
    comparison = az.compare(models, ic='loo')
    display(comparison[['loo', 'se', 'd_loo']])

    simulation_size = 10000
    fig, ax = plt.subplots(figsize=(8, 3))
    rs = np.random.RandomState(0)

    ax.set_yticks(())
    ax.set_title('Models elpd loo')

    draws_lin = stats.norm(comparison['loo']['linear_model'], comparison['se']['linear_model']) \
        .rvs(size=simulation_size, random_state=rs)
    ax.hist(draws_lin, bins=30, density=True, color='#00f', alpha=0.25, label="linear")

    draws_log = stats.norm(comparison['loo']['logistic_model'], comparison['se']['logistic_model']) \
        .rvs(size=simulation_size, random_state=rs)
    ax.hist(draws_log, bins=30, density=True, color='#f00', alpha=0.25, label="logistic")

    ax.legend()

    fig, ax = plt.subplots(figsize=(8, 3))
    diff = draws_lin - draws_log
    ax.hist(diff, bins=30, density=True)
    ax.set_yticks(())
    ax.set_title('Linear vs Logistic difference')

    prob = sum(diff > 0) / simulation_size

    plt.show()
    print(f"Linear model has higher elpd_loo than Logistic model with {prob * 100:.2f}% chance")


if __name__ == '__main__':
    main()
