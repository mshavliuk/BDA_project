import arviz as az
import pandas as pd
from IPython.display import display

import linear_regression as lin
import logistic_regression as log
from utils import suppress_stdout_stderr


def main():
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']

    with suppress_stdout_stderr():
        model_lin = lin.build(samples, outcomes)
        fit_lin = model_lin.sample(num_chains=4, num_samples=500, num_warmup=200)

        model_log = log.build(samples, outcomes)
        fit_log = model_log.sample(num_chains=4, num_samples=500, num_warmup=200)

    inference_lin = az.convert_to_inference_data(fit_lin, log_likelihood='log_lik')
    inference_log = az.convert_to_inference_data(fit_log, log_likelihood='log_lik')

    models = dict([
        ("linear_model", inference_lin),
        ("logistic_model", inference_log)
    ])
    comparison = az.compare(models, ic='loo')
    display(comparison)


if __name__ == '__main__':
    main()
