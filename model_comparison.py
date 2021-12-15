import arviz as az
import pandas as pd

from IPython.core.display import display
import linear_regression as lin
import logistic_regression as logit
from utils import suppress_stdout_stderr


def compare_models():
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']

    with suppress_stdout_stderr():
        model_lin = lin.build(samples, outcomes)
        fit_lin = model_lin.sample(num_chains=4, num_samples=500, num_warmup=200)

        model_log = logit.build(samples, outcomes)
        fit_log = model_log.sample(num_chains=4, num_samples=500, num_warmup=200)

    inference_lin = az.from_pystan(fit_lin, log_likelihood='log_lik')
    inference_log = az.from_pystan(fit_log, log_likelihood='log_lik')

    models = dict([
        ("linear_model", inference_lin),
        ("logistic_model", inference_log)
    ])
    comparison = az.compare(models, ic='loo')

    return comparison


if __name__ == '__main__':
    display(compare_models())
