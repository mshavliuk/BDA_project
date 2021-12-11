import arviz as az
import pandas as pd

import diagnostics
import linear_regression as lin
from utils import suppress_stdout_stderr


def double_exp_prior(samples, outcomes, **kwargs):
    return lin.build(samples, outcomes,
                     kw_priors={'beta_prior_type': lin.BetaPriorType.DoubleExponential})


def uniform_prior(samples, outcomes, **kwargs):
    return lin.build(samples, outcomes,
                     kw_priors={'beta_prior_type': lin.BetaPriorType.Uniform})


def normal_1_prior(samples, outcomes, **kwargs):
    return lin.build(samples, outcomes,
                     kw_priors={'prior_beta_mu': [0], 'prior_beta_sigma': [1],
                                'beta_prior_type': lin.BetaPriorType.Normal})


def normal_10_prior(samples, outcomes, **kwargs):
    return lin.build(samples, outcomes,
                     kw_priors={'prior_beta_mu': [0], 'prior_beta_sigma': [10],
                                'beta_prior_type': lin.BetaPriorType.Normal})


def normal_100_prior(samples, outcomes, **kwargs):
    return lin.build(samples, outcomes,
                     kw_priors={'prior_beta_mu': [0], 'prior_beta_sigma': [100],
                                'beta_prior_type': lin.BetaPriorType.Normal})


def test_model(model_builder, samples, outcomes):
    fit = lin.sample(model_builder(samples, outcomes))
    loo_ws = diagnostics.loo_within_sample(fit, outcomes)
    loo_kf = diagnostics.k_fold_cv(model_builder, lin.get_disease_prob, samples, outcomes, 5)
    loo_psis = az.loo(fit, pointwise=True)['loo']
    mean_alpha = fit['alpha'].mean()
    mean_betas = fit['beta'].mean(axis=1)

    return [
        f"{loo_ws / len(outcomes) * 100:.2f}%",
        f"{loo_kf / len(outcomes) * 100:.2f}%",
        loo_psis,
        mean_alpha,
        *mean_betas
    ]


def test_priors():
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']
    parameter_columns = ['alpha'] + [f'$\\beta_{{{i}}}$' for i in range(len(samples.columns))]

    summary = pd.DataFrame(
        columns=['within sample accuracy', 'k-fold accuracy', 'PSIS LOO ELPD', *parameter_columns])

    with suppress_stdout_stderr():
        results = test_model(double_exp_prior, samples, outcomes)
        summary.loc['double_exponential(0, 1)'] = results

        results = test_model(uniform_prior, samples, outcomes)
        summary.loc[r'uniform($-\infty, $\infty)'] = results

        results = test_model(normal_1_prior, samples, outcomes)
        summary.loc[r'normal(0, 1)'] = results
        results = test_model(normal_10_prior, samples, outcomes)
        summary.loc[r'normal(0, 10)'] = results
        results = test_model(normal_100_prior, samples, outcomes)
        summary.loc[r'normal(0, 100)'] = results

    return summary


if __name__ == '__main__':
    test_priors()
