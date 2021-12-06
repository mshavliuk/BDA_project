import pandas as pd

import linear_regression as lin


def main():
    data = pd.read_csv('heart.csv')
    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']

    lin.build(samples, outcomes,
              kw_priors={'beta_prior_type': lin.BetaPriorType.DoubleExponential})

    lin.build(samples, outcomes,
              kw_priors={'beta_prior_type': lin.BetaPriorType.Uniform})

    lin.build(samples, outcomes,
              kw_priors={'prior_beta_mu': [0], 'prior_beta_sigma': [1],
                         'beta_prior_type': lin.BetaPriorType.Normal})
    lin.build(samples, outcomes,
              kw_priors={'prior_beta_mu': [0], 'prior_beta_sigma': [10],
                         'beta_prior_type': lin.BetaPriorType.Normal})
    lin.build(samples, outcomes,
              kw_priors={'prior_beta_mu': [0], 'prior_beta_sigma': [100],
                         'beta_prior_type': lin.BetaPriorType.Normal})


if __name__ == '__main__':
    main()
