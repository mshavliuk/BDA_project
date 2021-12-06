import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler

import logistic_regression as logit
from diagnostics import psis_loo_summary, loo_within_sample


def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    SS = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[col_to_scale] = SS.fit_transform(data[col_to_scale])

    samples = data[data.columns.difference(['target'])]
    outcomes = data['target']
    model = logit.build(samples, outcomes)
    fit = logit.sample(model)
    loo_within_sample(fit, outcomes)
    psis_loo_summary(fit, 'Logistic')
    summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    display(summary)


if __name__ == '__main__':
    main()
