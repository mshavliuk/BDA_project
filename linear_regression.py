import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from IPython.display import display


def build(data: pd.DataFrame):
    outcomes = data['target']
    parameters = data[data.columns.difference(['target'])]
    with open('linear_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': parameters.shape[0], 'J': parameters.shape[1],
                 'x': parameters.values, 'y': outcomes.values,
                 'prior_alpha_mu': 0, 'prior_alpha_sigma': 1000,
                 'prior_beta_mu': 0, 'prior_beta_sigma': 10}
    return stan.build(stan_code, data=stan_data, random_seed=0)


def plot_chains(fit: stan.fit.Fit, data: pd.DataFrame):
    posterior = az.from_pystan(fit)['posterior']
    _, axes = plt.subplots(nrows=7, ncols=2, figsize=(14, 18),
                           gridspec_kw=dict(left=0.05, right=0.98, bottom=0.02, top=0.98))

    axes = axes.ravel()
    for param_i, param_name in enumerate(data.columns.difference(['target'])):
        axes[param_i].set_title(f"{param_name} beta (slope)")
        for chain_i in range(posterior.num_chains):
            axes[param_i].plot(posterior['beta'].data[chain_i, :, param_i], alpha=0.5)

    axes[-1].set_title(f"alpha")
    for chain_i in range(posterior.num_chains):
        axes[-1].plot(posterior['alpha'].data[chain_i, :], alpha=0.5)
    plt.show()


def jitter(array):
    return array + np.random.rand(*array.shape)


def plot_draws(fit, data):
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), sharey=True,
                             gridspec_kw=dict(left=0.05, right=0.98, bottom=0.05, top=0.98,
                                              wspace=0.1, hspace=0.3))
    axes = axes.ravel()
    for i, param_name in enumerate(data.columns.difference(['target'])):
        probs = fit['probs']
        values = data[param_name].values
        beta = np.broadcast_to(values, probs.T.shape).T
        axes[i].scatter(y=probs.ravel(), x=jitter(beta).ravel(), s=4, color='#00f', alpha=0.006)
        axes[i].set_title(param_name)
        axes[i].set_ylabel(r"$P_{disease}$")

        xs = (values.min(), values.max() + 1)
        alpha = fit['alpha'][i].mean()
        beta = fit['beta'][i].mean()
        ys = (alpha, alpha + beta)
        axes[i].plot(xs, ys, color='C3', linewidth=2, alpha=0.8)

        axes[i].set_ybound((0, 1))
    plt.show()


def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    model = build(data)
    fit = model.sample(num_chains=4, num_samples=100, num_warmup=200)
    # plot_chains(fit, data)
    plot_draws(fit, data)
    sum = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta', 'sigma']) \
        [['mean', 'sd', 'hdi_5%', 'hdi_95%', 'r_hat']]
    display(sum)


if __name__ == '__main__':
    main()
