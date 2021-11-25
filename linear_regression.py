import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from IPython.display import display


def build(samples: pd.DataFrame, outcomes: pd.DataFrame):
    with open('linear_regression.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N': samples.shape[0], 'J': samples.shape[1],
                 'x': samples.values, 'y': outcomes.values,
                 'prior_alpha_mu': 0, 'prior_alpha_sigma': 1000,
                 'prior_beta_mu': 0, 'prior_beta_sigma': 10}
    return stan.build(stan_code, data=stan_data, random_seed=0)


def jitter(array):
    return array + np.random.rand(*array.shape)


def plot_draws(fit, samples):
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), sharey=True,
                             gridspec_kw=dict(left=0.05, right=0.98, bottom=0.05, top=0.98,
                                              wspace=0.1, hspace=0.3))
    axes = axes.ravel()
    line_style = dict(color='C3', linewidth=2, alpha=0.8)
    for i, param_name in enumerate(samples.columns):
        probs = fit['probs'][:, :200]
        values = samples[param_name].values
        beta = np.broadcast_to(values, probs.T.shape).T
        axes[i].scatter(y=probs.ravel(), x=jitter(beta).ravel(), s=4, color='#00f', alpha=0.01)
        axes[i].set_title(param_name)
        axes[i].set_ylabel(r"$P_{disease}$")

        xs = (values.min(), values.max() + 1)
        alpha = fit['alpha'][i].mean()
        beta = fit['beta'][i].mean()
        ys = (alpha, alpha + beta)
        axes[i].plot(xs, ys, **line_style)
        axes[i].set_ybound((0, 1))
        axes[i].annotate(f"$\\alpha = {alpha:.3f}$\n$\\beta = {beta:.3f}$", xy=(0.7, 0.1),
                         xycoords='axes fraction', fontsize=14)

    axes[samples.shape[1]].legend(
        (
            plt.Line2D([], [], **line_style),
            plt.Line2D([], [], linestyle='', marker='o', markersize=10, color='#00f', alpha=0.8)
        ),
        (
            r'linear correlation $\alpha + x \times \beta$',
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
    fit = model.sample(num_chains=4, num_samples=200, num_warmup=200)
    plot_draws(fit, samples)
    sum = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta', 'sigma']) \
        [['mean', 'sd', 'hdi_5%', 'hdi_95%', 'r_hat']]
    display(sum)


if __name__ == '__main__':
    main()
