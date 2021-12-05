import contextlib
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from IPython.display import display
from stan.fit import Fit
from sklearn.model_selection import KFold

def build(N_train, N_test, features_number, X_train, y_train, X_test, verbose=False) -> stan.model.Model:
    with open('logistic_regression_accuracy.stan', 'r') as file:
        stan_code = file.read()

    stan_data = {'N_train': N_train, 
                 'N_test': N_test, 
                 'J': features_number,
                 'X_train': X_train, 
                 'y_train': y_train,
                 'X_test': X_test}

    return stan.build(stan_code, data=stan_data, random_seed=0)


def sample(model, verbose=False):
    return model.sample(num_chains=4, num_samples=500, num_warmup=200)


def jitter(array):
    return array + np.random.rand(*array.shape)


def psis_loo_summary(fit: Fit):
    loo = az.loo(fit, pointwise=True)
    display(loo)
    fig, ax = plt.subplots(figsize=(8, 3))
    az.plot_khat(loo, show_bins=True, ax=ax)
    ax.set_title('Loo Linear model')
    print(f"Mean Pareto K: {loo['pareto_k'].values.mean():.2f}")
    plt.show()


def get_disease_prob(fit):
    return fit['y_prob_pred'].mean(axis=1)

def cross_validation(samples: pd.DataFrame, outcomes: pd.DataFrame):
    X = samples.values
    no_of_features = samples.shape[1]
    y = outcomes.values
    kf = KFold(n_splits = 5, shuffle = True)    
    false_pos, false_neg, predicted = 0, 0, 0
    start_time = time.time()
    tot_test_len = 0

    for train_idx, test_idx in kf.split(X):
       # print("TRAIN: ", train_idx, "TEST: ", test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)
        N_test = len(y_test)
        
        model = build(N_train, N_test, no_of_features, X_train, y_train, X_test)
        fit = model.sample(num_chains=4, num_samples=200, num_warmup=200)

        prob = get_disease_prob(fit)
        tot_test_len += len(prob)
        print(N_test)
        print(prob.shape)
        print(y_test.shape)
        for i in range(len(prob)):
            false_pos += not y_test[i] and (prob[i] >= .5)
            false_neg += y_test[i] and (prob[i] < .5)
            predicted += y_test[i] == (prob[i] >= .5)
        
        print("---------------------------------------------------------------------------")
        psis_loo_summary(fit)
        summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
        display(summary)
        print("---------------------------------------------------------------------------")

    print(f"LOO score: {predicted / (tot_test_len) * 100:3.2f}%", end=' | ')
    print(f"Predicted: {predicted:3d} / {tot_test_len:3d}", end=' | ')
    print(f"False positives: {false_pos} | False negatives: {false_neg}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    return predicted

def main():
    plt.rcParams['figure.dpi'] = 200
    data = pd.read_csv('heart.csv')
    SS = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[col_to_scale] = SS.fit_transform(data[col_to_scale])

    samples = data[data.columns.difference(['target'])]
    #samples = data[['cp', 'trestbps', 'thalach', 'ca', 'oldpeak', 'thal']]
    outcomes = data['target']
    #model = build(samples, outcomes)
    #fit = sample(model)

    cross_validation(samples, outcomes)
    #psis_loo_summary(fit)
    #summary = az.summary(fit, round_to=3, hdi_prob=0.9, var_names=['alpha', 'beta'])
    #display(summary)


if __name__ == '__main__':
    main()