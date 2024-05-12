import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SIGMA = 3

def dynamics(setting, strategy = "None", model = "linear_gaussian", T = 100, perturbation = 0.1, xi_seq = None, theta_range = (2, 6), ucb_partition = 10):
    assert setting in ["static", "bounded", "unbounded-towards", "unbounded-away-slow", "unbounded-away-fast"]
    assert strategy in ["None", "forced-exploration-perturb", "forced-exploration-ucb", "forced-explore-bayesian", "robust-adaptive-ci"]
    assert model in ["linear_gaussian", "demand_learning"]
    if strategy != "None":
        assert setting == "static"
    if setting == "unbounded-towards":
        assert xi_seq is not None and T + 1 <= len(xi_seq)
    x = 1
    x_lst = []
    y_lst = []
    x_sq = 0
    xy = 0
    xay = 0
    a = 1
    K = np.exp(SIGMA ** 2 / 2)
    c = 0.75
    mu = (theta_range[0] + theta_range[1]) / 2
    tau = (mu - theta_range[0]) / 2
    theta_lst = []
    theta_est_lst = []
    delta_lst = []
    theta = 0
    ucb_arms = np.linspace(theta_range[0], theta_range[1], ucb_partition + 1)
    ucb_freq = np.zeros(ucb_partition)
    ucb_cum_profit = np.zeros(ucb_partition)
    ucb_conf = np.zeros(ucb_partition) + np.inf
    total_regret = 0
    total_regret_lst = []
    theta_lo, theta_hi = theta_range
    for t in range(T + 1):
        ## Compute \theta
        if setting == "static":
            theta = 2.5
        elif setting == "bounded":
            if t == 0:
                theta = 2.5
            else:
                if (t // 2000) % 2 == 0:
                    theta += 0.001
                else:
                    theta -= 0.001
        elif setting == "unbounded-towards":
            theta += xi_seq[t]
        elif setting == "unbounded-away-slow":
            theta = 1 + (8 * np.log(t + 2)) ** 0.5
        else: # setting == "unbounded-away-fast":
            theta = 1 + 2 * (t + 1) ** 0.5
        if t > 0:
            ## Estimate \theta
            if model == "linear_gaussian":
                theta_est = xy / x_sq
                if strategy == "forced-explore-bayesian":
                    tau_next_sq = 1 / (x_lst[-1] ** 2 / SIGMA ** 2 + 1 / tau ** 2)
                    mu = tau_next_sq * (x_lst[-1] * y_lst[-1] / SIGMA ** 2 + mu / tau ** 2)
                    tau = tau_next_sq ** 0.5
                    theta_est = mu
            elif model == "demand_learning":
                theta_est = max(theta_range[0], xay / x_sq)
                if strategy == "forced-exploration-ucb":
                    arm = np.argmax(ucb_conf)
                    ucb_lo = ucb_arms[arm]
                    ucb_hi = ucb_arms[arm + 1]
                    theta_est = np.random.uniform(low = ucb_lo, high = ucb_hi)
                elif strategy == "forced-explore-bayesian":
                    tau_next_sq = 1 / (x_lst[-1] ** 2 / SIGMA ** 2 + 1 / tau ** 2)
                    mu = tau_next_sq * (-x_lst[-1] * (y_lst[-1] - a) / SIGMA ** 2 + mu / tau ** 2)
                    tau = tau_next_sq ** 0.5
                    theta_est = mu
            theta_sd = SIGMA / ((x_sq / (t + 1)) ** 0.5)
            theta_coef = 1.96
            theta_lo = max(theta_range[0], theta_est - theta_coef * theta_sd)
            theta_hi = min(theta_range[1], theta_est + theta_coef * theta_sd)
            theta_est_lst.append(theta_est)
            delta = abs(1 - theta_est / theta)
            delta_lst.append(delta)
            theta_lst.append(theta)
            if strategy == "forced-explore-bayesian":
                theta_est = np.random.normal(mu, tau)
                theta_est = min(max(theta_est, theta_range[0]), theta_range[1])
        ## Compute dynamics
        if t == 0:
            if model == "linear_gaussian":
                x = 1
            elif model == "demand_learning":
                x = 1
        else:
            if model == "linear_gaussian":
                x = -1 + theta_est
                if strategy == "forced-exploration-perturb":
                    x += np.random.normal(0, perturbation)
                elif strategy == "forced-exploration-ucb":
                    pass
                elif strategy == "kalman-filter":
                    pass
            elif model == "demand_learning":
                x = np.log(c) - np.log(1 - 1/theta_est)
                x_opt = np.log(c) - np.log(1 - 1/theta)
                profit_opt = K * (np.exp(x_opt) - c) * np.exp(-theta * x_opt)
                if strategy == "forced-exploration-perturb":
                    x += np.random.normal(0, perturbation)
                elif strategy == "forced-exploration-ucb":
                    profit = K * (np.exp(x) - c) * np.exp(-theta * x)
                    ucb_freq[arm] += 1
                    ucb_cum_profit[arm] += profit
                    ucb_conf[arm] = ucb_cum_profit[arm] / ucb_freq[arm] + (2 * np.log(t + 1) / ucb_freq[arm]) ** 0.5
                elif strategy == "kalman-filter":
                    pass
                elif strategy == "robust-adaptive-ci":
                    x = np.log(c) - np.log(1 - 1/theta_hi)
                profit = K * (np.exp(x) - c) * np.exp(-theta * x)
                regret = profit_opt - profit
                total_regret += regret
                total_regret_lst.append(total_regret)
        if model == "linear_gaussian":
            y = theta * x + np.random.normal(0, SIGMA)
        elif model == "demand_learning":
            y = a - theta * x + np.random.normal(0, SIGMA)
        if t > 0:
            assert not np.isnan(theta_est)
        x_sq += x ** 2
        xy += x * y
        xay += x * (a - y)
        x_lst.append(x)
        y_lst.append(y)
    return theta_lst, theta_est_lst, delta_lst, total_regret_lst

def sim_batch(setting, strategy, model, T = 100, n_sample = 100):
    theta_mat = np.zeros((n_sample, T))
    theta_est_mat = np.zeros((n_sample, T))
    delta_mat = np.zeros((n_sample, T))
    total_regret_mat = np.zeros((n_sample, T))
    if setting == "unbounded-towards":
        np.random.seed(6230)
        xi_seq = np.random.normal(0, 1, size = T + 1)
    else:
        xi_seq = None
    for n in tqdm(range(n_sample)):
        theta_lst, theta_est_lst, delta_lst, total_regret_lst = dynamics(setting, strategy, model, T = T, xi_seq = xi_seq)
        theta_mat[n,:] = theta_lst
        theta_est_mat[n,:] = theta_est_lst
        delta_mat[n,:] = delta_lst
        if model == "demand_learning":
            total_regret_mat[n,:] = total_regret_lst
    return theta_mat, theta_est_mat, delta_mat, total_regret_mat

#setting_params_dict = {
#    "model": [
#        "demand_learning": [
#            ("static", "None"), ("static", "forced-exploration-perturb"), ("static", "forced-exploration-ucb"), ("static", "kalman-filter"),
#            ("bounded", "None"), ("unbounded-towards", "None"), ("unbounded-away-slow", "None"), ("unbounded-away-fast", "None")
#        ],
#        "linear_gaussian": [
#            ("static", "None"), ("static", "forced-exploration-perturb"), ("static", "forced-exploration-ucb"), ("static", "kalman-filter"),
#            ("bounded", "None"), ("unbounded-towards", "None"), ("unbounded-away-slow", "None"), ("unbounded-away-fast", "None")
#        ]
#    ]
#}

setting_params_dict = {
    "model": {
        "demand_learning": [
            ("static", "None"), ("static", "forced-exploration-perturb"), ("static", "forced-exploration-ucb"), ("static", "forced-explore-bayesian"), ("static", "robust-adaptive-ci"),
            ("bounded", "None"), #("unbounded-towards", "None"), ("unbounded-away-slow", "None"), ("unbounded-away-fast", "None")
        ],
        "linear_gaussian": [
            ("static", "None"), ("static", "forced-exploration-perturb"), ("static", "forced-explore-bayesian"),
            ("bounded", "None"), ("unbounded-towards", "None"), ("unbounded-away-slow", "None"), ("unbounded-away-fast", "None")
        ]
    }
}

for model in setting_params_dict["model"]:
    for setting, strategy in setting_params_dict["model"][model]:
        np.random.seed(6230)
        print("Model", model, "Setting:", setting, "Strategy:", strategy)
        theta_mat, theta_est_mat, delta_mat, total_regret_mat = sim_batch(setting, strategy = strategy, model = model, T = 10000, n_sample = 1000)
        plt.hist(theta_est_mat[:,-1])
        plt.xlim(left = 0)
        plt.savefig(f"Plots/{model}_{setting}_{strategy}_hist.png")
        plt.clf()
        plt.close()

        for n in range(theta_est_mat.shape[0]):
            plt.plot(theta_est_mat[n,:], alpha = 0.5)
        plt.plot(theta_mat.mean(axis = 0), color = "black")
        plt.xlabel("t")
        plt.ylabel(r"$\hat{\theta}_t$")
        plt.savefig(f"Plots/{model}_{setting}_{strategy}_traj.png")
        plt.clf()
        plt.close()
        
        if model == "demand_learning":
            for n in range(total_regret_mat.shape[0]):
                plt.plot(total_regret_mat[n,:], alpha = 0.5)
            plt.xlabel("t")
            plt.ylabel("Total Regret")
            plt.savefig(f"Plots/{model}_{setting}_{strategy}_regret.png")
            plt.clf()
            plt.close()
