import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SIGMA = 3

def dynamics(setting, T = 100):
    assert setting in ["static", "bounded", "unbounded-towards", "unbounded-away-slow", "unbounded-away-fast"]
    x = 1
    x_lst = []
    y_lst = []
    x_sq = 0
    xy = 0
    theta_lst = []
    theta_est_lst = []
    delta_lst = []
    theta = 0
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
            theta += np.random.normal(0, 1)
        elif setting == "unbounded-away-slow":
            theta = 1 + (8 * np.log(t + 2)) ** 0.5
        else:
            theta = 1 + 2 * (t + 1) ** 0.5
        if t > 0:
            ## Estimate \theta
            theta_est = xy / x_sq
            theta_est_lst.append(theta_est)
            delta = abs(1 - theta_est / theta)
            delta_lst.append(delta)
            theta_lst.append(theta)
        ## Compute dynamics
        if t == 0:
            x = 1
        else:
            x = -1 + theta_est
        y = theta * x + np.random.normal(0, SIGMA)
        x_sq += x ** 2
        xy += x * y
        x_lst.append(x)
        y_lst.append(y)
    return theta_lst, theta_est_lst, delta_lst

def sim_batch(setting, T = 100, n_sample = 100):
    theta_mat = np.zeros((n_sample, T))
    theta_est_mat = np.zeros((n_sample, T))
    delta_mat = np.zeros((n_sample, T))
    for n in tqdm(range(n_sample)):
        theta_lst, theta_est_lst, delta_lst = dynamics(setting, T = T)
        theta_mat[n,:] = theta_lst
        theta_est_mat[n,:] = theta_est_lst
        delta_mat[n,:] = delta_lst
    return theta_mat, theta_est_mat, delta_mat

setting_lst = ["static", "bounded", "unbounded-towards", "unbounded-away-slow", "unbounded-away-fast"]
for setting in setting_lst:
    print("Setting:", setting)
    theta_mat, theta_est_mat, delta_mat = sim_batch(setting, T = 10000, n_sample = 100)
    plt.hist(theta_est_mat[:,-1])
    plt.savefig(f"Plots/{setting}_hist.png")
    plt.clf()
    plt.close()

    for n in range(theta_est_mat.shape[0]):
        plt.plot(theta_est_mat[n,:], alpha = 0.5)
    plt.plot(theta_mat.mean(axis = 0), color = "black")
    plt.xlabel("t")
    plt.ylabel(r"$\hat{\theta}_t$")
    plt.savefig(f"Plots/{setting}_traj.png")
    plt.clf()
    plt.close()
