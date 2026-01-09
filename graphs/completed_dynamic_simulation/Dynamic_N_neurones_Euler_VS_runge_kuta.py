import numpy as np
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Functions_lib.Complete_dynamics import *
from Functions_lib.graph_setting import *

# Dynamique N neurone theta.
N = 100
euler = 0
scipy = 1
mean = 1


kappas = [0, 1, 3, 5, 8, 11, 15]
# kappas = [10]

# n_list = [1, 5, 10, 15, 30]
n_list = [1, 3, 5, 8, 12]
# n_list = [1]

# beta_mean = [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]
# beta_mean = [-1]

# beta_sigma = [0, 0.02, 0.05, 0.1, 0.3, 0.5]
# beta_sigma = [0]

# theta_sigma = [0.001]
# theta_sigma = [0, 0.002, 0.01, 0.05, 0.1, 0.5, 1]

params2 = n_list

# kappas=beta_sigma

f_map = np.zeros((len(kappas), len(params2)))
# f_map_norm = np.zeros((len(kappas), len(n_list)))
R_map = np.zeros((len(kappas), len(params2), 2, mean))
for m in range(mean):
    # theta_0 = np.random.normal(0, 0, (N, 1))
    # theta_0 = np.random.standard_normal((N, 1))
    theta_0 = np.random.random((N, 1))*2*np.pi

    A = np.ones((N, N))#-np.identity(N)
    # A = np.random.normal(0.3, 0.05, (N, N))
    for j, kappa in enumerate(kappas):
        beta_0 = np.random.normal(1, 0.01, (N, 1))
        # rng = np.random.default_rng()  # use Generator API
        # beta_0 = true_samples(rng.standard_cauchy((N, 1)), -0.5, bDelta)
        for i, n in enumerate(params2):
            # theta_0 = np.random.normal(0, tsigma, (N, 1))
            # theta_0 = np.random.random((N, 1))*2*np.pi

            # n = 10
            # kappa = 6

            # beta_0 = -0.5
            # rng = np.random.default_rng()  # use Generator API
            # beta_0 = true_samples(rng.standard_cauchy((N, 1)), -0.5, bdelta)

            # beta_0 = np.random.normal(bmean, bsigma, (N, 1))
            

            a_n = 2**n*(facto(n))**2/facto(2*n)
            step_approx = 0.01/(2*(0.00001+np.mean(beta_0) + a_n*kappa*2**n))
            
            time_start = 0
            time_step = np.abs(step_approx)
            time_stop = 10
            time_list = np.linspace(time_start, time_stop+time_step, int(time_stop/time_step)+1)


            if euler:
                theta, beta = dynamique_N_neurones(time_step, time_stop, theta_0, theta_model, betaNneurones, A, n, beta_0, kappa)
            
            elif scipy:
                sol  = solve_ivp(theta_model_scipy_solver, [time_start, time_stop], theta_0.T[0], first_step=1e-10, max_step=0.1, atol = 0, rtol = 1e-10, args = (betaNneurones, A, n, beta_0, kappa))
                time_list, theta = sol.t, sol.y
            
            synchro = indice_syncronisation(theta)
            R, std_modif = half_mean(synchro, time_list, True)
            # print(R, std_modif)
            R_map[j, i, :, m] = np.array([R, std_modif])

            raster_plot_scipy, fire_list = raster_and_frate_builder(time_list, theta, time_stop)
            f_map[j, i] += np.mean(fire_list[int(len(fire_list)/2):])/mean
            # f_map_norm[j, i] += half_mean(fire_list_norm)/mean
mean_std = np.std(R_map[:, :, 0, :], axis = 2)
R_map = np.mean(R_map, axis = 3)
R_map[:, :, 1] += mean_std/np.sqrt(mean)

if len(R_map) == 1 and mean == 1:
    if euler:
        fig_sync(time_list, theta)
        # fig_spike_rep(time_list, theta, theta_0)
        spike_rep_euler = spike_rep(theta, 30)/2**30
        raster_plot_array = []
        for neuron in spike_rep_euler:
            raster_plot_array += [time_list[find_peaks(neuron, 0.5)[0]]]
        fig_raster(raster_plot_array, time_stop)
        fig_heatmap(theta)

    # Scipy version
    if scipy:
        raster_plot_scipy, fire_list = raster_and_frate_builder(time_list, theta, time_stop)
        fig_sync_frate(time_list, theta, fire_list, neighbors=15)
        fig_sync(time_list, theta)
        # fig_frate(fire_list, 15)
        # fig_spike_rep(time_list, theta, theta_0)
        # fig_raster(raster_plot_scipy, time_stop)
        # fig_heatmap(theta)

else:
    fig_mean_sync_all(params2, R_map, kappas, labeltxt=r'$\kappa$', axis = 0, 
                    xlabel=r"n [-]", 
                    title=fr"Synchronisation moyenne pour différents $\kappa$ selon n.")#r'$\kappa$'
    
    fig_mean_sync_all(kappas, R_map, params2, labeltxt=r'n', 
                    xlabel=r'$\kappa$ [-]', 
                    title=fr"Synchronisation moyenne pour différents n selon $\kappa$.")
    
    fig_mean_frate(f_map, kappas, params2, xname=r'$\kappa [-]$', yname = r'n [-]')
    # fig_mean_frate(f_map_norm, kappas, n_list, title='Taux de décharge finale moyen normalisé.')

    fig_mean_sync_heat_map(R_map[:, :, 0], kappas, params2,xname=r'$\kappa$ [-]',  yname = r'n [-]')
    
    fig_3D(kappas, params2, R_map)
