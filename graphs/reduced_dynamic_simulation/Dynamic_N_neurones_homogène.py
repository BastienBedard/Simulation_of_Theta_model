import numpy as np
import networkx as nx
import numpy.linalg as nplin
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Functions_lib.Complete_dynamics import *
from Functions_lib.Reduced_dynamics import *
from Functions_lib.graph_setting import *

# Dynamique N neurone theta comparé avec DART
N = 100
mean = 1

sigmas = [0, 1, 1.5, 2, 3, 5, 7, 10, 15]
# sigmas = [2, 3, 5, 10]
# sigmas = [3]

s_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
# s_list = [1, 2, 3, 5, 7, 10, 15, 20, 30]
# s_list = [3]

# omega_list = [-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
# omega_list = [ -0.7, -0.5, 0, 0.5, 1]

# sig_omega_list = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 1]
# sig_omega_list = [0.01, 0.3, 0.6, 1]
# sig_omega_list = [0.1]

params2 = s_list

# sigmas=omega_list

c_eta_dict = c_n_dict_bulder(51)

f_map = np.zeros((len(sigmas), len(params2)))
R_map = np.zeros((len(sigmas), len(params2), 2, mean))
reduced_map = np.zeros((len(sigmas), len(params2), 2, mean))
for m in range(mean):
    theta_0 = np.random.random((N, 1))*np.pi*2
    # theta_0 = np.random.standard_normal((N, 1))
    A = np.ones((N, N))
    beta_0 = np.random.normal(1, 0, (N, 1))
    for i, s in enumerate(params2):
        for j, sigma in enumerate(sigmas):
            
            # s = 2
            # sigma = 2
            a_s = 2**s*(facto(s))**2/facto(2*s)

            time_start = 0
            time_stop = 10

            sol  = solve_ivp(theta_model_scipy_solver, [time_start, time_stop], theta_0.T[0], first_step=1e-10, max_step=0.1, atol = 0, rtol = 1e-10, args = (betaNneurones, A, s, beta_0, sigma))
            time_list, theta = sol.t, sol.y
            
            synchro = indice_syncronisation(theta)
            R, std_modif = half_mean(synchro, time_list, True)
            R_map[j, i, :, m] = np.array([R, std_modif])

            raster_plot_scipy, fire_list = raster_and_frate_builder(time_list, theta, time_stop)
            f_map[j, i] += np.mean(fire_list[int(len(fire_list)/2):])/mean



            z_0 = np.exp(1j*theta_0)
            omega_mean = np.mean(beta_0)
            M = np.ones((1, N))/N
            Z_0 = M@z_0

            # sol  = solve_ivp(Z_reduced_dynamic_homogeneOG, [time_start, time_stop], Z_0[0], max_step=0.1, atol = 0, rtol = 1e-10, args = (omega_mean, sigma, s, a_s))
            # time_list, Z = sol.t, sol.y[0]
            # R, Phi = np.abs(Z), np.angle(Z)
            # Rm, std_modif = half_mean(R, time_list, True)
            # R_map[j, i, :, m] = np.array([Rm, std_modif])

            sol  = solve_ivp(Z_reduced_dynamic_homogene, [time_start, time_stop], Z_0[0], max_step=0.1, atol = 0, rtol = 1e-10, args = (omega_mean, sigma, s, a_s, c_eta_dict))
            time_list_reduced, Z = sol.t, sol.y[0]
            R_reduced, Phi = np.abs(Z), np.angle(Z)
            Rmean, std_modif_reduced = half_mean(R_reduced, time_list_reduced, True)
            reduced_map[j, i, :, m] = np.array([Rmean, std_modif_reduced])

mean_std = np.std(R_map[:, :, 0, :], axis = 2)
R_map = np.mean(R_map, axis = 3)
R_map[:, :, 1] += mean_std/np.sqrt(mean)

mean_std_reduced = np.std(reduced_map[:, :, 0, :], axis = 2)
reduced_map = np.mean(reduced_map, axis = 3)
reduced_map[:, :, 1] += mean_std_reduced/np.sqrt(mean)


if len(R_map) == 1 and mean == 1:
    raster_plot_scipy, fire_list = raster_and_frate_builder(time_list, theta, time_stop)
    print(R, std_modif)
    fig_sync_frate(time_list, theta, fire_list, neighbors=15)
    # fig_sync(time_list, theta)
    # fig_frate(fire_list, 15)
    # fig_spike_rep(time_list, theta, theta_0)
    # fig_raster(raster_plot_scipy, time_stop)
    # fig_heatmap(theta)
    print(Rmean, std_modif_reduced)
    fig_sync_reduced(time_list_reduced, R_reduced, )

else:
    fig_mean_sync_and_reduced(params2, R_map, reduced_map, sigmas, labeltxt=r'$\sigma$', axis=0, xlabel=r"$s$ [-]",
                              title=fr"Comparaison de la synchronisation du modèle Thêta et de sa dynamique réduite.")
    fig_mean_sync_and_reduced(sigmas, R_map, reduced_map, params2, labeltxt=r'$s$', xlabel=r"$\sigma$ [-]",
                              title=fr"Comparaison de la synchronisation du modèle Thêta et de sa dynamique réduite.")

    fig_mean_frate(f_map, sigmas, params2, xname=r'$\sigma$ [-]', yname = r'$s$ [-]', colo_horizon=False)

    fig_mean_sync_heat_map(R_map[:, :, 0], sigmas, params2,xname=r'$\sigma$ [-]',  yname = r'$s$ [-]', colo_horizon=False)



    fig_mean_sync_all(params2, R_map, sigmas, labeltxt=r'$\sigma$', axis = 0, 
                    xlabel=r"$s$ [-]", 
                    title=fr"Synchronisation moyenne pour différents $\sigma$ selon $s$.")#r'$\kappa$'
    
    fig_mean_sync_all(sigmas, R_map, params2, labeltxt=r'$s$', 
                    xlabel=r'$\sigma$ [-]', 
                    title=fr"Synchronisation moyenne pour différents $s$ selon $\sigma$.")
    

    fig_mean_sync_all(params2, reduced_map, sigmas, labeltxt=r'$\sigma$', axis = 0, 
                    xlabel=r"$s$ [-]", 
                    title=fr"Synchronisation moyenne pour différents $\sigma$ selon $s$. DART")#r'$\kappa$'
    
    fig_mean_sync_all(sigmas, reduced_map, params2, labeltxt=r'$s$', 
                    xlabel=r'$\sigma$ [-]', 
                    title=fr"Synchronisation moyenne pour différents $s$ selon $\sigma$. DART")
