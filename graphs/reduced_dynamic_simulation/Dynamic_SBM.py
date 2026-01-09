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
from Functions_lib.Reduction_matrix_generation import get_reduction_matrix

# Dynamique N neurones avec A par bloc

DART = True
sigmas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
sigmas = [1, 3, 4, 5, 7, 9, 11, 13, 15]
# sigmas = sigmas[::2]
# sigmas = [3]
mean = 1

size1 = 20
size2 = 15
size3 = 10
size = size1+size2+size3
c_eta_dict = c_n_dict_bulder(31)

# R_map = np.empty((len(sigmas), 6, mean))
# reduced_map = np.zeros((len(sigmas), 6, mean))
R_map = np.empty((len(sigmas), 8, mean))
reduced_map = np.zeros((len(sigmas), 8, mean))
f_map = np.zeros((len(sigmas), 1))
for itm in range(mean):
    theta_0 = np.random.random((size, 1))*2*np.pi
    # theta_0 = np.random.normal(1, 0.5, (size, 1))
    # print(np.mean(theta_0), np.mean(theta_0[: size1]), np.mean(theta_0[-size2:]))
    # theta_0 = np.random.standard_normal((size, 1))

    # Graph SBM
    p1=0.6
    p2=0.9
    p3=0.8
    px12=0.2
    px13=0.05
    px23=0.1
    # G = nx.stochastic_block_model([size1, size2], [[p1, px12], [px12, p2]])
    G = nx.stochastic_block_model([size1, size2, size3], [[p1, px12, px13], [px12, p2, px23], [px13, px23, p3]])
    # G = nx.stochastic_block_model([size1, size2], [[1, 0], [0, 1]])
    A = np.array(nx.adjacency_matrix(G).toarray()).T
    A[:, -size3:] *= -1
    # A[size1:, :size1] = 0
    A[size1:size1+size2, :size1] = 0
    A[size1:size1+size2, -size3:] = 0
    # beta_0 = np.concatenate((np.random.normal(-2, 0.0001, (size1, 1)), np.random.normal(1, 0.0001, (size2, 1))), axis=0)
    beta_0 = np.concatenate((np.random.normal(-1.5, 0.0001, (size1, 1)), np.random.normal(1, 0.0001, (size2, 1)), np.random.normal(-0.5, 0.0001, (size3, 1))), axis=0)
    plt.imshow(A)
    plt.colorbar()
    plt.show()
    if DART:
        value, vector = nplin.eig(A)
        n=3
        # W = np.diag(beta_0.T[0])
        # valueW, vectorW = nplin.eig(W)
        # V_W = V_A_builder(np.real(valueW), np.real(vectorW), n)
        # print(valueW)
        # plt.imshow(vectorW)
        # plt.show()
        # plt.figure(figsize=(18, 12))
        # plt.imshow(np.pad(V_W, ((2, 2), (0, 0)), 'constant', constant_values=0))
        # plt.colorbar()
        # plt.show()
        V_A = V_A_builder(np.real(value), np.real(vector), n)
        # plt.figure(figsize=(18, 12))
        # plt.imshow(np.pad(V_A, ((2, 2), (0, 0)), 'constant', constant_values=0))
        # plt.colorbar()
        # plt.show()
        V_none = np.zeros(np.shape(V_A))
        V_T1, V_T2, V_T3 = V_A, V_none, V_none
        # M = np.array([np.pad(np.ones(150), (0, 100), 'constant', constant_values=0)/150, np.pad(np.ones(100), (150, 0), 'constant', constant_values=0)/100])
        M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = get_reduction_matrix(V_T1, V_T2, V_T3, number_initializations=10000)
        # M = np.eye(55)
        # M = msave+0
        M[M<1e-5/size] = 0
        M = M/np.sum(M, axis=1, keepdims=True)
        
        clearM=M+0
        clearM[clearM<1/size]=0
        clearM = clearM/np.sum(clearM, axis=1, keepdims=True)
        m1 = m_builder_weighted2(clearM[:, :size1])
        m2 = m_builder_weighted2(clearM[:, size1:size2+size1])
        idx1 = np.where(m1-m2==np.max(m1-m2))[0][0]
        idx2 = np.where(m2-m1==np.max(m2-m1))[0][0]
        if idx1 == 1 and idx2 == 0:
            M[:2] = M[1::-1]
        elif idx1 == 0 and idx2 == 1:
            pass
        else:
            if idx2 == 0:
                M[[1, 0]] = M[[0, 1]]
                M[[0, idx1]] = M[[idx1, 0]]
            else:
                M[[0, idx1]] = M[[idx1, 0]]
                M[[1, idx2]] = M[[idx2, 1]]

        # plt.figure(figsize=(18, 12))
        # plt.imshow(np.pad(M, ((2, 2), (0, 0)), 'constant', constant_values=0))
        # plt.colorbar()
        # plt.show()
        clearM=M+0
        clearM[clearM<1/size]=0
        clearM = clearM/np.sum(clearM, axis=1, keepdims=True)
        m = m_builder_weighted2(clearM)
        m1 = m_builder_weighted2(clearM[:, :size1])
        m2 = m_builder_weighted2(clearM[:, size1:size2+size1])
        m3 = m_builder_weighted2(clearM[:, size1+size2:])
        print(m, m1, m2, m3)
        if np.count_nonzero(clearM[0, -size2:])>np.count_nonzero(clearM[1, -size2:]):
            print(np.count_nonzero(M, axis=1), np.count_nonzero(M[:, :size1], axis=1), np.count_nonzero(M[:, -size2:], axis=1))
            plt.figure(figsize=(18, 12))
            plt.imshow(np.pad(M, ((2, 2), (0, 0)), 'constant', constant_values=0))
            plt.colorbar()
            plt.show()
            raise ValueError('wrong config pour M')
        z_0 = np.exp(1j*theta_0)
        Z_0, kappa, Omega, K_curl, W_curl, A_curl, W, K = reduced_dynamic_params_init(M, beta_0, A, z_0, size)
        # print('kappa=', kappa.T, '\n', 'Omega=', Omega.T, '\n', 'K_curl=', K_curl, '\n', 'W_curl=', W_curl, '\n', 'A_curl=', A_curl)
    for i, sigma in enumerate(sigmas):
        s = 5
        time_start = 0
        time_stop = 30

        sol  = solve_ivp(theta_model_scipy_solver, [time_start, time_stop], theta_0.T[0], max_step=0.1, atol = 0, rtol = 1e-10, args = (betaNneurones, A, s, beta_0, sigma))
        theta, time_list = sol.y, sol.t

        synchro = indice_syncronisation(theta)
        synchro35 = indice_syncronisation(theta[: size1])
        synchro25 = indice_syncronisation(theta[size1:size2+size1])
        synchro3 = indice_syncronisation(theta[size2+size1:])
        R, std = half_mean(synchro, time_list, True)
        R1, std1 = half_mean(synchro35, time_list, True)
        R2, std2 = half_mean(synchro25, time_list, True)
        R3, std3 = half_mean(synchro3, time_list, True)
        R_map[i, :, itm] = np.array([R, R1, R2, R3, std, std1, std2, std3])

        raster_plot_scipy, fire_list = raster_and_frate_builder(time_list, theta, time_stop)
        f_map[i, 0] += np.mean(fire_list[int(len(fire_list)/2):])/mean

        if DART:
            a_s = 2**s*(facto(s))**2/facto(2*s)

            sol  = solve_ivp(Z_reduced_dynamic_manager, [time_start, time_stop], Z_0, max_step=0.1, atol = 0, rtol = 1e-10, args = (n, sigma, kappa, Omega, W_curl, K_curl, A_curl, s, size, a_s, c_eta_dict))
            time_list_reduced, Zall = sol.t, sol.y
            # Z1, Z2 = Zall[0], Zall[1]
            Z1 = m1@Zall
            Z2 = m2@Zall
            Z3 = m3@Zall
            Z = m@Zall
            # Z = 0.6*Zall[0]+0.4*Zall[1]
            R1_reduced, Phi = np.abs(Z1), np.angle(Z1)
            R2_reduced, Phi = np.abs(Z2), np.angle(Z2)
            R3_reduced, Phi = np.abs(Z3), np.angle(Z3)
            R_reduced, Phi = np.abs(Z), np.angle(Z)

            Rmean, std_modif_reduced = half_mean(R_reduced, time_list_reduced, True)
            R1mean, std1_modif_reduced = half_mean(R1_reduced, time_list_reduced, True)
            R2mean, std2_modif_reduced = half_mean(R2_reduced, time_list_reduced, True)
            R3mean, std3_modif_reduced = half_mean(R3_reduced, time_list_reduced, True)
            reduced_map[i, :, itm] = np.array([Rmean, R1mean, R2mean, R3mean, std_modif_reduced, std1_modif_reduced, std2_modif_reduced, std3_modif_reduced])
mean_std = np.std(R_map[:, :4, :], axis = 2)
R_map = np.mean(R_map, axis = 2)
R_map[:, 4:] += mean_std/np.sqrt(mean)

if DART:
    mean_std_reduced = np.std(reduced_map[:, :4, :], axis = 2)
    reduced_map = np.mean(reduced_map, axis = 2)
    reduced_map[:, 4:] += mean_std_reduced/np.sqrt(mean)

if len(R_map) == 1 and mean == 1:
    raster_plot_scipy, fire_list = raster_and_frate_builder(time_list, theta, time_stop)
    print(R, std_modif_reduced)
    fig_sync_frate(time_list, theta, fire_list, neighbors=15)
    fig_sync(time_list, theta[: size1])
    fig_sync(time_list, theta[- size2:])
    # fig_sync(time_list, theta)
    # fig_frate(fire_list, 15)
    fig_spike_rep(time_list, theta, theta_0)
    fig_raster(raster_plot_scipy, time_stop)
    # fig_heatmap(theta)
    print(Rmean, std_modif_reduced)
    fig_sync_reduced(time_list_reduced, R_reduced)
    fig_sync_reduced(time_list_reduced, R1_reduced)
    fig_sync_reduced(time_list_reduced, R2_reduced)

else:
    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, R_map[:, 0], 'o-', lw=2, ms=2, markeredgecolor='red', c='black', label='Dynamique de phase complète')
    plt.fill_between(sigmas, R_map[:, 0]+R_map[:, 4], R_map[:, 0]-R_map[:, 4], alpha=0.3, color='grey')
    if DART:
        plt.plot(sigmas, reduced_map[:, 0], '--', lw=1, c='green', label='Dynamique réduite')#, label=f'<R>={R}'
        plt.fill_between(sigmas, reduced_map[:, 0]+reduced_map[:, 4], reduced_map[:, 0]-reduced_map[:, 4], alpha=0.5, color='green')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r"<R> [-]")
    plt.title(fr"Synchronisation totale d'un graph SBM à 3 population")#avec P11={p1}, P22={p2}, P12=P21={px}
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.01)
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(sigmas, R_map[:, 1], 'o-', lw=2, ms=2, markeredgecolor='red', c='black', label='Dynamique de phase complète')
    plt.fill_between(sigmas, R_map[:, 1]+R_map[:, 5], R_map[:, 1]-R_map[:, 5], alpha=0.3, color='grey')
    if DART:
        plt.plot(sigmas, reduced_map[:, 1], '--', lw=1, c='blue', label='Dynamique réduite')#, label=f'<R>={R}'
        plt.fill_between(sigmas, reduced_map[:, 1]+reduced_map[:, 5], reduced_map[:, 1]-reduced_map[:, 5], alpha=0.5, color='blue')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r'< $R_1$ > [-]')
    plt.title(r"Synchronisation Partie 1 du graph SBM")
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.02)
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(sigmas, R_map[:, 2], 'o-', lw=2, ms=2, markeredgecolor='red', c='black', label='Dynamique de phase complète')
    plt.fill_between(sigmas, R_map[:, 2]+R_map[:, 6], R_map[:, 2]-R_map[:, 6], alpha=0.3, color='grey')
    if DART:
        plt.plot(sigmas, reduced_map[:, 2], '--', lw=1, c='orange', label='Dynamique réduite')#, label=f'<R>={R}'
        plt.fill_between(sigmas, reduced_map[:, 2]+reduced_map[:, 6], reduced_map[:, 2]-reduced_map[:, 6], alpha=0.5, color='orange')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r"< $R_2$ >")
    plt.title(r"Synchronisation partie 2 du graph SBM")
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.03)
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(sigmas, R_map[:, 3], 'o-', lw=2, ms=2, markeredgecolor='red', c='black', label='Dynamique de phase complète')
    plt.fill_between(sigmas, R_map[:, 3]+R_map[:, 7], R_map[:, 3]-R_map[:, 7], alpha=0.3, color='grey')
    if DART:
        plt.plot(sigmas, reduced_map[:, 3], '--', lw=1, c='orange', label='Dynamique réduite')#, label=f'<R>={R}'
        plt.fill_between(sigmas, reduced_map[:, 3]+reduced_map[:, 7], reduced_map[:, 3]-reduced_map[:, 7], alpha=0.5, color='orange')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r"< $R_3$ >")
    plt.title(r"Synchronisation partie 3 du graph SBM")
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.03)
    plt.legend()
    plt.show()

    fig_mean_frate(f_map, sigmas, np.array([s]), xname=r'$\sigma$ [-]', yname = r'$s$ [-]', colo_horizon=True)

    fig_mean_sync_heat_map(R_map[:, :1], sigmas, np.array([s]),xname=r'$\sigma$ [-]',  yname = r'$s$ [-]', colo_horizon=True)
