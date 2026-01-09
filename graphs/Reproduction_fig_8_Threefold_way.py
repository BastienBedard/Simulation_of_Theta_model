import numpy as np
from scipy.integrate import solve_ivp
import networkx as nx
import numpy.linalg as nplin
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Functions_lib.Complete_dynamics import *
from Functions_lib.Reduced_dynamics import *
from Functions_lib.graph_setting import *
from Functions_lib.Reduction_matrix_generation import get_reduction_matrix

# test reproduction résultat Threefold way


graph = False
# sigmas = [10]
# sigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sigmas = [0, 2, 4, 4.5, 5, 5.5, 6, 7, 8, 10]
# sigmas = sigmas[::2]
mean = 20

size = 250
c_eta_dict = c_n_dict_bulder(31)

R_map = np.empty((len(sigmas), 6, mean))
reduced_map = np.zeros((len(sigmas), 6, mean))
RR = np.empty((len(sigmas), 2, mean))
for m in range(mean):
    # theta_0 = np.random.standard_normal((size, 1))
    theta_0 = np.random.random((size, 1))*2*np.pi

    # Graph bibarti
    # G = nx.bipartite.random_graph(int(size*3/5), int(size*2/5), 0.2, directed=True)
    # A = np.array(nx.adjacency_matrix(G).toarray())

    # Graph SBM
    G = nx.stochastic_block_model([150, 100], [[0.7, 0.2], [0.2, 0.5]])
    A = np.array(nx.adjacency_matrix(G).toarray()).T
    
    beta_0 = np.concatenate((np.random.normal(-1.1, 0.0001, (int(size*3/5), 1)), np.random.normal(-0.9, 0.0001, (int(size*2/5), 1))), axis=0)
    value, vector = nplin.eig(A)

    # plt.imshow(A)
    # plt.show()


    # norm1 = np.sum(vector[:, 0][:150])
    # norm2 = np.sum(vector[:, 1][150:])
    # M2 = np.array([np.pad(vector[:, 0][:150], (0, 100), 'constant', constant_values=0)/norm1, np.pad(vector[:, 1][150:], (150, 0), 'constant', constant_values=0)/norm2])
    
    # value, vector = nplin.eig(A)
    # V_A = np.real(vector).T[:2:-1] #np.array([vector[:, 0], vector[:, 1]], dtype = float)
    V_A = np.abs(V_A_builder(np.real(value), np.real(vector)))
    V_T2, V_T3 = np.zeros(np.shape(V_A)), np.zeros(np.shape(V_A))
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = get_reduction_matrix(V_A, V_T2, V_T3)

    # plt.figure(figsize=(18, 12))
    # plt.imshow(V_A)
    # plt.colorbar()
    # plt.show()
    
    # if M[0, 0] < M[0, -1]:
    #     M[:] = M[::-1]
    #     print('wut')
    # print(M[0, :5])
    # print(M[0, -5:])
    # print(M[1, :5])
    # print(M[1, -5:])
    # print(np.max(M), np.max(M2), np.max(np.abs(M-M2)))
    
    # M = np.array([np.pad(np.ones(150), (0, 100), 'constant', constant_values=0)/150, np.pad(np.ones(100), (150, 0), 'constant', constant_values=0)/100])
    # M = np.ones((1, size))/size
    for i, sigma in enumerate(sigmas):
        s = 1
        time_start = 0
        time_stop = 50

        sol  = solve_ivp(theta_model_scipy_solver, [time_start, time_stop], theta_0.T[0], max_step=0.1, atol = 0, rtol = 1e-10, args = (betaNneurones, A, s, beta_0, sigma))
        theta, time_list = sol.y, sol.t

        synchro = indice_syncronisation(theta)
        synchro35 = indice_syncronisation(theta[: int(size*3/5)])
        synchro25 = indice_syncronisation(theta[- int(size*2/5)+1:])
        R, std = half_mean(synchro, time_list, True)
        R1, std1 = half_mean(synchro35, time_list, True)
        R2, std2 = half_mean(synchro25, time_list, True)
        R_map[i, :, m] = np.array([R, R1, R2, std, std1, std2])



        z_0 = np.exp(1j*theta_0)
        Z_0, kappa, Omega, K_curl, W_curl, A_curl, W, K = reduced_dynamic_params_init(M, beta_0, A, z_0, size)
        a_s = 2**s*(facto(s))**2/facto(2*s)

        # n=1
        # sol  = solve_ivp(Z_reduced_dynamic_manager, [time_start, time_stop], Z_0, max_step=0.1, atol = 0, rtol = 1e-10, args = (n, sigma, kappa, Omega, W_curl, K_curl, A_curl, s, size, a_s))
        # time_list_reduced, Z= sol.t, sol.y[0]
        # R_reduced, Phi = np.abs(Z), np.angle(Z)
        # Rmean, std_modif_reduced = half_mean(R_reduced, time_list_reduced, True)
        # RR[i, :, m] = np.array([Rmean, std_modif_reduced])
        
        n=2
        sol  = solve_ivp(Z_reduced_dynamic_manager, [time_start, time_stop], Z_0, max_step=0.1, atol = 0, rtol = 1e-10, args = (n, sigma, kappa, Omega, W_curl, K_curl, A_curl, s, size, a_s, c_eta_dict))
        time_list_reduced, Z1, Z2 = sol.t, sol.y[0], sol.y[1]
        Z=Z1*3/5+Z2*2/5
        R1_reduced, Phi = np.abs(Z1), np.angle(Z1)
        R2_reduced, Phi = np.abs(Z2), np.angle(Z2)
        R_reduced, Phi = np.abs(Z), np.angle(Z)

        Rmean, std_modif_reduced = half_mean(R_reduced, time_list_reduced, True)
        R1mean, std1_modif_reduced = half_mean(R1_reduced, time_list_reduced, True)
        R2mean, std2_modif_reduced = half_mean(R2_reduced, time_list_reduced, True)
        reduced_map[i, :, m] = np.array([Rmean, R1mean, R2mean, std_modif_reduced, std1_modif_reduced, std2_modif_reduced])
mean_std = np.std(R_map[:, :3, :], axis = 2)
R_map = np.mean(R_map, axis = 2)
R_map[:, 3:] += mean_std/np.sqrt(mean)

mean_std_reduced = np.std(reduced_map[:, :3, :], axis = 2)
reduced_map = np.mean(reduced_map, axis = 2)
reduced_map[:, 3:] += mean_std_reduced/np.sqrt(mean)

# mean_std_RR = np.std(RR[:, 0, :], axis = 1)
# RR = np.mean(RR, axis = 2)
# RR[:, 1] += mean_std_RR/np.sqrt(mean)

if not graph:
    plt.figure(figsize=(6.5, 5))
    plt.plot(sigmas, R_map[:, 0], lw=2, c='black')
    plt.fill_between(sigmas, R_map[:, 0]+R_map[:, 3], R_map[:, 0]-R_map[:, 3], alpha=0.5, color='grey')
    plt.plot(sigmas, reduced_map[:, 0], lw=2, c='green')#, label=f'<R>={R}'
    plt.fill_between(sigmas, reduced_map[:, 0]+reduced_map[:, 3], reduced_map[:, 0]-reduced_map[:, 3], alpha=0.5, color='green')
    # plt.plot(sigmas, RR[:, 0], lw=2, c='red')#, label=f'<R>={R}'
    # plt.fill_between(sigmas, RR[:, 0]+RR[:, 1], RR[:, 0]-RR[:, 1], alpha=0.5, color='red')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r"<R> [-]")
    plt.title(r"Synchronisation totale du graph bipartie pour différents $\sigma$")
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.01)
    plt.show()

    plt.figure(figsize=(3.5, 5))
    plt.plot(sigmas, R_map[:, 1], lw=2, c='black')
    plt.fill_between(sigmas, R_map[:, 1]+R_map[:, 4], R_map[:, 1]-R_map[:, 4], alpha=0.5, color='grey')
    plt.plot(sigmas, reduced_map[:, 1], lw=2, c='blue')#, label=f'<R>={R}'
    plt.fill_between(sigmas, reduced_map[:, 1]+reduced_map[:, 4], reduced_map[:, 1]-reduced_map[:, 4], alpha=0.5, color='blue')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r'< $R_1$ > [-]')
    plt.title(r"Synchronisation Partie 1 du graph bipartie")
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.02)
    plt.show()

    plt.figure(figsize=(6.5, 5))
    plt.plot(sigmas, R_map[:, 2], lw=2, c='black')
    plt.fill_between(sigmas, R_map[:, 2]+R_map[:, 5], R_map[:, 2]-R_map[:, 5], alpha=0.5, color='grey')
    plt.plot(sigmas, reduced_map[:, 2], lw=2, c='orange')#, label=f'<R>={R}'
    plt.fill_between(sigmas, reduced_map[:, 2]+reduced_map[:, 5], reduced_map[:, 2]-reduced_map[:, 5], alpha=0.5, color='orange')
    plt.xlabel(r"$\sigma$ [-]")
    plt.ylabel(r"< $R_2$ >")
    plt.title(r"Synchronisation partie 2 du graph bipartie")
    plt.tick_params(direction = 'in')
    plt.ylim(0, 1.03)
    plt.show()

if graph:
    plt.figure(figsize=(10, 6))
    plt.plot(time_list, synchro, label=f'<R>={R}')#, 'ro', ms=1
    plt.plot(time_list_reduced, R_reduced, label=f'<R>={Rmean}')#, 'ro', ms=1
    plt.xlabel(r"Temps [s]")
    plt.ylabel(r"<R> [-]")
    plt.title(r"Évolution de la synchronisation au fils du temps")
    plt.tick_params(direction = 'in')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time_list, synchro35, label=fr'< R$>_1$={R1}')#, 'ro', ms=1
    plt.xlabel(r"Temps [s]")
    plt.ylabel(r"< R$>_1$ [-]")
    plt.title(r"Évolution de la synchronisation au fils du temps (groupe 1)")
    plt.tick_params(direction = 'in')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time_list, synchro25, label=fr'< R$>_2$={R2}')#, 'ro', ms=1
    plt.plot(time_list_reduced, R2_reduced, label=fr'< R$>_2$={R2mean}')#, 'ro', ms=1
    plt.xlabel(r"Temps [s]")
    plt.ylabel(r"< R$>_2$ [-]")
    plt.title(r"Évolution de la synchronisation au fils du temps (groupe 2)")
    plt.tick_params(direction = 'in')
    plt.grid(True)
    plt.legend()
    plt.show()


    raster_plot_array = []

    plt.figure(figsize=(10, 10))
    for i in range(len(theta_0)):
        spike_rep_list = spike_rep(theta[i], 30)/2**30
        plt.plot(time_list, i+spike_rep_list, label = fr'$\theta_0={theta_0[i]}$')
        raster_plot_array += [time_list[find_peaks(spike_rep_list, 0.5)[0]]]
    plt.xlabel(r"Temps [s]")
    plt.ylabel(r"valeur de (1-cos$(\theta ))^{30}$ normalisé [-]")
    plt.title("Représentation neuronal de theta pour explicité les pics de potentiel.")
    plt.tick_params(direction = 'in')
    plt.grid(True)
    # plt.legend()
    plt.show()

    plt.figure(figsize=(15, 4))
    output = spike_rep(theta, 30)/2**30
    V_spike_mat = np.repeat(output, 10, axis=0)
    plt.imshow(V_spike_mat)
    plt.colorbar()
    time_step = 0.1
    plt.xlabel(fr"Temps [{time_step}s]")
    plt.ylabel(r"Énumération des neurones. [-]")
    plt.title(r"Heat map d'émission de potentiel des neurones.")
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.eventplot(raster_plot_array[::-1])
    plt.xlabel(fr"Temps [s]")
    plt.ylabel(r"Énumération des neurones. [-]")
    plt.title(r"Raster plot des pics d'émission de potentiel des neurones.")
    plt.grid(True)
    # plt.margins(0)
    plt.xlim(0, time_stop)
    plt.ylim(0)
    plt.show()