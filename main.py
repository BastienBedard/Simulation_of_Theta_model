import numpy as np
import networkx as nx
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import factorial as facto
import time
rng = np.random.default_rng(seed=420)  # use Generator API



# Définition des fonctions

def theta_model(theta:np.ndarray, beta:np.ndarray):
    return 1-np.cos(theta)+np.multiply(1+np.cos(theta), beta)

def theta_model_scipy_solver(t, y:np.ndarray, beta, A:np.ndarray, n:int, beta_0:float, kappa:float):
    return 1-np.cos(y)+np.multiply(1+np.cos(y), beta(np.array([y]).T, A, n, beta_0, kappa).T[0])

def spike_rep(theta:np.ndarray, n = 2):
   return (1-np.cos(theta))**n

def beta_Vconst(beta:float, beta_switch:bool, beta_low:float = -0.01, beta_high:float = 0.01):
    if beta >= 0.2:
       beta_switch = False
    elif beta <= -0.3:
       beta_switch = True
    if beta_switch:
        return beta_switch, beta_high
    else:
        return beta_switch, beta_low

def beta_const(beta:float, beta_switch:bool):
   return False, 0

def dynamique_1_neurone(time_step:int, time_stop:int, theta_0:float, beta_0:float, theta_model, beta_model, *kwarg):
  theta = np.array([theta_0])
  beta = np.array([beta_0])
  beta_switch = True
  for _ in range(int(time_stop/time_step)):
    Vtheta = theta_model(theta[-1], beta[-1])
    beta_switch, Vbeta = beta_model(beta[-1], beta_switch, *kwarg)
    theta = np.concatenate((theta, theta[-1:]+Vtheta*time_step))
    beta = np.concatenate((beta, beta[-1:]+Vbeta*time_step))
  return theta, beta

def beta2neurones(theta:np.ndarray, n:int = 2, beta_0:float=0, kappa:float = 1): # point critique n:int = 1, seuil:float=-0.184151
   a_n = 2**n*(facto(n))**2/facto(2*n)
   # a_n = 1/2**(n+1)
   beta = spike_rep(theta, n)*a_n*kappa
   beta[[0, 1], :] = beta[[1, 0], :]
   return beta_0 + beta

def betaNneurones(theta_list:np.ndarray, A:np.ndarray, n:int = 2, beta_0:float=0, kappa:float = 1):
   N = len(theta_list)
   a_n = 2**n*(facto(n))**2/facto(2*n)
   if np.all(A == 1):
      beta1 = np.array([[np.sum(spike_rep(theta_list, n))]])
      beta = np.repeat(beta1, N, axis=0)
   else:
      beta = np.empty(np.shape(theta_list))
      for i in range(N):
         beta[i, 0] = np.sum(spike_rep(np.multiply(A[i], theta_list.T[0]), n))
   return beta_0 + a_n*kappa*beta/N

def dynamique_N_neurones(time_step:float, time_stop:int, theta_0:np.ndarray, theta_model, beta_model, *kwarg):
  N = len(theta_0)
  total_steps = int(time_stop/time_step)
  theta = np.empty((N, total_steps+1))
  beta = np.empty((N, total_steps+1))
  theta[:, 0:1] = theta_0
  beta[:, 0:1] = beta_model(theta_0, *kwarg)
  for step in range(1, total_steps+1):
   last_theta = theta[:, step-1:step]
   Vtheta = theta_model(last_theta, beta[:, step-1:step])
   new_theta = last_theta+Vtheta*time_step
   theta[:, step:step+1] = new_theta%(2*np.pi)
   beta[:, step:step+1] = beta_model(new_theta, *kwarg)
  return theta, beta

def indice_syncronisation(theta):
   return np.sqrt(np.sum(np.cos(theta), axis = 0)**2 + np.sum(np.sin(theta), axis = 0)**2)/len(theta)

def dist_lorentzienne(beta:np.ndarray, beta_0:float=0, gamma:float=0.1):
   return (gamma/np.pi)/((beta-beta_0)**2+gamma**2)

def true_samples(samples, beta_0, gamma):
   return beta_0+gamma*samples

def half_mean(data:np.ndarray, time_list:np.ndarray=False, std:bool=False):
   half_data = data[int(len(data)/2):]
   if isinstance(time_list, np.ndarray):
      midtime = (time_list[-1]-time_list[0])/2
      idx = np.abs(time_list - midtime).argmin()
      half_data = data[idx:]
   if std:
      return (np.mean(half_data), np.std(half_data))
   return np.mean(half_data)

# Fonctions d'affichage des figures
def fig_sync(time_list:np.ndarray, theta:np.ndarray, xlabel:str="Temps [s]", 
             ylabel:str="<R>", title:str="Évolution de la synchronisation au fils du temps"):
    plt.figure(figsize=(10, 6))
    plt.plot(time_list, indice_syncronisation(theta))#, 'ro', ms=1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.grid(True)
    plt.show()

def fig_heatmap(theta:np.ndarray, xlabel:str="pas de temps [-]", 
                ylabel:str="Énumération des neurones. [-]", 
                title:str="Heat map d'émission de potentiel des neurones."):
    plt.figure(figsize=(15, 4))
    if len(theta[0]) < 10000:
        rep = 1+int(len(theta[0])/(len(theta)*8))
        output = spike_rep(theta, 30)/2**30
        V_spike_mat = np.repeat(output, rep, axis=0)
    elif len(theta[0]) < 30000:
        rep = 1+int(len(theta[0])/(len(theta)*10))
        output = spike_rep(theta, 30)/2**30
        V_spike_mat = np.repeat(output, rep, axis=0)
    else:
        rep = int(30000/(len(theta)*20))
        output = spike_rep(theta[:, -30000:], 10)/2**30
        V_spike_mat = np.repeat(output, rep, axis=0)
    plt.imshow(V_spike_mat)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def fig_raster(raster_plot_array:np.ndarray, time_stop:int, xlabel:str="Temps [s]", 
                ylabel:str="Énumération des neurones. [-]", 
                title:str="Raster plot des pics d'émission de potentiel des neurones."):
    plt.figure(figsize=(12, 3))
    plt.eventplot(raster_plot_array[::-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, time_stop)
    plt.ylim(0)
    plt.show()

def fig_spike_rep(time_list:np.ndarray, spike_rep_list:np.ndarray, theta_0:np.ndarray, legend:bool=False, 
                  xlabel:str="Temps [s]", ylabel:str=r"valeur de (1-cos$(\theta ))^{30}$ normalisé [-]", 
                  title:str="Représentation neuronal de theta pour explicité les pics de potentiel."):
    plt.figure(figsize=(10, 10))
    for i in range(len(theta_0)):
        plt.plot(time_list, i+spike_rep_list[i], label = fr'$\theta_0={theta_0[i]}$')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.grid(True)
    if legend:
        plt.legend()
    plt.show()

def fig_mean_sync(kappas:list, R_list:np.ndarray, labeltxt:str,
                  xlabel:str=r"$\kappa$ [-]", ylabel:str=r"<R> [-]", 
                  title:str=r"Synchronisation totale d'un graph pour différents $\kappa$"):
    # plt.figure(figsize=(10, 5))
    plt.plot(kappas, R_list[:, 0], 'o-', lw=2, ms=3, markeredgecolor='red', label=labeltxt)
    plt.fill_between(kappas, R_list[:, 0]+R_list[:, 1], R_list[:, 0]-R_list[:, 1], alpha=0.5, color='grey')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.tick_params(direction = 'in')
    # plt.ylim(np.min(R_list[:, 0]-R_list[:, 1])-0.01, np.max(R_list[:, 0]+R_list[:, 1])+0.01)
    # plt.show()

def fig_mean_sync2(kappas:list, R_list:np.ndarray,
                  xlabel:str=r"$\kappa$ [-]", ylabel:str=r"<R> [-]", 
                  title:str=r"Synchronisation totale d'un graph pour différents $\kappa$"):
    plt.figure(figsize=(10, 5))
    plt.plot(kappas, R_list[:, 0], 'o-', lw=2, ms=3, markeredgecolor='red', label=fr'$\kappa$={n}')
    plt.fill_between(kappas, R_list[:, 0]+R_list[:, 1], R_list[:, 0]-R_list[:, 1], alpha=0.5, color='grey')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.ylim(np.min(R_list[:, 0]-R_list[:, 1])-0.01, np.max(R_list[:, 0]+R_list[:, 1])+0.01)
    plt.show()

def fig_3D(kappas:list, n_list:list, R_map:np.ndarray,
                  xlabel:str=r"$\kappa$ [-]", ylabel:str=r"n [-]", zlabel:str=r"<R> [-]", 
                  titlename:str=r"Synchronisation totale d'un graph pour différents $\kappa$"):
    ax = plt.figure(figsize=(10, 5)).add_subplot(projection='3d')

    kappa_grid, nlist_grid = np.meshgrid(kappas, n_list)
    X, Y, Z = nlist_grid, kappa_grid, R_map[:, :, 0]
    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    # ax.contour(X, Y, Z, zdir='z', offset=np.min(R_map[:, :, 0]-R_map[:, :, 1])-0.01, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=np.max(n_list)+2, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=np.min(kappas)-2, cmap='coolwarm')

    ax.set(xlim=(np.min(n_list)-2, np.max(n_list)+2), ylim=(np.min(kappas)-2, np.max(kappas)+2), 
           zlim=(np.min(R_map[:, :, 0]-R_map[:, :, 1])-0.01, np.max(R_map[:, :, 0]+R_map[:, :, 1])+0.01),
        xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    # plt.fill_between(kappas, R_list[:, 0]+R_list[:, 1], R_list[:, 0]-R_list[:, 1], alpha=0.5, color='grey')
    # ax.title(titlename)
    plt.tick_params(direction = 'in')
    plt.show()


    # Dynamique N neurone theta.
size = 100
euler = 0
scipy = 1
mean = 100

kappas = [1, 5, 10, 15, 20]#, 15, 20, 30
# kappas = [2, 5, 10]

n_list = [1, 5, 10, 15, 20]
# n_list = [1, 4, 8, 12]
# n_list = [2, 5, 10]

R_map = np.zeros((len(kappas), len(n_list), 2))
for _ in range(mean):
    # theta_0 = np.random.normal(np.pi, 1, (size, 1))
    theta_0 = np.random.standard_normal((size, 1))
    # theta_0 = np.random.rand(size, 1)*2*np.pi

    A = np.ones((size, size))#-np.identity(size)
    # A = np.random.normal(0.3, 0.05, (size, size))
    for j, kappa in enumerate(kappas):
        for i, n in enumerate(n_list):

            # n = 2
            beta_0 = 0
            # kappa = 10

            a_n = 2**n*(facto(n))**2/facto(2*n)
            step_approx = 0.01/(2*(0.00001+beta_0 + a_n*kappa*2**n))

            time_start = 0
            time_step = step_approx
            time_stop = 500
            time_list = np.linspace(time_start, time_stop+time_step, int(time_stop/time_step)+1)


            if euler:
                theta, beta = dynamique_N_neurones(time_step, time_stop, theta_0, theta_model, betaNneurones, A, n, beta_0, kappa)
            
            if scipy:
                sol  = solve_ivp(theta_model_scipy_solver, [time_start, time_stop], theta_0.T[0], first_step=1e-6, max_step=1, atol = 0, rtol = 1e-8, args = (betaNneurones, A, n, beta_0, kappa))
                time_list, theta = sol.t, sol.y
            
            synchro = indice_syncronisation(theta)
            R, std = half_mean(synchro, time_list, True)
            R_map[j, i, :] += np.array([R, std])/mean

if len(R_map) == 1 and mean == 1:
    if euler:
        fig_sync(time_list, theta)
        spike_rep_euler = spike_rep(theta, 30)/2**30
        # fig_spike_rep(time_list, spike_rep_euler, theta_0)
        raster_plot_array = []
        for neuron in spike_rep_euler:
            raster_plot_array += [time_list[find_peaks(neuron, 0.5)[0]]]
        fig_raster(raster_plot_array, time_stop)
        fig_heatmap(theta)

    # Scipy version
    if scipy:
        fig_sync(time_list, theta)
        spike_rep_scipy = spike_rep(theta, 30)/2**30
        # fig_spike_rep(sol.t, spike_rep_scipy, theta_0)
        raster_plot_scipy = []
        for neuron in spike_rep_scipy:
            raster_plot_scipy += [time_list[find_peaks(neuron, 0.5)[0]]]
        fig_raster(raster_plot_scipy, time_stop)
        # fig_heatmap(theta)

else:
    plt.figure(figsize=(10, 5))
    for k, kappa in enumerate(kappas):
        fig_mean_sync(n_list, R_map[k], labeltxt=fr'$\kappa$={kappa}', 
                        xlabel=r"n [-]", ylabel=r"<R> [-]", 
                        title=fr"Synchronisation moyenne pour différents $\kappa$ selon n.")
    
    plt.tick_params(direction = 'in')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    for i, n in enumerate(n_list):
        fig_mean_sync(kappas, R_map[:, i, :], labeltxt=fr'n={n}', 
                        xlabel=r"$\kappa$ [-]", ylabel=r"<R> [-]", 
                        title=fr"Synchronisation moyenne pour différents n selon $\kappa$.")
    
    plt.tick_params(direction = 'in')
    plt.legend()
    plt.show()
    fig_3D(kappas, n_list, R_map)
