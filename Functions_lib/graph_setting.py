import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from .Complete_dynamics import spike_rep

def half_mean(data:np.ndarray, time_list:np.ndarray, std:bool=False):
   midtime = (time_list[-1]-time_list[0])/2
   idx = np.abs(time_list - midtime).argmin()
   half_data = data[idx:]
   half_time = time_list[idx:]
   data = np.multiply(half_data[1:], np.diff(half_time))/np.mean(np.diff(half_time))
   hmean = np.mean(data)
   if std:
      return hmean, np.std(data)/np.sqrt(len(data))
   return hmean

def raster_and_frate_builder(time_list:list, theta:np.ndarray, time_stop:int, normilise:bool=False):
   spike_rep_array = spike_rep(theta, 30)/2**30
   raster_plot = []
   for neuron in spike_rep_array:
      raster_plot += [time_list[find_peaks(neuron, 0.5)[0]]]
   fire_list = np.bincount(np.array(np.concatenate(raster_plot), dtype=int))/len(theta)
   if len(fire_list) != time_stop:
      pad_width = time_stop - len(fire_list)
      fire_list = np.pad(fire_list, (0, pad_width), constant_values=0)
   if normilise:
      return raster_plot, fire_list, fire_list/np.max(fire_list)
   return raster_plot, fire_list

def indice_syncronisation(theta):
   return np.sqrt(np.sum(np.cos(theta), axis = 0)**2 + np.sum(np.sin(theta), axis = 0)**2)/len(theta)

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
    plt.ylim(-0.05, 1.05)
    plt.show()

def fig_sync_frate(time_list:np.ndarray, theta:np.ndarray, fire_list:np.ndarray, neighbors:int=0, xlabel:str="Temps [s]", 
             ylabel1:str="<R>", ylabel2:str=r"Taux de decharge moyen par neurones. [$s^{-1}$]", title:str="Évolution temporelle de la synchronisation et du taux de décharge"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(time_list, indice_syncronisation(theta), color='blue')
    ax1.set_ylabel(ylabel1, color='blue')
    ax1.tick_params(axis='y', direction = 'in', colors='blue')
    if neighbors:
        window_size = 2*neighbors+1
        fire_list = uniform_filter1d(fire_list, size=window_size, mode='mirror')
    ax2.plot(fire_list, color='green')
    ax2.set_ylabel(ylabel2, color='green')
    ax2.tick_params(axis='y', direction = 'in', colors='green')
    ax1.set_xlabel(xlabel)
    plt.title(title)
    ax1.tick_params(axis='x', direction = 'in')
    ax1.set_ybound(-0.05, 1.05)
    ax2.set_ybound(-0.05*np.max(fire_list), np.max(fire_list)*1.05)
    ax1.grid(True)
    plt.show()

def fig_frate(fire_list:np.ndarray, neighbors:int=0, xlabel:str="Temps [s]", 
             ylabel:str=r"Taux de decharge moyen par neurones [$s^{-1}$]", title:str="Évolution du taux de décharge au fils du temps"):
    plt.figure(figsize=(10, 6))
    if neighbors:
        window_size = 2*neighbors+1
        fire_list = uniform_filter1d(fire_list, size=window_size, mode='mirror')
    plt.plot(fire_list)#, 'ro', ms=1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.grid(True)
    plt.ylim(-0.05, np.max(fire_list)+0.05)
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

def fig_raster(raster_plot_list:list, time_stop:int, xlabel:str="Temps [s]", 
                ylabel:str="Énumération des neurones. [-]", 
                title:str="Raster plot des pics d'émission de potentiel des neurones."):
    plt.figure(figsize=(12, 3))
    plt.eventplot(raster_plot_list[::-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, time_stop)
    plt.ylim(0)
    plt.show()

def fig_spike_rep(time_list:np.ndarray, theta:np.ndarray, theta_0:np.ndarray, legend:bool=False, 
                  xlabel:str="Temps [s]", ylabel:str=r"valeur de (1-cos$(\theta ))^{30}$ normalisé [-]", 
                  title:str="Représentation neuronal de theta pour explicité les pics de potentiel."):
    plt.figure(figsize=(10, 10))
    spike_rep_list = spike_rep(theta, 30)/2**30
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

def fig_mean_sync_all(xdata:list, R_map:np.ndarray, params:np.ndarray, labeltxt:str = r'n', axis:int = 1, stretchlim:bool = True,
                  xlabel:str=r"$\kappa$ [-]", ylabel:str="<R> [-]", 
                  title:str=r"Synchronisation totale d'un graph pour différents $\kappa$"):
    plt.figure(figsize=(7, 4))
    for i, param in enumerate(params):
        if axis:
            R_list = R_map[:, i, :]
        else:
            R_list = R_map[i]
        plt.plot(xdata, R_list[:, 0], 'o-', lw=1, ms=2, markeredgecolor='red', label=labeltxt+f'={param}')
        plt.fill_between(xdata, R_list[:, 0]+R_list[:, 1], R_list[:, 0]-R_list[:, 1], alpha=0.5, color='grey')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.legend()
    plt.ylim(-0.05, 1.05)
    if not stretchlim:
        plt.ylim(np.min(R_list[:, 0]-R_list[:, 1])-0.01, np.max(R_list[:, 0]+R_list[:, 1])+0.01)
    plt.grid(True)
    plt.show()

def fig_mean_sync(kappas:list, R_list:np.ndarray,
                  xlabel:str=r"$\kappa$ [-]", ylabel:str=r"<R> [-]", 
                  title:str=r"Synchronisation totale d'un graph pour différents $\kappa$"):
    plt.figure(figsize=(7, 4))
    plt.plot(kappas, R_list[:, 0])#, 'o-', lw=1, ms=2, markeredgecolor='red'
    plt.fill_between(kappas, R_list[:, 0]+R_list[:, 1], R_list[:, 0]-R_list[:, 1], alpha=0.5, color='grey')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.ylim(np.min(R_list[:, 0]-R_list[:, 1])-0.01, np.max(R_list[:, 0]+R_list[:, 1])+0.01)
    plt.show()

def fig_mean_frate(f_map:np.ndarray, kappas:list, n_list:list, yname:str, xname:str = r'$\kappa$ [-]', title:str='Taux de décharge final moyen par neurone.', colo_horizon:bool = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(f_map.T, origin='lower')
    ax.set_xticks(np.arange(len(kappas)))
    ax.set_yticks(np.arange(len(n_list)))
    ax.set_xticklabels(kappas)
    ax.set_yticklabels(n_list)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.xaxis.set_ticks_position('bottom')
    plt.title(title)
    if colo_horizon is True:
        plt.colorbar(cax, orientation="horizontal")
    else:
        plt.colorbar(cax)
    plt.show()

def fig_mean_sync_heat_map(sync_map:np.ndarray, kappas:list, n_list:list, yname:str, xname:str = r'$\kappa$ [-]', title:str="Synchronisation totale moyenne.", colo_horizon:bool = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(sync_map.T, origin='lower')
    ax.set_xticks(np.arange(len(kappas)))
    ax.set_yticks(np.arange(len(n_list)))
    ax.set_xticklabels(kappas)
    ax.set_yticklabels(n_list)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.xaxis.set_ticks_position('bottom')
    plt.title(title)
    if colo_horizon is True:
        plt.colorbar(cax, orientation="horizontal")
    else:
        plt.colorbar(cax)
    plt.show()

def fig_3D(kappas:list, n_list:list, R_map:np.ndarray,
                  xlabel:str=r"$\kappa$ [-]", ylabel:str=r"n [-]", zlabel:str=r"<R> [-]", 
                  titlename:str=r"Synchronisation totale d'un graph pour différents $\kappa$"):
    ax = plt.figure(figsize=(10, 5)).add_subplot(projection='3d')

    kappa_grid, nlist_grid = np.meshgrid(kappas, n_list)
    X, Y, Z = kappa_grid.T, nlist_grid.T, R_map[:, :, 0]

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=1, cstride=1, alpha=0.3)
    # ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

    ax.contour(X, Y, Z, levels = len(kappas), zdir='x', offset=np.min(kappas)-2, cmap='coolwarm')
    ax.contour(X, Y, Z, levels = len(n_list), zdir='y', offset=np.max(n_list)+2, cmap='coolwarm')

    ax.set(xlim=(np.min(kappas)-2, np.max(kappas)+2), ylim=(np.min(n_list)-2, np.max(n_list)+2), 
            zlim=(np.min(R_map[:, :, 0]-R_map[:, :, 1])-0.01, np.max(R_map[:, :, 0]+R_map[:, :, 1])+0.01),
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    plt.show()

def fig_sync_reduced(time_list:np.ndarray, R:np.ndarray, xlabel:str="Temps [s]", 
             ylabel:str="<R> [-]", title:str="Évolution temporel de la synchronisation du modèle réduit"):
    plt.figure(figsize=(10, 6))
    plt.plot(time_list, R)#, 'ro', ms=1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.show()

def fig_mean_sync_and_reduced(xdata:list, R_map:np.ndarray, reduced_map:np.ndarray, params:np.ndarray, labeltxt:str = r'$s$', axis:int = 1, stretchlim:bool = True,
                  xlabel:str=r"$\sigma$ [-]", ylabel:str="<R> [-]", 
                  title:str=r"Synchronisation totale d'un graph pour différents $\sigma$"):
    plt.figure(figsize=(7, 4))
    for i, param in enumerate(params):
        if axis:
            R_list = R_map[:, i, :]
            reduced_list = reduced_map[:, i, :]
        else:
            R_list = R_map[i]
            reduced_list = reduced_map[i]
        plt.plot(xdata, reduced_list[:, 0], 'o--', lw=1, ms=1, markeredgecolor='black', alpha=0.9, c='black')
        plt.fill_between(xdata, reduced_list[:, 0]+reduced_list[:, 1], reduced_list[:, 0]-reduced_list[:, 1], alpha=0.5, color='grey')
        plt.plot(xdata, R_list[:, 0], 'o-', lw=1, ms=2, markeredgecolor='red', label=labeltxt+f'={param}', alpha=0.7, c=f'C{i}')
        plt.fill_between(xdata, R_list[:, 0]+R_list[:, 1], R_list[:, 0]-R_list[:, 1], alpha=0.3, color=f'C{i}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(direction = 'in')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(-0.05, 1.05)
    if not stretchlim:
        plt.ylim(np.min(R_list[:, 0]-R_list[:, 1])-0.01, np.max(R_list[:, 0]+R_list[:, 1])+0.01)
    plt.grid(True)
    plt.show()