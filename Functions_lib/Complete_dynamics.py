import numpy as np
from math import factorial as facto



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
         beta[i, 0] = np.sum(np.multiply(A[i], spike_rep(theta_list, n).T[0]))
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

def dist_lorentzienne(beta:np.ndarray, beta_0:float=0, gamma:float=0.1):
   return (gamma/np.pi)/((beta-beta_0)**2+gamma**2)

def true_samples(samples, beta_0, gamma):
   return beta_0+gamma*samples
