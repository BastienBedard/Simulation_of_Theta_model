import numpy as np
import numpy.linalg as nplin
import scipy.linalg as sclin
import matplotlib.pyplot as plt

print("Hello world!")

time_step = np.arange(0, 10, 0.001)
theta_list = np.linspace(0, 4*np.pi, 1000)
beta_list = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])

def theta_model(theta, beta):
    return 1-np.cos(theta)+np.dot((1+np.cos(theta)),beta)

def spike_rep(theta, definition = 1):
    return (1-np.cos(theta))**definition

def beta_model(beta):
    return np.cos(beta*5*np.pi)/(5*np.pi)

def dynamique(time_step, time_stop, theta_0, beta_0):
  theta = np.array([theta_0])
  beta = np.array([beta_0])
  for i in range(time_stop/time_step):
    Vtheta = theta_model(theta[i], beta[i])
    Vbeta = beta_model(beta[i])
    theta = np.concatenate((theta, theta[i]+Vtheta*time_step))
    beta = np.concatenate((beta, beta[i]+Vbeta*time_step))
  return theta, beta


plt.figure(figsize=(10, 6))

for beta in beta_list:
  plt.plot(theta_list, theta_model(theta_list, beta), label=fr'$\beta$ = {beta}')#+f' {beta}'
plt.xlabel(r"Valeurs de $\theta$")
plt.ylabel(r"Vitesse de $\theta$")
plt.title("Title")
plt.tick_params(direction = 'in')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(theta_list, spike_rep(theta_list))
plt.xlabel(r"Valeurs de $\theta$")
plt.ylabel(r"1-cos($\theta$ )")
plt.title("Title")
plt.tick_params(direction = 'in')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
spike_list = np.array([1, 2, 3, 4, 6, 10, 15, 30])
half_theta = theta_list[:int(len(theta_list)/2)]
for i in spike_list:
  ydata = spike_rep(half_theta, i)
  plt.plot(half_theta, ydata/np.max(ydata), label=f'n = {i}')
plt.xlabel(r"Valeurs de $\theta$")
plt.ylabel(r"valeur de (1-cos$(\theta ))^n normalisé$")
plt.title("Title")
plt.tick_params(direction = 'in')
plt.legend()
plt.grid(True)
plt.show()