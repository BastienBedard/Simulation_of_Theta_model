import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Functions_lib.Complete_dynamics import *

# distribution lorentzienne

gammas = np.array([0.5, 0.1, 0.05, 0.01])
gamma = 0.1
beta_0 = -0.4
beta_list = np.linspace(-2, 1.2, 1000)


rng = np.random.default_rng(seed=420)  # use Generator API
beta_samples = true_samples(rng.standard_cauchy(size=1000), beta_0, gamma)
betas = true_samples(np.random.standard_cauchy((1, 1000)), beta_0, gamma)

plt.figure(figsize=(10, 6))
for gamma in gammas:
    plt.plot(beta_list, dist_lorentzienne(beta_list, beta_0, gamma), label=fr' $\gamma$= {gamma}')

plt.xlabel(r"Valeurs de $\beta$ [-]")
plt.ylabel(r"Densité de probabilité [-]")
plt.title(fr"Distribution de Lorentz pour $\beta_0$={beta_0} et différents $\gamma$")
plt.tick_params(direction = 'in')
plt.legend()
plt.grid(True)
plt.show()
plt.scatter(betas, dist_lorentzienne(betas, beta_0, gamma), s=2)
plt.show()
plt.scatter(beta_samples, dist_lorentzienne(beta_samples, beta_0, gamma), s=2)
plt.show()