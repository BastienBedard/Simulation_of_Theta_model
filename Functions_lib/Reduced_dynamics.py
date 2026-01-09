import numpy as np
from scipy.integrate import quad
from math import comb


# Definition des fonctions pour la réduction DART

# c_n = {1:{0:1, 1:-1}, 2:{0:3/2, 1:-2, 2:0.5}, 3:{0:5/2, 1:-3.75, 2:1.5, 3: -1/4}, 
#        4:{0:35/8, 1:-7, 2:3.5, 3:-1, 4:1/8}, 5:{0:63/8, 1:-105/8, 2:7.5, 3:-45/16, 4:15/24, 5:-15/240}}
# c_n = c_n_dict_bulder(51)

# def a_n_dict_check(s: int, n: int):
#     if s in c_n.keys():
#         a_n = c_n[s][n]
#     else:
#         print('STOP S is too big!')
#         a_n = numerical_An(s, n)
#     return a_n

def numerical_An(s: int, n: int):
    """Compute A_n for given s using numerical integration."""
    integrand = lambda theta: (1 - np.cos(theta))**s * np.cos(n * theta)
    result, _ = quad(integrand, 0, 2 * np.pi)
    if n==0:
        result /= 2
    return round(result / np.pi, 12) 

def a_n_close_form(s: int, n: int):
    a_n = 0
    for r in range(int(np.floor((s-n)/2))+1):
        a_n += comb(s, n+2*r)*(-0.5)**(n+2*r)*(comb(n+2*r, n+r)+comb(n+2*r, r))
    if n==0: a_n*=0.5
    return a_n

def c_n_dict_bulder(s: int):
    c_n={}
    for i in range(1, s+1):
        c_n[i]={}
        for j in range(i+1):
            c_n[i][j]=a_n_close_form(i, j)
    return c_n

def cos_expansion(Z: complex, s: int, c_n: dict):
    rep = 0
    a_n = 0
    max=s+1
    if max>101:
        max=101
    for n in range(max):
        a_n = c_n[s][n]
        rep+=a_n*(Z**n+np.conj(Z)**n)/2
    return rep

def Z_reduced_dynamic_manager(t, Z: np.ndarray, n: int, sigma: float, kappa: np.ndarray, Omega: np.ndarray,
                       W_curl: np.ndarray, K_curl: np.ndarray, A_curl: np.ndarray, s: int, N: int, a_s: float, c_eta: dict):
    Zdot = np.empty(np.shape(Z), dtype=complex)
    for mu in range(n):
        Zdot[mu]=Z_reduced_dynamic(t, Z, mu, sigma, kappa, Omega, W_curl, K_curl, A_curl, s, N, a_s, c_eta)[0]
    return Zdot

def Z_reduced_dynamic(t, Z: np.ndarray, mu: int, sigma: float, kappa: np.ndarray, Omega: np.ndarray,
                       W_curl: np.ndarray, K_curl: np.ndarray, A_curl: np.ndarray, s: int, N: int, a_s: float, c_eta: dict):
    sumeta = 0
    sumOm = W_curl[mu]@Z
    sumkap = K_curl[mu]@Z
    sumA = A_curl[mu]@Z
    sumAconj = A_curl[mu]@np.conj(Z)
    for eta in range(s+1):
        a_eta = c_eta[s][eta]
        sumeta += a_eta*(sumA**eta+sumAconj**eta)/(2*kappa[mu]**eta)
    return -0.5j*(Z[mu]-1)**2+0.5j*Omega[mu]*(sumOm/Omega[mu]+1)**2+0.5j*kappa[mu]*sigma*a_s*(sumkap/kappa[mu]+1)**2*(sumeta[0])/N

def Z_reduced_dynamic_homogeneOG(t, Z: complex, omega_mean: float, sigma: float, s: int, a_s: float):
    return -0.5j*(Z-1)**2+0.5j*(Z+1)**2*(omega_mean+sigma*a_s*(1-0.5*(Z+np.conj(Z)))**s)


def Z_reduced_dynamic_homogene(t, Z: complex, omega_mean: float, sigma: float, s: int, a_s: float, c_n: dict):
    h = a_s*cos_expansion(Z, s, c_n)
    return -0.5j*(Z-1)**2+0.5j*(Z+1)**2*(omega_mean+sigma*h)

def R_reduced_dynamic_homogene(R: float, Phi: float, omega_mean: float, sigma: float, s: int, a_s: float, c_n: dict):
    h = a_s*cos_expansion(R*np.exp(1j*Phi), s, c_n)
    return 0.5*(1-R**2)*np.sin(Phi)*(omega_mean-1+sigma*h)

def Phi_reduced_dynamic_homogene(R: float, Phi: float, omega_mean: float, sigma: float, s: int, a_s: float, c_n: dict):
    h = a_s*cos_expansion(R*np.exp(1j*Phi), s, c_n)
    return 1-(1+R**2)*np.cos(Phi)/(2*R)+(1+(1+R**2)*np.cos(Phi)/(2*R))*(omega_mean+sigma*h)

def R_Phi_reduced_dynamic_homogene(t, RPhi: np.ndarray, omega_mean: float, sigma: float, s: int, a_s: float, c_n: dict):
    R, Phi = RPhi
    Rdot = R_reduced_dynamic_homogene(R, Phi, omega_mean, sigma, s, a_s, c_n)
    Phidot = Phi_reduced_dynamic_homogene(R, Phi, omega_mean, sigma, s, a_s, c_n)
    return np.array([Rdot, Phidot])


def Moore_penrose(M: np.ndarray):
    return np.conj(M.T)@np.linalg.inv((M@(np.conj(M.T))))

def reduced_dynamic_params_init(M: np.ndarray, beta_0: np.ndarray, A: np.ndarray, z_0: np.ndarray, N: int):
    W = np.diag(beta_0.T[0])
    K = np.diag(np.sum(A, axis=1))
    # K = np.diag(np.sum(np.abs(A), axis=1))

    W_curl = M@W@Moore_penrose(M)
    K_curl = M@K@Moore_penrose(M)
    A_curl = M@A@Moore_penrose(M)
    Z_0 = M@z_0

    kappa, Omega = K_curl[0], W_curl[0]

    if not (M==np.ones((1, N))/N).all():
        kappa = M@K@np.ones((N, 1))
        Omega = M@W@np.ones((N, 1))
    return Z_0.T[0], kappa, Omega, K_curl, W_curl, A_curl, W, K

def V_A_builder(value, vector, n=2):
    V_A = np.empty((n, len(vector[:, 0])))
    valuemodified = value+0
    for i in range(n):
        v_max = np.max(np.abs(valuemodified))
        if v_max == 0:
            raise ValueError('La valeur propre maximal est trivial')
        v_idx = np.where(np.abs(valuemodified)==v_max)[0][0]
        valuemodified[v_idx] = 0
        V_A[i, :] = vector[:, v_idx]*np.sign(np.mean(vector[:, v_idx]))
    return V_A

def m_builder(M):
    m = np.count_nonzero(M, axis=1)/np.count_nonzero(M)
    if np.count_nonzero(M) > int(len(M[0])*len(M)):
        print('countM=', np.count_nonzero(M))
        print(m)
        raise ValueError('Too many non zero value in M')
    return m

def m_builder_weighted2(mat):
    m = np.sum(mat, axis=1)*np.count_nonzero(mat, axis=1)/np.count_nonzero(mat)
    return m/np.sum(m)