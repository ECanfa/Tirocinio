from sippy import systemIdentification, control
from scipy.linalg import svd, lstsq
import numpy as np
import matplotlib.pyplot as plt
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.metrics import root_relative_squared_error
from scipy.signal import dlsim

data = np.loadtxt("simulazione.csv", delimiter=",", skiprows=1)

t = data[:, 0]
x1 = data[:, 1] #posizione
x2 = data[:, 2] #velocità
u = data[:, 3]

y = x1

Ts = t[1] - t[0]

def n4sid(y, u, order, i=10):
    """
    y: array (N,) output
    u: array (N,) input
    order: ordine del sistema
    i: numero di righe delle matrici di Hankel (i > order)
    """
    N = len(y)
    j = N - 2*i  # numero di colonne

    # Costruzione matrici di Hankel
    Y = np.zeros((2*i, j))
    U = np.zeros((2*i, j))
    for k in range(2*i):
        Y[k, :] = y[k:k+j]
        U[k, :] = u[k:k+j]

    # Divisione passato/futuro
    Yp = Y[:i, :]
    Yf = Y[i:, :]
    Up = U[:i, :]
    Uf = U[i:, :]

    # Proiezione
    W = np.vstack([Up, Yp, Uf])
    proj = Yf @ W.T @ np.linalg.pinv(W @ W.T) @ W

    # SVD
    U_svd, S, Vt = svd(proj)
    
    # Mostra i valori singolari per scegliere l'ordine
    plt.figure()
    plt.bar(range(1, len(S)+1), S)
    plt.xlabel("Indice")
    plt.ylabel("Valore singolare")
    plt.title("Valori singolari - scegli ordine")
    plt.show()

    # Estrazione sottospazio di ordine n
    U1 = U_svd[:, :order]
    S1 = np.diag(S[:order])
    
    # Sequenza di stati
    X = np.sqrt(S1) @ U1.T @ proj

    # Stima matrici A, C
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Y_out = Yf[:1, :-1]  # prima riga = output

    # Least squares per A, B, C, D
    M1 = np.vstack([X2, Y_out])
    M2 = np.vstack([X1, u[i:i+X1.shape[1]].reshape(1,-1)])
    
    result, _, _, _ = lstsq(M2.T, M1.T)
    result = result.T

    n = order
    A = result[:n, :n]
    B = result[:n, n:]
    C = result[n:, :n]
    D = result[n:, n:]

    return A, B, C, D


# Uso
A, B, C, D = n4sid(y.flatten(), u.flatten(), order=2)



validazione = np.loadtxt("validazione.csv", delimiter=",",skiprows=1)

tc = validazione[:,0]
xc2 = validazione[:,2]
xc1 = validazione[:, 1]
uc = validazione[:,3]

yc = xc1

sys_discrete = (A, B, C, D, Ts)
t_out,_, yhat = dlsim(sys_discrete, uc.flatten().reshape(-1,1))
yhat = yhat.reshape(-1, 1)

rrse = root_relative_squared_error(yc, yhat) 
print(rrse)

plot_results(
    y=yc,
    yhat=yhat, 
    n=1000,
    title="test",
    xlabel="Samples",
    ylabel=r"y, $\hat{y}$",
    data_color="#1f77b4",
    model_color="#ff7f0e",
    marker="o",
    model_marker="*",
    linewidth=1.5,
    figsize=(10, 6), 
    style="seaborn-v0_8-notebook",  
    facecolor="white",
)

ee = compute_residues_autocorrelation(yc, yhat)
plot_residues_correlation(
    data=ee, title="Residues", ylabel="$e^2$", style="seaborn-v0_8-notebook"
)
x1e = compute_cross_correlation(yc, yhat, uc)
plot_residues_correlation(
    data=x1e, title="Residues", ylabel="$ue$", style="seaborn-v0_8-notebook"
)