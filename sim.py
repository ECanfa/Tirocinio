import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#Sistema in forma di stato
# dx1dt = x2
# dx2dt = -kk/mm x1 - bb/mm x2 + 1/m u(t)

mm   = 1  
kk   = 70   
bb   = 8   

#Condizioni iniziali
x_1_0 = 10
x_2_0 = 0
x_0 = [x_1_0,x_2_0]

t = (0,20)
t_eval = np.linspace(*t, 10000) 

def u(t):
    return np.sin(t)


def massa_molla(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u(t)
    return [dx1dt, dx2dt]

u_v  = np.random.randn(10000) #poichè i valori sono generati randomicamente li salvo in un array
u_interp = interp1d(t_eval, u_v, fill_value="extrapolate")

def massa_molla_v(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u_interp(t)
    return [dx1dt, dx2dt]



soluzione = solve_ivp(massa_molla, t, x_0, t_eval=t_eval)


plt.plot(soluzione.t, soluzione.y[0])
plt.xlabel("Tempo")
plt.ylabel("Posizione x1(t)")
plt.title("Sistema massa-molla-smorzatore")
plt.grid()
plt.show()

plt.plot(soluzione.t, soluzione.y[1])
plt.xlabel("Tempo")
plt.ylabel("Velocità x2(t)")
plt.title("Sistema massa-molla-smorzatore")
plt.grid()
plt.show()


data = np.column_stack((soluzione.t, soluzione.y[0], soluzione.y[1], u(soluzione.t)))

np.savetxt("simulazione.csv", data, delimiter=",",
           header="tempo,posizione,velocita,input", comments="")


# Simulo nuovamente il sistema con input differente per generare dati di validazione
validazione = solve_ivp(massa_molla_v, t, x_0, t_eval=t_eval)

data_v= np.column_stack((validazione.t, validazione.y[0], validazione.y[1], u_interp(validazione.t)))
np.savetxt("validazione.csv", data_v, delimiter=",",
           header="tempo,posizione,velocita,input", comments="")
