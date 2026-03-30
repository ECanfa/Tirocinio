import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Sistema in forma di stato
# dx1dt = x2
# dx2dt = -kk/mm x1 - bb/mm x2 + 1/m u(t)

## Parametri presi da un esempio fatto in aula

mm   = 0.5 # 
kk   = 80  # 
bb   = 10  # 

def u(t):
    return np.sin(t)

def massa_molla(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u(t)
    return [dx1dt, dx2dt]


#Condizioni iniziali
x_1_0 = 10
x_2_0 = 0
x_0 = [x_1_0,x_2_0]

t = (0,20)
t_eval = np.linspace(*t, 10000)

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

# esempio: sol.t e sol.y dal solve_ivp
data = np.column_stack((soluzione.t, soluzione.y[0], soluzione.y[1]))

np.savetxt("simulazione.csv", data, delimiter=",",
           header="tempo,posizione,velocita", comments="")


