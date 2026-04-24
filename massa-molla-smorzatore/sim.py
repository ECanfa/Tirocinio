import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math as m
import random

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
omega_n = m.sqrt(kk/mm)
xi = bb/(2*m.sqrt(mm*kk))
print(xi)
omega_ris = omega_n* m.sqrt(1-2*xi*xi)
t_eval = np.linspace(*t, int(60*(max(2*omega_ris, 15*np.pi))/(2*np.pi))) #frequenza di campionamento nel rispetto del teorema di Shannon Nyquist 

def massa_molla(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u_t(t)
    return [dx1dt, dx2dt]



def u_t(t):
    moltiplicatori = [0.2, 0.5, 1.0, 1.5, 2.0]
    
    return (sum(np.sin(i*omega_ris*t + random.random()*np.pi) for i in moltiplicatori)) #input multisine intorno alla pulsazione di risonanza + fasi pseudo-casuali 
     

def massa_molla_v(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u_v(t)
    return [dx1dt, dx2dt]



def u_v(t):
    frequenze_v = np.linspace(0.15, 5.2, 15)
    return sum(np.sin(np.pi*f*t) for f in frequenze_v) #come scegliere i valori dell'input? 
    


soluzione = solve_ivp(massa_molla, t, x_0, t_eval=t_eval)

'''
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
'''



data = np.column_stack((soluzione.t, soluzione.y[0], soluzione.y[1], u_t(soluzione.t)))

np.savetxt("simulazione.csv", data, delimiter=",",
           header="tempo,posizione,velocita,input", comments="")


# Simulo nuovamente il sistema con input differente per generare dati di validazione
validazione = solve_ivp(massa_molla_v, t, x_0, t_eval=t_eval)

data_v= np.column_stack((validazione.t, validazione.y[0], validazione.y[1], u_v(validazione.t)))
np.savetxt("validazione.csv", data_v, delimiter=",",
           header="tempo,posizione,velocita,input", comments="")
