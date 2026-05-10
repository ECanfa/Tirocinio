import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math as m
import random

#per avere numeri randomici "costanti"
random.seed(42)
np.random.seed(42)

#Sistema in forma di stato
# dx1dt = x2
# dx2dt = -kk/mm x1 - bb/mm x2 + 1/m u(t)

#parametri del sistema
mm   = 1  
kk   = 70   
bb   = 8   

#Condizioni iniziali
x_1_0 = 0.5
x_2_0 = 0.05
x_0_t = [x_1_0,x_2_0]

x_1_0_v = 0.3
x_2_0_v = 0.03
x_0_v = [x_1_0_v, x_2_0_v]

#calcolo pulsazione naturale e di risonanza
omega_n = m.sqrt(kk/mm)
xi = bb/(2*m.sqrt(mm*kk))
omega_ris = omega_n* m.sqrt(1-2*xi*xi)
print(omega_ris)

#definizione dei sistemi per training e validazione
def massa_molla(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u_t(t)
    return [dx1dt, dx2dt]

def massa_molla_v(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(kk/mm)*x1 - (bb/mm)*x2 + (1/mm)*u_v(t)
    return [dx1dt, dx2dt]

#definizione input di training e validazione
frequenze_t = np.linspace(0.1, 5, 30) * omega_ris
fasi_t = [random.uniform(0, 2*np.pi) for _ in range(len(frequenze_t))] # aggiungo fasi per evitare picco di somma delle sinusoidi
max_f_t = max(frequenze_t)

def u_t(t):
    
    return (sum(5*np.sin(f*t + p) for f,p in zip(frequenze_t, fasi_t))) #input multisine intorno alla pulsazione di risonanza 

frequenze_v = np.linspace(3, 8, 30) * omega_ris     
fasi_v = [random.uniform(0, 2*np.pi) for _ in range(len(frequenze_v))]
max_f_v = max(frequenze_v)

def u_v(t):
    
    return sum(5*np.sin(f*t + p) for f,p in zip(frequenze_v, fasi_v)) #come scegliere i valori dell'input? 

#Tempo di simulazione e campionamento
t = (0,30)
t_eval = np.linspace(*t, int(100*(max(max_f_t, max_f_v))/(2*np.pi))) #frequenza di campionamento nel rispetto del teorema di Shannon Nyquist 


#simulazione training e validazione
soluzione = solve_ivp(massa_molla, t, x_0_t, t_eval=t_eval)

validazione = solve_ivp(massa_molla_v, t, x_0_v, t_eval=t_eval)


#salvataggio dei dati
data = np.column_stack((soluzione.t, soluzione.y[0], soluzione.y[1], u_t(soluzione.t)))
np.savetxt("simulazione.csv", data, delimiter=",",
           header="tempo,posizione,velocita,input", comments="")


data_v= np.column_stack((validazione.t, validazione.y[0], validazione.y[1], u_v(validazione.t)))
np.savetxt("validazione.csv", data_v, delimiter=",",
           header="tempo,posizione,velocita,input", comments="")
