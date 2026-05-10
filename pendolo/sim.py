import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math as m
import random

#Sistema in forma di stato
# x1 angolo, x2 velocità angolare
# dx1dt = x2 
# dx2dt = -g/l*sin(x1) - b/ml^2*x2 + 1/ml^2 u

random.seed(42)
np.random.seed(42)

gg = 9.81
bb = 2
mm = 0.5
ll = 1

#Condizioni iniziali
x_1_0 = np.pi/4#in radianti
x_2_0 = 0.005
x_0_t = [x_1_0,x_2_0]


x_1_0 = 3*np.pi/10#in radianti
x_2_0 = 0.002
x_0_v = [x_1_0,x_2_0]

omega_n = m.sqrt(gg/ll)
xi = bb/(2*mm*ll*ll*omega_n)
omega_ris = omega_n* m.sqrt(1-2*xi*xi)
    

def pendolo_t(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(gg/ll)*np.sin(x1) - (bb/(mm*ll*ll))*x2 + (1/(mm*ll*ll))*u_t(t)
    return [dx1dt, dx2dt]


def pendolo_v(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(gg/ll)*np.sin(x1) - (bb/(mm*ll*ll))*x2 + (1/(mm*ll*ll))*u_v(t)
    return [dx1dt, dx2dt]

frequenze_t = np.linspace(0.1, 5, 30) * omega_ris
fasi_t = [random.uniform(0, 2*np.pi) for _ in range(len(frequenze_t))] # aggiungo fasi per evitare picco di somma delle sinusoidi
max_f_t = max(frequenze_t)

def u_t(t):
    
    return (sum(10*np.sin(f*t+p) for f,p in zip(frequenze_t, fasi_t))) #uso un input intorno alla frequenza di risonanza perchè è dove il sistema è maggiormente stimolato 

frequenze_v = np.linspace(3 , 8, 30) * omega_ris     
fasi_v = [random.uniform(0, 2*np.pi) for _ in range(len(frequenze_v))]
max_f_v = max(frequenze_v)

def u_v(t):
    return sum(15*np.sin(f*t + p) for f,p in zip(frequenze_v, fasi_v)) #come scegliere i valori dell'input?      


t = (0,30)
t_eval = np.linspace(*t, int(90*(max(max_f_t, max_f_v))/(2*np.pi))) #in questo caso campiono comunque con la pulsazione di risonanza del sistema linearizzato


soluzione = solve_ivp(pendolo_t, t, x_0_t, t_eval=t_eval)

data = np.column_stack((soluzione.t, soluzione.y[0], soluzione.y[1], u_t(soluzione.t)))
np.savetxt("simulazione.csv", data, delimiter=",",
           header="tempo,angolo,velocita angolare,input", comments="")


validazione = solve_ivp(pendolo_v, t, x_0_v, t_eval=t_eval)

data_v= np.column_stack((validazione.t, validazione.y[0], validazione.y[1], u_v(validazione.t)))
np.savetxt("validazione.csv", data_v, delimiter=",",
           header="tempo,angolo,velocita angolare,input", comments="")
