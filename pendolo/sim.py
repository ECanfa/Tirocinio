import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math as m
import random

#Sistema in forma di stato
# x1 angolo, x2 velocità angolare
# dx1dt = x2 
# dx2dt = -g/l*sin(x1) - b/ml^2*x2 + 1/ml^2 u

gg = 9.81
bb = 2
mm = 0.5
ll = 1

#Condizioni iniziali
x_1_0 = np.pi/4#in radianti
x_2_0 = 0.005
x_0 = [x_1_0,x_2_0]

t = (0,20)
omega_n = m.sqrt(gg/ll)
xi = bb/(2*mm*ll*ll*omega_n)
print(xi)
omega_ris = omega_n* m.sqrt(1-2*xi*xi)
t_eval = np.linspace(*t, int(60*(max(2*omega_ris,15*np.pi))/(2*np.pi))) #in questo caso campiono comunque con la pulsazione di risonanza del sistema linearizzato
    

def pendolo_t(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(gg/ll)*np.sin(x1) - (bb/(mm*ll*ll))*x2 + (1/(mm*ll*ll))*u_t(t)
    return [dx1dt, dx2dt]



def u_t(t):
    moltiplicatori = [0.2, 0.5, 1.0, 1.5, 2.0]
    
    return (sum(np.sin(i*omega_ris*t) for i in moltiplicatori)) #uso un input intorno alla frequenza di risonanza perchè è dove il sistema è maggiormente stimolato 

     

def pendolo_v(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = -(gg/ll)*np.sin(x1) - (bb/(mm*ll*ll))*x2 + (1/(mm*ll*ll))*u_v(t)
    return [dx1dt, dx2dt]



def u_v(t):
    frequenze_v = np.linspace(0.15, 5.2, 15)
    return sum(0.4*np.sin(np.pi*f*t) for f in frequenze_v) #come scegliere i valori dell'input? 
    


soluzione = solve_ivp(pendolo_t, t, x_0, t_eval=t_eval)

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
           header="tempo,angolo,velocita angolare,input", comments="")


validazione = solve_ivp(pendolo_v, t, x_0, t_eval=t_eval)

data_v= np.column_stack((validazione.t, validazione.y[0], validazione.y[1], u_v(validazione.t)))
np.savetxt("validazione.csv", data_v, delimiter=",",
           header="tempo,angolo,velocita angolare,input", comments="")
