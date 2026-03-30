import numpy as np

data = np.loadtxt("simulazione.csv", delimiter=",", skiprows=1)

#Prelevo i dati dal file creato precedentemente con la simulazione

t = data[:, 0]
x1 = data[:, 1]
x2 = data[:, 2]

posizione = np.array(x1)
velocita = np.array(x2)

#matrice dei regressori
phi = np.column_stack((-velocita, -posizione))

print(phi)

#assumiamo output del sistema che sia l'accellerazione

#differenziale del tempo
dt = t[1] -t[0]

#Calcolo accelerazione
a = np.gradient(velocita, dt) 

y = a

theta, residuals, rank, s = np.linalg.lstsq(phi, y, rcond=None)

print("b/m:", theta[0])
print("k/m:", theta[1])

if len(theta) == 3:
    print("1/m:", theta[2])
    m = 1/theta[2]
    c = theta[0]*m
    k = theta[1]*m


