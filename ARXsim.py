#System Identification ARX
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
import numpy as np
import pandas as pd
from sysidentpy.utils.display_results import results

def simula(model, y, u):
    theta = model.theta.flatten()
    final_model = model.final_model
    N=len(u)
    y_prev = np.zeros(N)
    
    # Inizializzao i primi termini con valori reali (na =1 )
    y_prev[0] = y[0].item()
    for k in range(1, N):
        y_prev[k] = 0
        for i, regr in enumerate(final_model):
            code = regr[0] #1XX per y, 2XX per u, 0 per bias
            
            if code==0:
                y_prev[k] += theta[i] 
            elif 1000 <= code < 2000 :
                na = code - 1000
                y_prev[k] += theta[i] * y_prev[k - na]   #simulazione free run (tende a divergere)
                #y_prev[k] += theta[i] * y[k -na].item() #simulazione one step 
            else:
                nb = code - 2000
                y_prev[k] += theta[i] * u[k - nb].item() 

    return y_prev

data = np.loadtxt("simulazione.csv", delimiter=",", skiprows=1)

t = data[:, 0]
x2 = data[:, 2] #velocità
u = data[:, 3]

# y = accellerazione

#dt = t[1] - t[0]
#y = np.gradient(x2, dt)

#y = velocità
y = x2

u = u.reshape(-1, 1)
y = y.reshape(-1, 1)



model = FROLS(
    ylag=1,
    xlag=1,
    basis_function=Polynomial(degree=1),
    n_info_values=3
)

model.fit(X=u, y=y) #stima dei parametri

r = pd.DataFrame(    
results(        
model.final_model, model.theta, model.err, 
model.n_terms, err_precision=8, dtype='sci'
),    
columns=['Regressors', 'Parameters', 'ERR'])
print(r)

y_prev = simula(model, y, u)
for i in range (0, 100):
    print(f"y previsto = {y_prev[i].item()}, y effettivo = {y[i].item()}") 


# "Validazione" del sistema con dati generati con input differente (randomico)

validazione = np.loadtxt("validazione.csv", delimiter=",",skiprows=1)
tc = validazione[:,0]
xc2 = validazione[:,2]
uc = validazione[:,3]

dtc = tc[1] - tc[0]
yc = np.gradient(xc2, dtc)
uc = uc.reshape(-1,1)
yc = yc.reshape(-1,1)
print()
print()
print("Validazione:\n")

yc_prev = simula(model, yc, uc)
for i in range (0, 100):
    print(f"y previsto = {yc_prev[i].item()}, y effettivo = {yc[i].item()}") 
    