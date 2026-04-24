#System Identification ARX
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
import numpy as np
import pandas as pd
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.parameter_estimation import LeastSquares
import matplotlib.pyplot as plt
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
#Caricamento dei dati

data = np.loadtxt("simulazione.csv", delimiter=",", skiprows=1)

t = data[:, 0]
x1 = data[:, 1] #angolo
x2 = data[:, 2] #velocità angolare
u = data[:, 3]

y = x1

y = y.reshape(-1, 1)
u = u.reshape(-1, 1)

#Parametri del modello
estimator = LeastSquares()
model = FROLS(
    order_selection = True,
    info_criteria="aic",
    n_info_values= 10,
    n_terms = 4,
    ylag=2,      
    xlag=2,
    estimator = estimator,  
    basis_function=Polynomial(degree=3),
)

#Stima dei parametri

model.fit(X=u, y=y) 

#per scegliere n_terms
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")

r = pd.DataFrame(    
    results(        
    model.final_model, model.theta, model.err, 
    model.n_terms, err_precision=8, dtype='sci'
    ),    
    columns=['Regressors', 'Parameters', 'ERR'])  
print(r)



validazione = np.loadtxt("validazione.csv", delimiter=",",skiprows=1)
tc = validazione[:,0]
xc2 = validazione[:,2]
xc1 = validazione[:, 1]
uc = validazione[:,3]

yc = xc1
uc = uc.reshape(-1,1)
yc = yc.reshape(-1,1)

yhat = model.predict(X=uc, y=yc[:model.max_lag])


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




