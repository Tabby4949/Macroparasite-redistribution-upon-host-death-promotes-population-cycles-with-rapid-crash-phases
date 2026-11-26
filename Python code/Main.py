import numpy as np
import Models as Model
import Analysis as analysis
import Plots as plot

#%% Main paper: Figure 1

model = Model.Redist_4D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 1e-4
lamb = 40
r = None

r_vals = [0, 1]  

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r)

x, y = "alpha", "h"           
n = 500                   

x_vals = np.linspace(0, 2e-3, n)
y_vals = np.linspace(0, 100, n)

results_1 = analysis.bifurcation_analysis(model, x, y, x_vals, y_vals, r_vals, params)

label_x_pos = 1.9e-3
label_y_pos = 95
plot.boundary_plot(x, y, x_vals, y_vals, r_vals, results_1, label_x_pos, label_y_pos)

#%% Main paper: Figure 2(a)

model = Model.Redist_4D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 1e-4
lamb = 40
r = None

r_vals = [0, 0.25, 0.5, 0.75, 1]  

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r)

x, y = "alpha", "h"                      
n = 100                                  

x_vals = np.linspace(0, 2e-3, n)         
y_vals = np.linspace(0, 100, n)

results_2a = analysis.bifurcation_analysis(model, x, y, x_vals, y_vals, r_vals, params)

r_label_pos = {0.00: 0.5, 0.25: 0.6, 0.50: 0.7, 0.75: 0.7, 1.00: 0.9}
plot.plot_Hopf_comparison(x, y, x_vals, y_vals, r_vals, results_2a, r_label_pos)


#%% Main paper: Figure 2(b)

model = Model.Redist_4D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 1e-4
lamb = 40
r = None

r_vals = [0, 0.5, 1]  

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r)

x, y = "b", "d"                      
n = 100                              

x_vals = np.linspace(0, 3, n)         
y_vals = np.linspace(0, 2, n)

results_2b = analysis.bifurcation_analysis(model, x, y, x_vals, y_vals, r_vals, params)

r_label_pos = {0.00: 0.25, 0.50: 0.35, 1.00: 0.45}
plot.plot_Hopf_comparison(x, y, x_vals, y_vals, r_vals, results_2b, r_label_pos)


#%% Main paper: Figure 2(c)

model = Model.Redist_4D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 1e-4
lamb = 40
r = None

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r)

x, y = "alpha", "h"                      
n = 600

x_vals = np.linspace(0, 2e-3, n)         
y_vals = np.linspace(0, 300, n)

z_vals = {"nu": [1e-3, 1e-4, 1e-5]}
results_2c = analysis.vary_Hopf_analysis(model, x, y, x_vals, y_vals, z_vals, params)

label_pos = {1e-3: 0.6, 1e-4: 0.5, 1e-5: 0.5}
colours = ["green", "blue", "magenta"]
plot.vary_Hopf_plot(x, y, x_vals, y_vals, z_vals, results_2c, colours, label_pos, r_plot=None)    # use r_plot=0 or r_plot=1 to view the two cases seperately


#%% Appendix: Figure A1

model = Model.Redist_4D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 0
lamb = 40
r = None

r_vals = [0, 1]  

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r)

x, y = "alpha", "h"           
n = 600                   

x_vals = np.linspace(0, 2e-3, n)
y_vals = np.linspace(0, 3000, n)

results_S1 = analysis.bifurcation_analysis(model, x, y, x_vals, y_vals, r_vals, params)

label_x_pos = 1.9e-3
label_y_pos = 2850
plot.boundary_plot(x, y, x_vals, y_vals, r_vals, results_S1, label_x_pos, label_y_pos)


#%% Appendix: Figure A2

model = Model.Redist_4D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 1e-4
lamb = 40
r = None

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r)

x, y = "alpha", "h"                      
n = 100 

x_vals = np.linspace(0, 5e-3, n)         
y_vals = np.linspace(0, 200, n)

z_vals = {"lamb": [0, 40, 200, 500]}
results_S2 = analysis.vary_Hopf_analysis(model, x, y, x_vals, y_vals, z_vals, params)

label_pos = {0: 0.95, 40: 0.8, 200: 0.55, 500: 0.5}
colours = ["orange", "blue", "magenta", "green"]
plot.vary_Hopf_plot(x, y, x_vals, y_vals, z_vals, results_S2, colours, label_pos, r_plot=None)       # use r_plot=0 or r_plot=1 to view the two cases seperately

#%% Appendix: Figure A3

model = Model.Redist_3D

b = 0.6
d = 0.5
beta = 0.12
psi = 0.5
sigma = 4
alpha = 1.8e-3
h = 100
delta = 60
nu = 1e-4
xi = 1e-4
k = 2.05
r = None

r_vals = [0, 1]  

params = (d, beta, psi, sigma, b, nu, xi, alpha, h, delta, k, r)

x, y = "alpha", "h"           
n = 500                   

x_vals = np.linspace(0, 1.5e-4, n)
y_vals = np.linspace(0, 100, n)

results_S3 = analysis.bifurcation_analysis(model, x, y, x_vals, y_vals, r_vals, params)

label_x_pos = 3.5e-5
label_y_pos = 95
plot.boundary_plot(x, y, x_vals, y_vals, r_vals, results_S3, label_x_pos, label_y_pos)

