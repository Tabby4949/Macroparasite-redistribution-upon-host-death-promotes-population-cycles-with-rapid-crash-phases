# Code for figures for the paper 'Macroparasite redistribution upon host death promotes population cycles with rapid crash phases'
This repository contains Python and MATLAB code for the results obtained in the paper 'Macroparasite redistribution upon host death promotes population cycles with rapid crash phases'.

The Python code folder contains the following Python files:

- Models.py: Here, the two redistribution models (Model (2) in the main paper and Equations S1-S3 in the supplementary information) are implemented, alongside their equilibria, which were calculated using the fsolve wrapper from the SciPy optimisation library.
- Prep.py: This contains the following:
  - An implementation that identifies any parameters being changed and returns the new/updated parameter set
  - Code that calculates intitial states ([H(0), x(0), A(0), L(0)] or [H(0), P(0), L(0)]) for the simulations of the models. Due to the sensitivity of the models, it was necessary to find initial conditions that were close to (but not on) an endemic steady state (if one exists for the parameter set). This was achieved by altering the law of the birth rate of hosts from b(1+(A-1)xi)^(-x/(A-1)to b-xi*x in the model. This simplification allows the equilibrium to be obtained semi-analytically: H, A, and L can be expressed as functions of x, which is determined using Brent's root-finding algorithm (brentq). For Equations S1-S3, changing b(k/(xi(P/H)+k))^k to b-xi(P/H) results in an equilibrium that can be calculated analytically. See [1, 2] for models that incorporate the simpler birth rate.
  - The implementation to find the first endemic steady state in a given parameter space to start the sweep for the bifurcation analysis. It was found that starting the sweep from an existing endemic steady state (regardless of stability) was more reliable than starting from a coordinate which was parasite-free; while it was possible to successfully sweep from a parasite-free steady state, it was inconsistent and could result in neighbouring coordinates falling back to the trivial parasite-free state instead of converging to the endemic state.
- Analysis.py: There are two analysis methods present in this file:
  - bifurcation_analysis: This performs a two-parameter bifurcation analysis across the (x,y) grid for each specified value of r by iterating through the parameter space, attempting to find the endemic steady state via analytical guesses, neighbouring cached states or an ODE integration before using Jacobians and eigenvalues to determine stability.
  - vary_Hopf_analysis: Here, the Hopf bifurcation curve is extracted for values of r=0 and r=1 in a (x,y) parameter space when you vary a parameter z.
- Plots.py: This contains all the plotting code that shows the results obtained from the analysis.
- Main.py: Within this file is the specific code for Figures 1, 2(a), 2(b), 2(c), S1, S2 and S3, which the user can choose to obtain.

The MATLAB code folder contains the simulations for Figures 4 and 5.

[1] https://www.jstor.org/stable/3933

[2] https://www.jstor.org/stable/5339?origin=crossref&seq=1
