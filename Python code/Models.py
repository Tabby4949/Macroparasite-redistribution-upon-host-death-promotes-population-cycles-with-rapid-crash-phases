import numpy as np
import jax.numpy as jnp
from scipy.optimize import fsolve

# ====================================================
# Equations S1-S3 model from supplementary information
# ====================================================

def Redist_3D(t, state, parameters):
    
    d, beta, psi, sigma, b, nu, xi, alpha, h, delta, k, r = parameters
    H, P, L = state
    
    dH = H*(b*(k/(xi*(P/H) + k))**k - d - nu*H) - alpha*P
    dP = beta*psi*L*H - P*(sigma + d + nu*H + alpha*(1 + ((k + 1)/k)*(P/H)))
    dL = h*P - delta*L - beta*H*L + r*P*(d + nu*H + alpha*(1 + ((k + 1)/k)*(P/H)))  
    
    return jnp.array([dH, dP, dL])

# ===============================================
# Equilbrium of Redist_3D calculated using fsolve
# ===============================================

def Redist_3D_Equilibrium(parameters, initial_guess, num_zero):
    
    d, beta, psi, sigma, b, nu, xi, alpha, h, delta, k, r = parameters
    
    def equations(state):
        H, P, L = state
        
        if abs(H) <= num_zero:
            H = num_zero
        
        x = P/H
        
        if abs(x) <= num_zero:
            x = num_zero
        
        dH = H*(b*(k/(xi*x + k))**k - d - nu*H) - alpha*P
        dP = beta*psi*L*H - P*(sigma + d + nu*H + alpha*(1 + ((k + 1)/k)*x))
        dL = h*P - delta*L - beta*H*L + r*P*(d + nu*H + alpha*(1 + ((k + 1)/k)*(P/H)))  
        
        return [dH, dP, dL]
    
    try:
        equilibrium_state, info, ier, mesg = fsolve(equations, initial_guess, full_output=True)

        if ier != 1:
            return None
        
        H_star, P_star, L_star = equilibrium_state
        
        if not np.all(np.isfinite(equilibrium_state)):
            return None

        if np.any(equilibrium_state < -num_zero):
            return None

        if H_star <= num_zero:
            return None

        if P_star <= num_zero:
            return None

        if L_star <= num_zero:
            return None

        return H_star, P_star, L_star

    except Exception as e:
        return None
    
# ==============================================================================================================
# Test given equilbrium by imputting it into governing equations and compare the residual of answer against zero
# ==============================================================================================================
    
def Redist_3D_equilibrium_residual(equilibrium, parameters, num_zero):

    d, beta, psi, sigma, b, nu, xi, alpha, h, delta, k, r = parameters
    H, P, L = equilibrium

    if abs(H) <= num_zero:
        H = num_zero

    x = P/H

    if abs(x) <= num_zero:
        x = num_zero

    dH = H*(b*(k/(xi*x + k))**k - d - nu*H) - alpha*P
    dP = beta*psi*L*H - P*(sigma + d + nu*H + alpha*(1 + ((k + 1)/k)*x))
    dL = h*P - delta*L - beta*H*L + r*P*(d + nu*H + alpha*(1 + ((k + 1)/k)*(P/H)))  

    return np.array([dH, dP, dL])

# =========
# Model (2)
# =========

def Redist_4D(t, state, parameters):
    
    d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r, = parameters
    H, x, A, L = state
    
    dH = H*(b*(1 + (A - 1)*xi)**(- x/(A - 1)) - d - nu*H - alpha*x)
    dx = beta*psi*L - x*(b*(1 + (A - 1)*xi)**(- x/(A - 1)) + sigma + alpha*A)
    dA = (1 - A)*(sigma + alpha*A + (beta*psi*L/x)) + b*x*(1 + (A - 1)*xi)**(-x/(A - 1)) + (beta*psi*L*lamb)/x
    dL = h*x*H - delta*L - beta*L*H + r*x*H*(d + nu*H + alpha*(A + x))

    return jnp.array([dH, dx, dA, dL])

# ===============================================
# Equilbrium of Redist_4D calculated using fsolve
# ===============================================

def Redist_4D_Equilibrium(parameters, initial_guess, num_zero):

    d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r = parameters

    def equations(state):
        H, x, A, L = state

        if abs(x) <= num_zero:
            x = num_zero
            
        if abs(A) <= 1 + num_zero:
            A = 1 + num_zero
        
        dH = H*(b*(1 + (A - 1)*xi)**(- x/(A - 1)) - d - nu*H - alpha*x)
        dx = beta*psi*L - x*(b*(1 + (A - 1)*xi)**(- x/(A - 1)) + sigma + alpha*A)
        dA = (1 - A)*(sigma + alpha*A + (beta*psi*L/x)) + b*x*(1 + (A - 1)*xi)**(-x/(A - 1)) + (beta*psi*L*lamb)/x
        dL = h*x*H - delta*L - beta*L*H + r*x*H*(d + nu*H + alpha*(A + x))

        return [dH, dx, dA, dL]

    try:
        equilibrium_state, info, ier, mesg = fsolve(equations, initial_guess, full_output=True)

        if ier != 1:
            return None
        
        H_star, x_star, A_star, L_star = equilibrium_state
        
        if not np.all(np.isfinite(equilibrium_state)):
            return None

        if np.any(equilibrium_state < -num_zero):
            return None

        if x_star <= num_zero:
            return None

        if L_star <= num_zero:
            return None

        if A_star <= 1 + num_zero:
            return None

        return H_star, x_star, A_star, L_star

    except Exception as e:
        return None

# ==============================================================================================================
# Test given equilbrium by imputting it into governing equations and compare the residual of answer against zero
# ==============================================================================================================

def Redist_4D_equilibrium_residual(equilibrium, parameters, num_zero):

    d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r = parameters
    H, x, A, L = equilibrium

    if abs(x) <= num_zero:
        x = num_zero

    if abs(A) <= 1 + num_zero:
        A = 1 + num_zero

    dH = H*(b*(1 + (A - 1)*xi)**(- x/(A - 1)) - d - nu*H - alpha*x)
    dx = beta*psi*L - x*(b*(1 + (A - 1)*xi)**(- x/(A - 1)) + sigma + alpha*A)
    dA = (1 - A)*(sigma + alpha*A + (beta*psi*L/x)) + b*x*(1 + (A - 1)*xi)**(-x/(A - 1)) + (beta*psi*L*lamb)/x
    dL = h*x*H - delta*L - beta*L*H + r*x*H*(d + nu*H + alpha*(A + x))

    return np.array([dH, dx, dA, dL])

# =============================================
# Test a given equilbrium to see if it is valid
# =============================================

def validate_equilibrium(model, equilibrium, parameters, num_zero, tolerance=1e-7):

    if equilibrium is None:
        return None

    equilibrium = np.asarray(equilibrium, dtype=float)

    if not np.all(np.isfinite(equilibrium)):
        return None

    if np.any(equilibrium < -num_zero):
        return None

    try:
        if model == Redist_3D:
            residual = Redist_3D_equilibrium_residual(equilibrium, parameters, num_zero)
        else:
            residual = Redist_4D_equilibrium_residual(equilibrium, parameters, num_zero)
    except Exception:
        return None

    if not np.all(np.isfinite(residual)):
        return None

    if np.max(np.abs(residual)) > tolerance:
        return None

    return equilibrium

# ============================
# Retrieve desired equilibrium
# ============================

def Equilibrium(model, parameters, initial_guess, num_zero):
    if model == Redist_3D:
        return Redist_3D_Equilibrium(parameters, initial_guess, num_zero)
    else:
        return Redist_4D_Equilibrium(parameters, initial_guess, num_zero)