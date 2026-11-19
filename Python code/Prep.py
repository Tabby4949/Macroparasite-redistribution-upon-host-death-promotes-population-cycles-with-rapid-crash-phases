import numpy as np
import Models as Model
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# ======================================================================
# Identify the parameters being updated and return the new parameter set
# ======================================================================

def params_ID(i, j=None, i_val=None, j_val=None, model=None, params=None):
    
    if model==Model.Redist_4D:
        parameter_names = {
            "d": 0,
            "beta": 1,
            "psi": 2,
            "sigma": 3,
            "b": 4,
            "nu": 5,
            "xi": 6,
            "alpha": 7,
            "h": 8,
            "delta": 9,
            "lamb": 10,
            "r": 11,
        }

    else:
        
        parameter_names = {
            "d": 0,
            "beta": 1,
            "psi": 2,
            "sigma": 3,
            "b": 4,
            "nu": 5,
            "xi": 6,
            "alpha": 7,
            "h": 8,
            "delta": 9,
            "k": 10,
            "r": 11,
        }

    params_new = list(params)

    params_new[parameter_names[i]] = i_val

    if j is not None:
        params_new[parameter_names[j]] = j_val

    return tuple(params_new)

# =============================================================================================================================================================
# Find an initial state for the systems by using the analytic (for Redist_3D) and semi analytical (for Redist_4D) equilibrium of the models when the birth rate
# of hosts is b - xi*x, instead of the current versions of b*(k/(xi*(P/H) + k))**k and b*(1 + (A - 1)*xi)**(- x/(A - 1) 
# =============================================================================================================================================================

def initial_state(model, params, num_zero):
    
    if model == Model.Redist_3D:
        d, beta, psi, sigma, b, nu, xi, alpha, h, delta, k, r = params
        lamb = None
    else:
        d, beta, psi, sigma, b, nu, xi, alpha, h, delta, lamb, r = params
        k = None
    
    
    if b <= d:
        raise ValueError("No endemic equilibrium possible because b <= d.")
        
    # ===========================
    # Initial guess for Redist_3D
    # ===========================   

    if model == Model.Redist_3D:   
        
        H_max = (b - d) / nu   # carry capacity of hosts
        
        if H_max <= 0:
            return None   # negative values break biological constraints so return no values

        def F_of_H(H):   # root of F(H)=0 gives endemic host population
            P = H * (b - d - nu * H) / (alpha + xi)   # Endemic macroparasite population
            numerator_L = h * P + r * P * (d + nu * H + alpha * (1 + ((k + 1) / k) * (P / H)))
            L = numerator_L / (delta + beta * H)   # endemic free-living-stage population
            return beta * psi * L * H - (sigma + alpha + d + nu * H) * P - alpha * ((P ** 2) / H) * ((k + 1) / k)   # F(H) (which is the governing equation of P)

        H_grid = np.linspace(1e-5, H_max * 0.999, 400)   # grid of possiple host steady state
        F_grid = np.array([F_of_H(H) for H in H_grid])   # evaluate F(H)=0 at all points in in H_grid
        sign_changes = np.where(np.diff(np.sign(F_grid)))[0]   # search for points where F(H_i) and F(H_(i+1)) change signs to identify roots
        
        for idx in sign_changes:
            try:
                H0 = brentq(F_of_H, H_grid[idx], H_grid[idx+1])   # uses the root finder brentq in between the two identified points of signage change
                P0 = H0 * (b - d - nu * H0) / (alpha + xi)
                L0 = L0 = (h * P0 + r * P0 * (d + nu * H0 + alpha * (1 + ((k + 1) / k) * (P0 / H0)))) / (delta + beta * H0)  # update P* and L* with H*
                if P0 > 0 and L0 > 0:
                    return [H0, P0, L0]
            except ValueError:
                continue
        return None
    
    # ===========================
    # Initial guess for Redist_4D
    # ===========================  
    
    else:
        
        def A_root(x):

            a = 2*alpha
            B = -(alpha*(2 + lamb) + xi*x - 2*sigma - b)
            c = (x*(xi*(1 + lamb + x) - b) - b*(1 + lamb) - sigma*(2 + lamb))

            roots = np.roots([a, B, c])

            real_roots = [root.real for root in roots if abs(root.imag) < num_zero and root.real > 1 + num_zero]     # Keep real roots above the physical boundary A > 1

            if len(real_roots) == 0:
                raise ValueError("No physical A equilibrium found.")

            return min(real_roots)    # If there are two physical roots, use the smaller one

        def ana_H(x):
            return (b - d - (alpha + xi)*x) / nu

        def ana_L(A, x):
            return (x/(beta*psi) * (b + sigma + alpha*A - xi*x))

        def L_govEq(x):

            A = A_root(x)
            H = ana_H(x)
            L = ana_L(A, x)

            value = (h*x*H - delta*L - beta*L*H + r*x*H*(d + nu*H + alpha*(A + x)))

            if not np.isfinite(value):
                raise ValueError("Non-finite equilibrium equation.")

            return value


        x_upper = (b - d)/(alpha + xi)

        x_grid = np.linspace(num_zero, x_upper*(1 - 1e-10), 500)

        f_values = []

        for x_test in x_grid:

            try:
                f_test = L_govEq(x_test)

                if np.isfinite(f_test):
                    f_values.append((x_test, f_test))

            except (ValueError, RuntimeError, FloatingPointError):
                continue

        if len(f_values) < 2:
            raise ValueError("Could not find a valid bracket for x equilibrium.")

        x0 = None

        for k in range(len(f_values) - 1):

            x_left, f_left = f_values[k]
            x_right, f_right = f_values[k + 1]

            if f_left == 0:
                x0 = x_left
                break

            if f_left * f_right < 0:
                x0 = brentq(L_govEq, x_left, x_right)

                break

        if x0 is None:
            raise ValueError("No endemic x equilibrium found.")

        A0 = A_root(x0)
        H0 = ana_H(x0)
        L0 = ana_L(A0, x0)

        equilibrium = np.array([H0, x0, A0, L0])

        if (not np.all(np.isfinite(equilibrium)) or np.any(equilibrium <= 0)):
            raise ValueError("Computed endemic initial state is not physical.")

        return equilibrium

# =======================================================================================
# Search for the first endemic steady state in the paramter space to start the sweep from
# =======================================================================================

def starting_coordinate(x, y, i_vals, j_vals, model, params, num_zero):

    n_x = len(i_vals)
    n_y = len(j_vals)

    print("\n" + "="*70)
    print("SEARCHING FOR AN ENDEMIC STARTING EQUILIBRIUM")
    print("="*70)

    tested = set()

    for level in range(max(n_x, n_y)):

        if level < n_y:     # Row search

            j = level

            print(f"\nSearching row {level + 1}/{n_y}: "f"{y} = {j_vals[j]:.12g}")

            for i in range(n_x):
                if (i, j) in tested:
                    continue

                tested.add((i, j))

                i_val = i_vals[i]
                j_val = j_vals[j]

                params_point = params_ID(x, y, i_val, j_val, model, params)

                print(f"Testing {x} = {i_val:.12g}, "f"{y} = {j_val:.12g}")

                try:
                    initial = initial_state(model, params_point, num_zero)

                except (ValueError, RuntimeError, FloatingPointError):

                    continue

                equilibrium = Model.Equilibrium(model, params_point, initial_guess=initial, num_zero = num_zero)
                equilibrium = Model.validate_equilibrium(model, equilibrium, params_point, num_zero)

                if equilibrium is None:

                    try:
                        solution = solve_ivp(model, (0, 100), initial, args=(params_point,), rtol=1e-6, atol=1e-8, max_step=0.1)

                        if solution.success:
                            equilibrium = Model.Equilibrium(model, params_point, initial_guess=solution.y[:, -1], num_zero = num_zero)
                            equilibrium = Model.validate_equilibrium(model, equilibrium, params_point, num_zero)

                    except Exception:

                        equilibrium = None

                if equilibrium is None:
                    continue

                equilibrium = np.asarray(equilibrium, dtype=float)

                if (np.all(np.isfinite(equilibrium)) and np.all(equilibrium > 0)):

                    print("\n" + "="*70)
                    print("VALID ENDEMIC STARTING EQUILIBRIUM FOUND")
                    print("="*70)

                    print(f"{x} = {i_val:.12g}")

                    print(f"{y} = {j_val:.12g}")

                    if model == Model.Redist_3D:
                        print(f"H* = {equilibrium[0]:.12g}, "f"P* = {equilibrium[1]:.12g}, "f"L* = {equilibrium[2]:.12g}")
                    else:
                        print(f"H* = {equilibrium[0]:.12g}, "f"x* = {equilibrium[1]:.12g}, "f"A* = {equilibrium[2]:.12g}, "f"L* = {equilibrium[3]:.12g}")

                    return [{x: i_val, y: j_val, "params": params_point, "equilibrium": equilibrium, "type": "endemic"}]

        if level < n_x:     # Column search

            i = level

            print(f"\nSearching column {level + 1}/{n_x}: "f"{x} = {i_vals[i]:.12g}")

            for j in range(n_y):

                if (i, j) in tested:
                    continue

                tested.add((i, j))

                i_val = i_vals[i]
                j_val = j_vals[j]

                params_point = params_ID(x, y, i_val, j_val, model, params)

                print(f"Testing {x} = {i_val:.12g}, "f"{y} = {j_val:.12g}")

                try:
                    initial = initial_state(model, params_point, num_zero)

                except (ValueError, RuntimeError, FloatingPointError):

                    continue

                equilibrium = Model.Equilibrium(model, params_point, initial_guess=initial, num_zero = num_zero)
                equilibrium = Model.validate_equilibrium(model, equilibrium, params_point, num_zero)

                if equilibrium is None:

                    try:
                        solution = solve_ivp(model, (0, 100), initial, args=(params_point,), rtol=1e-6, atol=1e-8, max_step=0.1)

                        if solution.success:
                            equilibrium = Model.Equilibrium(model, params_point, initial_guess=solution.y[:, -1], num_zero = num_zero)
                            equilibrium = Model.validate_equilibrium(model, equilibrium, params_point, num_zero)

                    except Exception:

                        equilibrium = None

                if equilibrium is None:
                    continue

                equilibrium = np.asarray(equilibrium, dtype=float)

                if (np.all(np.isfinite(equilibrium)) and np.all(equilibrium > 0)):

                    print("\n" + "="*70)
                    print("VALID ENDEMIC STARTING EQUILIBRIUM FOUND")
                    print("="*70)

                    print(f"{x} = {i_val:.12g}")

                    print(f"{y} = {j_val:.12g}")

                    if model == Model.Redist_3D:
                        print(f"H* = {equilibrium[0]:.12g}, "f"P* = {equilibrium[1]:.12g}, "f"L* = {equilibrium[2]:.12g}")
                    else:
                        print(f"H* = {equilibrium[0]:.12g}, "f"x* = {equilibrium[1]:.12g}, "f"A* = {equilibrium[2]:.12g}, "f"L* = {equilibrium[3]:.12g}")

                    return [{x: i_val, y: j_val, "params": params_point, "equilibrium": equilibrium, "type": "endemic"}]

    print("\n" + "="*70)
    print("NO ENDEMIC EQUILIBRIUM FOUND")
    print("="*70)

    return []