import numpy as np
import Models as Model
import Prep as prep
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import jax
import jax.numpy as jnp

# =======================
# Define a numerical zero
# =======================

def num_zero():
    return 1e-7

# ==================================
# Two-parameter bifurcation analysis
# ==================================

def bifurcation_analysis(model, x, y, x_vals, y_vals, r_vals, params):
    
    num_zero_value = num_zero()
    
    results = {}
    
    for r in r_vals:
        params_r = prep.params_ID("r", i_val=r, model=model, params=params)
        
        starter_equilibria = prep.starting_coordinate(x, y, x_vals, y_vals, model, params_r, num_zero_value)

        if len(starter_equilibria) == 0:
            raise ValueError(f"No viable endemic equilibrium was found for r = {r}.")

        start_coordinate = starter_equilibria[0]

        x_start = np.argmin(np.abs(x_vals - start_coordinate[x]))
        y_start = np.argmin(np.abs(y_vals - start_coordinate[y]))

        print("\n" + "=" * 70)
        print("STARTING FULL PARAMETER SWEEP")
        print("=" * 70)
        print(f"r = {r}")
        print(f"Starting point: {x} = {x_vals[x_start]:.12g}, "f"{y} = {y_vals[y_start]:.12g}")
        print(f"Total parameter combinations: "f"{len(x_vals) * len(y_vals)}")

        if model == Model.Redist_3D:
            state_dimension = 3
        else:
            state_dimension = 4

        phase = np.full((len(x_vals), len(y_vals)), np.nan)
        hopf_indicator = np.full((len(x_vals), len(y_vals)), np.nan)
        imag_indicator = np.full((len(x_vals), len(y_vals)), np.nan)

        eq_cache = np.full((len(x_vals), len(y_vals), state_dimension), np.nan)

        eps_boundary = 0
        eps_eig = 0

        def Model_state(state, params):
            return model(0, state, params)

        jax_jacobian = jax.jit(jax.jacfwd(Model_state))

        eq_cache[x_start, y_start] = start_coordinate["equilibrium"]

        # ================================
        # Order points from starting point
        # ================================

        points = []

        for j in range(len(y_vals)):
            for i in range(len(x_vals)):

                distance = max(abs(i - x_start), abs(j - y_start))
                points.append((distance, i, j))

        points.sort(key=lambda z: (z[0], z[2], z[1]))

        total_points = len(points)
        analysed = 0

        # ==========
        # Main sweep
        # ==========

        for distance, i, j in points:
            x_val = x_vals[i]
            y_val = y_vals[j]

            params_update = prep.params_ID(x, y, x_val, y_val, model, params_r)
            equilibrium = None

            # ==================================
            # Try analytical endemic equilibrium
            # ==================================

            initial = prep.initial_state(model, params_update, num_zero_value)

            if initial is not None:
                equilibrium = Model.Equilibrium(model, params_update, initial, num_zero_value)

            if equilibrium is not None:
                equilibrium = np.asarray(equilibrium, dtype=float)

                if model == Model.Redist_3D:
                    valid = (np.all(np.isfinite(equilibrium)) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > eps_boundary)

                else:
                    valid = (np.all(np.isfinite(equilibrium)) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > 1 + eps_boundary and equilibrium[3] > eps_boundary)

                if not valid:
                    equilibrium = None

            # ==========================================
            # Try neighbouring cached endemic equilibria
            # ==========================================

            if equilibrium is None:
                neighbour_indices = [
                    (i - 1, j),
                    (i + 1, j),
                    (i, j - 1),
                    (i, j + 1),
                    (i - 1, j - 1),
                    (i - 1, j + 1),
                    (i + 1, j - 1),
                    (i + 1, j + 1)
                ]

                candidate_guesses = []

                for ni, nj in neighbour_indices:
                    if (0 <= ni < len(x_vals) and 0 <= nj < len(y_vals)):
                        neighbour = eq_cache[ni, nj]

                        if model == Model.Redist_3D:
                            valid_neighbour = (np.all(np.isfinite(neighbour)) and neighbour[0] > eps_boundary and neighbour[1] > eps_boundary and neighbour[2] > eps_boundary)

                        else:
                            valid_neighbour = (np.all(np.isfinite(neighbour)) and neighbour[0] > eps_boundary and neighbour[1] > eps_boundary and neighbour[2] > 1 + eps_boundary and neighbour[3] > eps_boundary)

                        if valid_neighbour:
                            candidate_guesses.append(neighbour)

                for initial_guess in candidate_guesses:
                    
                    equilibrium = Model.Equilibrium(model, params_update, initial_guess, num_zero_value)

                        equilibrium = None

                    if equilibrium is None:
                        continue

                    equilibrium = np.asarray(equilibrium, dtype=float)

                    if model == Model.Redist_3D:
                        valid = (np.all(np.isfinite(equilibrium)) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > eps_boundary)

                    else:
                        valid = (np.all(np.isfinite(equilibrium)) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > 1 + eps_boundary and equilibrium[3] > eps_boundary)

                    if valid:
                        break

                    equilibrium = None

            # ============================================
            # Try ODE integration if equilibrium not found
            # ============================================

            if equilibrium is None:
                
                initial = prep.initial_state(model, params_update, num_zero_value)

                if initial is not None:
                    solution = solve_ivp(model, (0, 100), initial, args=(params_update,), rtol=1e-6, atol=1e-8, max_step=0.1)

                    if solution.success:
                        equilibrium = Model.Equilibrium(model, params_update, initial_guess=solution.y[:, -1], num_zero=num_zero_value)

                        if equilibrium is not None:
                            equilibrium = np.asarray(equilibrium, dtype=float)

                            if model == Model.Redist_3D:
                                valid = (np.all(np.isfinite(equilibrium)) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > eps_boundary)

                            else:
                                valid = (np.all(np.isfinite(equilibrium)) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > 1 + eps_boundary and equilibrium[3] > eps_boundary)

                            if not valid:
                                equilibrium = None

            # ============================================
            # Test DFE if no endemic equilibrium was found
            # ============================================

            if equilibrium is None:

                d_corner = params_update[0]
                b_corner = params_update[4]
                nu_corner = params_update[5]

                if b_corner <= d_corner:
                    phase[i, j] = 2

                else:
                    DFE_H = (b_corner - d_corner) / nu_corner

                    if model == Model.Redist_3D:
                        dfe_state = np.array([DFE_H, num_zero_value, 0.0])

                    else:
                        dfe_state = np.array([DFE_H, num_zero_value, 1.0 + num_zero_value, 0.0])

                    J_dfe = np.asarray(jax_jacobian(jnp.asarray(dfe_state), params_update))

                    eigs_dfe = np.linalg.eigvals(J_dfe)
                    max_real_dfe = np.max(np.real(eigs_dfe))

                    if max_real_dfe < -eps_eig:
                        phase[i, j] = 3
                        eq_cache[i, j] = dfe_state

                    else:
                        phase[i, j] = 4 # no endemic and DFE unstable

                analysed += 1

                if analysed % 50 == 0:
                    print(f"r = {r}: Progress: "f"{analysed}/{total_points} "f"({100 * analysed / total_points:.2f}%)")

                continue

            equilibrium = np.asarray(equilibrium, dtype=float)
            eq_cache[i, j] = equilibrium

            # ===============================================
            # Jacobian and eigenvalues of endemic equilibrium
            # ===============================================

            J_jax = np.asarray(jax_jacobian(jnp.asarray(equilibrium), params_update))
            eigs = np.linalg.eigvals(J_jax)

            # =========================
            # Eigenvalue classification
            # =========================

            real_parts = np.real(eigs)
            imag_parts = np.abs(np.imag(eigs))

            idx = np.argmax(real_parts)

            max_real = real_parts[idx]

            hopf_indicator[i, j] = max_real
            imag_indicator[i, j] = imag_parts[idx]

            if max_real < -eps_eig:   # Endemic equilibrium is stable, test DFE

                d_corner = params_update[0]
                b_corner = params_update[4]
                nu_corner = params_update[5]

                DFE_H = (b_corner - d_corner) / nu_corner

                if model == Model.Redist_3D:
                    dfe_state = np.array([DFE_H, num_zero_value, 0.0])
                else:
                    dfe_state = np.array([DFE_H, num_zero_value, 1.0 + num_zero_value, 0.0])

                J_dfe = np.asarray(jax_jacobian(jnp.asarray(dfe_state), params_update))
                eigs_dfe = np.linalg.eigvals(J_dfe)
                max_real_dfe = np.max(np.real(eigs_dfe))

                dfe_stable = max_real_dfe < -eps_eig

                if dfe_stable:
                    phase[i, j] = 0 # bistablity but class it as endemic (region will appear for lower numbers of data points)
                else:
                    phase[i, j] = 0

            # =================================================================
            # If endemic equilibrium is unstable, test the stability of the DFE
            # =================================================================

            else:

                d_corner = params_update[0]
                b_corner = params_update[4]
                nu_corner = params_update[5]

                DFE_H = (b_corner - d_corner) / nu_corner

                if model == Model.Redist_3D:
                    dfe_state = np.array([DFE_H, num_zero_value, 0.0])
                else:
                    dfe_state = np.array([DFE_H, num_zero_value, 1.0 + num_zero_value, 0.0])

                J_dfe = np.asarray(jax_jacobian(jnp.asarray(dfe_state), params_update))
                eigs_dfe = np.linalg.eigvals(J_dfe)
                max_real_dfe = np.max(np.real(eigs_dfe))

                dfe_stable = max_real_dfe < -eps_eig

                if np.max(imag_parts) > num_zero_value:
                    if dfe_stable:
                        phase[i, j] = 3
                    else:
                        phase[i, j] = 1

                else:
                    if dfe_stable:
                        phase[i, j] = 3
                    else:
                        phase[i, j] = 6 # both unstable and no limit cycle

            analysed += 1

            if analysed % 50 == 0:

                print(f"r = {r}: Progress: "f"{analysed}/{total_points} "f"({100 * analysed / total_points:.2f}%)")

        # ================
        # Hopf and results
        # ================

        hopf_mask = np.where(imag_indicator > num_zero_value, hopf_indicator, np.nan)
        results[r] = {"phase": phase.copy(), "hopf": hopf_mask.copy(), "eq_cache": eq_cache.copy()}

    return results

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

# ==================================================================================================================================
# Analysis that compares the Hopf bifurcation curve for values of r=0 and r=1 in a (x,y) parameter space when you vary a parameter z
# ==================================================================================================================================

def vary_Hopf_analysis(model, x, y, x_vals, y_vals, z_vals, params):
    
    num_zero_value = num_zero()
    
    if not isinstance(z_vals, dict) or len(z_vals) != 1:
        raise ValueError("z_vals must be a dictionary containing one parameter name and its values, e.g. {'nu': [1e-5, 5e-5, 1e-4]}.")

    z = next(iter(z_vals))
    z_values = np.asarray(z_vals[z], dtype=float)

    results = {}
    
    for z_val in z_values:
        params_z = prep.params_ID(z, i_val=z_val, model=model, params=params)
        results[z_val] = {}
        
        for r in [0, 1]:
            params_r = prep.params_ID("r", i_val=r, model=model, params=params_z)
            starter_equilibria = prep.starting_coordinate(x, y, x_vals, y_vals, model, params_r, num_zero_value)

            if len(starter_equilibria) == 0:
                print(f"No viable endemic equilibrium was found for {z} = {z_val:.12g}, r = {r}.")
                results[z_val][r] = {
                    "hopf": np.full((len(x_vals), len(y_vals)), np.nan),
                    "imag": np.full((len(x_vals), len(y_vals)), np.nan),
                    "eq_cache": np.full((len(x_vals), len(y_vals), 3 if model == Model.Redist_3D else 4), np.nan)
                }
                continue

            start_coordinate = starter_equilibria[0]

            x_start = np.argmin(np.abs(x_vals - start_coordinate[x]))
            y_start = np.argmin(np.abs(y_vals - start_coordinate[y]))

            print("\n" + "=" * 70)
            print("STARTING HOPF PARAMETER SWEEP")
            print("=" * 70)
            print(f"{z} = {z_val:.12g}")
            print(f"r = {r}")
            print(f"Starting point: {x} = {x_vals[x_start]:.12g}, {y} = {y_vals[y_start]:.12g}")
            print(f"Total parameter combinations: {len(x_vals) * len(y_vals)}")

            # ============================================
            # Define state dimension for the selected model
            # ============================================

            if model == Model.Redist_3D:
                state_dimension = 3
            else:
                state_dimension = 4

            hopf_indicator = np.full((len(x_vals), len(y_vals)), np.nan)
            imag_indicator = np.full((len(x_vals), len(y_vals)), np.nan)
            eq_cache = np.full((len(x_vals), len(y_vals), state_dimension), np.nan)

            eps_boundary = 1e-10
            eps_eig = 1e-10

            def Model_state(state, params):
                return model(0, state, params)

            jax_jacobian = jax.jit(jax.jacfwd(Model_state))

            eq_cache[x_start, y_start] = start_coordinate["equilibrium"]

            points = []

            for j in range(len(y_vals)):
                for i in range(len(x_vals)):
                    distance = max(abs(i - x_start), abs(j - y_start))
                    points.append((distance, i, j))

            points.sort(key=lambda z: (z[0], z[2], z[1]))

            total_points = len(points)
            analysed = 0

            for distance, i, j in points:

                x_val = x_vals[i]
                y_val = y_vals[j]

                params_update = prep.params_ID(x, y, x_val, y_val, model, params_r)
                equilibrium = None

                # ======================================
                # Try the analytical endemic equilibrium
                # ======================================

                initial = prep.initial_state(model, params_update, num_zero_value)
                equilibrium = Model.Equilibrium(model, params_update, initial, num_zero_value)

                if equilibrium is not None:
                    equilibrium = np.asarray(equilibrium, dtype=float)

                    if model == Model.Redist_3D:
                        if not (np.all(np.isfinite(equilibrium)) and np.all(equilibrium > -num_zero_value) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > eps_boundary):
                            equilibrium = None

                    else:
                        if not (np.all(np.isfinite(equilibrium)) and np.all(equilibrium > -num_zero_value) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > 1 + eps_boundary and equilibrium[3] > eps_boundary):
                            equilibrium = None

                # =====================================
                # Try the neighbouring cached equilbria
                # =====================================

                if equilibrium is None:
                    neighbour_indices = [
                        (i - 1, j), (i + 1, j),
                        (i, j - 1), (i, j + 1),
                        (i - 1, j - 1), (i - 1, j + 1),
                        (i + 1, j - 1), (i + 1, j + 1)
                    ]

                    candidate_guesses = []

                    for ni, nj in neighbour_indices:
                        if 0 <= ni < len(x_vals) and 0 <= nj < len(y_vals):
                            neighbour = eq_cache[ni, nj]

                            if model == Model.Redist_3D:
                                if np.all(np.isfinite(neighbour)) and neighbour[0] > eps_boundary and neighbour[1] > eps_boundary and neighbour[2] > eps_boundary:
                                    candidate_guesses.append(neighbour)

                            else:
                                if np.all(np.isfinite(neighbour)) and neighbour[0] > eps_boundary and neighbour[1] > eps_boundary and neighbour[2] > 1 + eps_boundary and neighbour[3] > eps_boundary:
                                    candidate_guesses.append(neighbour)

                    for initial_guess in candidate_guesses:
                        equilibrium = Model.Equilibrium(model, params_update, initial_guess, num_zero_value)

                        if equilibrium is None:
                            continue

                        equilibrium = np.asarray(equilibrium, dtype=float)

                        if model == Model.Redist_3D:
                            if np.all(np.isfinite(equilibrium)) and np.all(equilibrium > -num_zero_value) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > eps_boundary:
                                break

                        else:
                            if np.all(np.isfinite(equilibrium)) and np.all(equilibrium > -num_zero_value) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > 1 + eps_boundary and equilibrium[3] > eps_boundary:
                                break

                        equilibrium = None

                # =====================================================
                # Try ODE integration from the analytical initial state
                # =====================================================

                if equilibrium is None:
                    initial = prep.initial_state(model, params_update, num_zero_value)
                    solution = solve_ivp(model, (0, 100), initial, args=(params_update,), rtol=1e-6, atol=1e-8, max_step=0.1)

                    if solution.success:
                        equilibrium = Model.Equilibrium(model, params_update, initial_guess=solution.y[:, -1], num_zero=num_zero_value)

                        if equilibrium is not None:
                            equilibrium = np.asarray(equilibrium, dtype=float)

                            if model == Model.Redist_3D:
                                if not (np.all(np.isfinite(equilibrium)) and np.all(equilibrium > -num_zero_value) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > eps_boundary):
                                    equilibrium = None

                            else:
                                if not (np.all(np.isfinite(equilibrium)) and np.all(equilibrium > -num_zero_value) and equilibrium[0] > eps_boundary and equilibrium[1] > eps_boundary and equilibrium[2] > 1 + eps_boundary and equilibrium[3] > eps_boundary):
                                    equilibrium = None

                # ===================================================
                # If no endemic equilibrium was found, skip the point
                # ===================================================

                if equilibrium is None:
                    analysed += 1

                    if analysed % 50 == 0:
                        print(f"{z} = {z_val:.12g}, r = {r}: Progress: {analysed}/{total_points} ({100 * analysed / total_points:.2f}%)")

                    continue

                equilibrium = np.asarray(equilibrium, dtype=float)

                if np.any(~np.isfinite(equilibrium)) or np.any(equilibrium < -num_zero_value):
                    analysed += 1

                    if analysed % 50 == 0:
                        print(f"{z} = {z_val:.12g}, r = {r}: Progress: {analysed}/{total_points} ({100 * analysed / total_points:.2f}%)")

                    continue

                eq_cache[i, j] = equilibrium

                # ==============================
                # Extract equilibrium components
                # ==============================

                if model == Model.Redist_3D:
                    H_star = equilibrium[0]
                    P_star = equilibrium[1]
                    L_star = equilibrium[2]

                    if H_star < eps_boundary:
                        analysed += 1

                        continue

                    if P_star < eps_boundary or L_star < eps_boundary:
                        analysed += 1

                        continue

                else:
                    H_star = equilibrium[0]
                    x_star = equilibrium[1]
                    A_star = equilibrium[2]
                    L_star = equilibrium[3]

                    if H_star < eps_boundary:

                        analysed += 1

                        continue

                    if x_star < eps_boundary or L_star < eps_boundary:

                        analysed += 1

                        continue

                # ==================================
                # Calculate Jacobian and eigenvalues
                # ==================================

                J_jax = jax_jacobian(jnp.asarray(equilibrium), params_update)
                J_jax = np.asarray(J_jax)

                if not np.all(np.isfinite(J_jax)):
                    analysed += 1

                    continue

                eigs = np.linalg.eigvals(J_jax)

                real_parts = np.real(eigs)
                imag_parts = np.abs(np.imag(eigs))

                idx = np.argmax(real_parts)
                max_real = real_parts[idx]

                hopf_indicator[i, j] = max_real
                imag_indicator[i, j] = imag_parts[idx]

                analysed += 1

                if analysed % 50 == 0:
                    print(f"{z} = {z_val:.12g}, r = {r}: Progress: {analysed}/{total_points} ({100 * analysed / total_points:.2f}%)")

            results[z_val][r] = {"hopf": hopf_indicator.copy(), "imag": imag_indicator.copy(), "eq_cache": eq_cache.copy()}

    return results