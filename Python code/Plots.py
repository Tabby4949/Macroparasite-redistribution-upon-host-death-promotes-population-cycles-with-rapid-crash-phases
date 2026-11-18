import numpy as np
from matplotlib.colors import ListedColormap
import Analysis as analysis
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =====================================
# Retrive the parameters in string form
# =====================================

def params_string(x, y, z=None):

    parameter_strings = {
        "d": "d",
        "beta": r"$\beta$",
        "psi": r"$\psi$",
        "sigma": r"$\sigma$",
        "b": "b",
        "nu": r"$\nu$",
        "xi": r"$\xi$",
        "alpha": r"$\alpha$",
        "h": "h",
        "delta": r"$\delta$",
        "lamb": r"$\lambda$",
        "k": "k",
        "r": "r"
    }

    if z is None:
        return parameter_strings[x], parameter_strings[y]

    return parameter_strings[x], parameter_strings[y], parameter_strings[z]


# ===========================================================
# Plot two-parameter bifurcation boundaries with phase labels
# ===========================================================

def boundary_plot(x, y, x_vals, y_vals, r_vals, results, label_x_pos, label_y_pos):

    phase_labels = {
        0: "SE",
        1: "SLC",
        2: "HE",
        3: "PF"
    }

    # ==============
    # Tick formatter
    # ==============

    x_ticks = np.linspace(np.min(x_vals), np.max(x_vals), 6)
    x_step = x_ticks[1] - x_ticks[0]
    x_exponent = int(np.floor(np.log10(abs(x_step))))

    y_ticks = np.linspace(np.min(y_vals), np.max(y_vals), 6)
    y_step = y_ticks[1] - y_ticks[0]
    y_exponent = int(np.floor(np.log10(abs(y_step))))

    def x_tick_formatter(value, pos):
        if x_exponent in [-1, 0, 1, 2]:
            return f"{value:g}"
        else:
            return f"{value / 10**x_exponent:g}"

    def y_tick_formatter(value, pos):
        if y_exponent in [-1, 0, 1, 2]:
            return f"{value:g}"
        else:
            return f"{value / 10**y_exponent:g}"

    x_formatter = FuncFormatter(x_tick_formatter)
    y_formatter = FuncFormatter(y_tick_formatter)

    # ===========================================
    # Create one separate figure for each r value
    # ===========================================

    for idx, r in enumerate(r_vals):

        fig, ax = plt.subplots(figsize=(6.1, 6.1))

        ax.set_box_aspect(1)

        # ==========================
        # Plot boundaries and labels
        # ==========================

        phase = results[r]["phase"]

        for p in phase_labels:

            mask = np.where(phase == p, 1.0, 0.0)

            if np.any(phase == p):

                ax.contour(x_vals, y_vals, mask.T, levels=[0.5], colors="black", linewidths=2)

        # ======================================
        # Label each phase using its mean centre
        # ======================================

        for p, label in phase_labels.items():

            indices = np.argwhere(phase == p)

            if len(indices) == 0:
                continue

            centre = np.mean(indices, axis=0)

            distances = np.sum((indices - centre)**2, axis=1)
            i, j = indices[np.argmin(distances)]

            ax.text(x_vals[i], y_vals[j], label, fontsize=18, ha="center", va="center", fontweight="bold", zorder=10)

        x_string, y_string = params_string(x, y)
        
        # ===========
        # Panel label
        # ===========
        
        panel_letter = chr(97 + idx)
        ax.text(label_x_pos, label_y_pos, rf"({panel_letter}) $r={r:g}$", fontsize=18, ha="right", va="top")
        
        # ===========
        # Axis labels
        # ===========

        ax.set_xlabel("")
        ax.set_ylabel("")

        if x_exponent not in [-1, 0, 1, 2]:
            ax.text(0.5, -0.08, x_string, transform=ax.transAxes, ha="right", va="top", fontsize=22)
            ax.text(0.5, -0.08, rf"$(\times 10^{{{x_exponent}}})$", transform=ax.transAxes, ha="left", va="top", fontsize=16)
            
        else:
            ax.text(0.5, -0.08, x_string, transform=ax.transAxes, ha="center", va="top", fontsize=22)

        if y_exponent not in [-1, 0, 1, 2]:
            ax.text(-0.10, 0.5, y_string, transform=ax.transAxes, ha="right", va="center", fontsize=22)
            ax.text(-0.10, 0.5, rf"$(\times 10^{{{y_exponent}}})$", transform=ax.transAxes, ha="left", va="center", fontsize=16, rotation=90)
            
        else:
            ax.text(-0.10, 0.5, y_string, transform=ax.transAxes, ha="center", va="center", fontsize=22, rotation=90)

        # ====
        # Axis
        # ====

        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.tick_params(axis="x", labelsize=14)

        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):

            if tick == 0:
                label.set_rotation(0)
                label.set_ha("center")

        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.tick_params(axis="y", labelsize=14)

        # ======
        # Layout
        # ======

        plt.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.98)

        plt.show()
    
# -----------------------------------------------------------------------------   

# =============================================
# Plot the Hopf bifurcation curve for various r
# =============================================
  
  
def plot_Hopf_comparison(x, y, x_vals, y_vals, r_vals, results, r_label_pos):

    cmap = plt.get_cmap("cool")
    colors = cmap(np.linspace(0.2, 0.9, len(r_vals)))

    fig, ax = plt.subplots(figsize=(9, 7))

    # ==============
    # Tick formatter
    # ==============

    x_ticks = np.linspace(np.min(x_vals), np.max(x_vals), 6)
    x_step = x_ticks[1] - x_ticks[0]
    x_exponent = int(np.floor(np.log10(abs(x_step))))

    y_ticks = np.linspace(np.min(y_vals), np.max(y_vals), 6)
    y_step = y_ticks[1] - y_ticks[0]
    y_exponent = int(np.floor(np.log10(abs(y_step))))

    def x_tick_formatter(value, pos):
        if x_exponent in [-1, 0, 1, 2]:
            return f"{value:g}"
        else:
            return f"{value / 10**x_exponent:g}"

    def y_tick_formatter(value, pos):
        if y_exponent in [-1, 0, 1, 2]:
            return f"{value:g}"
        else:
            return f"{value / 10**y_exponent:g}"

    x_formatter = FuncFormatter(x_tick_formatter)
    y_formatter = FuncFormatter(y_tick_formatter)

    # ====================
    # Plot the Hopf curves
    # ====================

    for r, col in zip(r_vals, colors):

        hopf = results[r]["hopf"]

        cs = ax.contour(x_vals, y_vals, hopf.T, levels=[0], colors=[col], linewidths=4)

        if len(cs.allsegs[0]) == 0:
            continue

        seg = max(cs.allsegs[0], key=len)

        if len(seg) < 20:
            continue

        # ======================
        # Position of the labels
        # ======================

        frac = r_label_pos[round(float(r), 2)]

        i = int(frac * (len(seg) - 1))

        x_label_pos, y_label_pos = seg[i]

        ax.text(x_label_pos, y_label_pos, rf"$r={r:.2f}$", fontsize=19, rotation_mode="anchor", ha="center", va="center", color=col, bbox=dict(facecolor="white", edgecolor="none", pad=0), zorder=10) #fontsize=17

    x_string, y_string = params_string(x, y)

    # ===========
    # Axis labels
    # ===========

    ax.set_xlabel("")
    ax.set_ylabel("")

    if x_exponent not in [-1, 0, 1, 2]:
        ax.text(0.5, -0.08, x_string, transform=ax.transAxes, ha="right", va="top", fontsize=28) 
        ax.text(0.5, -0.085, rf"$(\times 10^{{{x_exponent}}})$", transform=ax.transAxes, ha="left", va="top", fontsize=19) 

    else:
        ax.text(0.5, -0.08, x_string, transform=ax.transAxes, ha="center", va="top", fontsize=28)

    if y_exponent not in [-1, 0, 1, 2]:
        ax.text(-0.10, 0.5, y_string, transform=ax.transAxes, ha="right", va="center", fontsize=28)
        ax.text(-0.10, 0.5, rf"$(\times 10^{{{y_exponent}}})$", transform=ax.transAxes, ha="left", va="center", fontsize=19, rotation=90) 

    else:
        ax.text(-0.10, 0.5, y_string, transform=ax.transAxes, ha="center", va="center", fontsize=28, rotation=90)

    # ====
    # Axis
    # ====

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.tick_params(axis="x", labelsize=17) 

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):

        if tick == 0:
            label.set_rotation(0)
            label.set_ha("center")

    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis="y", labelsize=17)

    # ======
    # Layout
    # ======

    plt.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.98)

    plt.show()
    
# ------------------------------------------------------------------------------
    
# ===========================================================================================
# Plot the Hopf bifurcation curves for various values of parameter z in {x,y} parameter space
# ===========================================================================================
    

def vary_Hopf_plot(x, y, x_vals, y_vals, z_vals, results, colours, label_pos, r_plot=None):
    
    num_zero_value = analysis.num_zero()

    z = next(iter(z_vals))
    z_values = np.asarray(z_vals[z], dtype=float)

    x_string, y_string = params_string(x, y)
    z_string, _ = params_string(z, z)

    # ==============
    # Tick formatter
    # ==============

    x_ticks = np.linspace(np.min(x_vals), np.max(x_vals), 6)
    x_step = x_ticks[1] - x_ticks[0]
    x_exponent = int(np.floor(np.log10(abs(x_step))))

    y_ticks = np.linspace(np.min(y_vals), np.max(y_vals), 6)
    y_step = y_ticks[1] - y_ticks[0]
    y_exponent = int(np.floor(np.log10(abs(y_step))))

    def x_tick_formatter(value, pos):
        if x_exponent in [-1, 0, 1, 2]:
            return f"{value:g}"
        else:
            return f"{value / 10**x_exponent:g}"

    def y_tick_formatter(value, pos):
        if y_exponent in [-1, 0, 1, 2]:
            return f"{value:g}"
        else:
            return f"{value / 10**y_exponent:g}"

    x_formatter = FuncFormatter(x_tick_formatter)
    y_formatter = FuncFormatter(y_tick_formatter)

    # ===============
    # Label formatter
    # ===============

    def format_z_value(z_val):
        if z_val == 0:
            return "0"
        elif abs(z_val) < 0.099:
            exponent = int(np.floor(np.log10(abs(z_val))))
            coefficient = z_val / 10**exponent
            return rf"$\mathregular{{{coefficient:g}\times 10^{{{exponent}}}}}$"
        else:
            return f"{z_val:g}"

    fig, ax = plt.subplots(figsize=(9, 7))

    # ================================
    # Determine which r values to plot
    # ================================

    if r_plot is None:
        r_values = [(0, "--"), (1, "-")]
    elif r_plot in [0, 1]:
        r_values = [(r_plot, "-")]
    else:
        raise ValueError("r_plot must be None, 0, or 1.")
        
    # ================
    # Plot Hopf curves
    # ================

    for z_val, col in zip(z_values, colours):

        for r, linestyle in r_values:

            hopf_indicator = results[z_val][r]["hopf"]
            imag_indicator = results[z_val][r]["imag"]

            hopf_mask = np.where(imag_indicator > num_zero_value, hopf_indicator, np.nan)

            if not np.any(np.isfinite(hopf_mask)):
                continue

            ax.contour(x_vals, y_vals, hopf_mask.T, levels=[0], colors=[col], linewidths=4, linestyles=linestyle)
            
            if (r_plot is not None or r == 1):

                contour = ax.contour(x_vals, y_vals, hopf_mask.T, levels=[0], colors="none")

                segments = contour.allsegs[0]

                if len(segments) > 0:

                    seg = max(segments, key=len)

                    if len(seg) > 10:

                        i = int(0.55 * (len(seg) - 1))
                        x_point, y_point = seg[i]

                        position = label_pos.get(z_val, 0.85)
                        i = int(position * (len(seg) - 1))
                        x_point, y_point = seg[i]

                        ax.text(x_point, y_point, rf"{z_string}={format_z_value(z_val)}", fontsize=19, color=col, rotation=0, rotation_mode="anchor", ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    
    # ===========
    # Axis labels
    # ===========

    ax.set_xlabel("")
    ax.set_ylabel("")

    if x_exponent not in [-1, 0, 1, 2]:
        ax.text(0.5, -0.08, x_string, transform=ax.transAxes, ha="right", va="top", fontsize=28)
        ax.text(0.5, -0.08, rf"$(\times 10^{{{x_exponent}}})$", transform=ax.transAxes, ha="left", va="top", fontsize=19)
        
    else:
        ax.text(0.5, -0.08, x_string, transform=ax.transAxes, ha="center", va="top", fontsize=28)

    if y_exponent not in [-1, 0, 1, 2]:
        ax.text(-0.10, 0.5, y_string, transform=ax.transAxes, ha="right", va="center", fontsize=28)
        ax.text(-0.10, 0.5, rf"$(\times 10^{{{y_exponent}}})$", transform=ax.transAxes, ha="left", va="center", fontsize=19, rotation=90)
        
    else:
        ax.text(-0.10, 0.5, y_string, transform=ax.transAxes, ha="center", va="center", fontsize=28, rotation=90)

    # ====
    # Axis
    # ====

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.tick_params(axis="x", labelsize=17)

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):

        if tick == 0:
            label.set_rotation(0)
            label.set_ha("center")

    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis="y", labelsize=17)

    plt.tight_layout()
    plt.show()