import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_pca_rectangles(
    scores_lo,
    scores_hi,
    labels=None,
    pc_x=0,
    pc_y=1,
    title="Interval PCA Score Plot",
    xlabel=None,
    ylabel=None,
    figsize=(9, 7),
    alpha=0.3,
    color="skyblue",
    edgecolor="navy",
):
    """Plots interval PCA scores as bounding boxes (rectangles) on a 2D factorial plane.

    Parameters
    ----------
    scores_lo : np.ndarray
        Lower bounds of scores, shape (N, p).
    scores_hi : np.ndarray
        Upper bounds of scores, shape (N, p).
    labels : list of str, optional
        Names of the statistical units (e.g., oil names).
    pc_x : int, default=0
        Index of the Principal Component for the X-axis (0 for PC1).
    pc_y : int, default=1
        Index of the Principal Component for the Y-axis (1 for PC2).
    """
    fig, ax = plt.subplots(figsize=figsize)

    N = scores_lo.shape[0]

    for i in range(N):
        # Extract bounds for the chosen PCs
        x_min, x_max = scores_lo[i, pc_x], scores_hi[i, pc_x]
        y_min, y_max = scores_lo[i, pc_y], scores_hi[i, pc_y]

        width = x_max - x_min
        height = y_max - y_min

        # Draw Rectangle
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=1.5,
            edgecolor=edgecolor,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Plot Midpoint
        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0
        ax.scatter(x_mid, y_mid, color=edgecolor, s=20, zorder=3)

        # Label placement
        if labels is not None:
            ax.annotate(
                labels[i],
                (x_mid, y_max),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                weight="bold",
            )

    # Set reasonable axis limits with margin
    all_x_lo, all_x_hi = scores_lo[:, pc_x], scores_hi[:, pc_x]
    all_y_lo, all_y_hi = scores_lo[:, pc_y], scores_hi[:, pc_y]

    margin_x = (np.max(all_x_hi) - np.min(all_x_lo)) * 0.1
    margin_y = (np.max(all_y_hi) - np.min(all_y_lo)) * 0.1

    ax.set_xlim(np.min(all_x_lo) - margin_x, np.max(all_x_hi) + margin_x)
    ax.set_ylim(np.min(all_y_lo) - margin_y, np.max(all_y_hi) + margin_y)

    # Reference axes (x=0, y=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel(xlabel if xlabel else f"PC {pc_x + 1}")
    ax.set_ylabel(ylabel if ylabel else f"PC {pc_y + 1}")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig, ax

def plot_correlation_circle(
    loadings,
    var_names=None,
    pc_x=0,
    pc_y=1,
    title="Circle of Correlations",
    figsize=(7, 7),
    color="crimson"
):
    """
    Plots the Correlation Circle for PCA loadings/correlations.

    Parameters
    ----------
    loadings : np.ndarray
        Loadings or correlations matrix of shape (p, n_components).
    var_names : list of str, optional
        Names of the original variables.
    pc_x : int, default=0
        Index of Principal Component for X-axis (0 for PC1).
    pc_y : int, default=1
        Index of Principal Component for Y-axis (1 for PC2).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the unit circle (r = 1)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circle)

    # Reference axes (x=0, y=0)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)

    p = loadings.shape[0]
    if var_names is None:
        var_names = [f"Var {i+1}" for i in range(p)]

    # Plot vectors for each variable
    for i in range(p):
        x = loadings[i, pc_x]
        y = loadings[i, pc_y]

        # Draw arrow
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, mutation_scale=15)
        )

        # Add text label slightly beyond the arrow tip
        ax.text(
            x * 1.1,
            y * 1.1,
            var_names[i],
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold"
        )

    # Formatting limits and aspect ratio
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(f"PC {pc_x + 1}")
    ax.set_ylabel(f"PC {pc_y + 1}")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    return fig, ax