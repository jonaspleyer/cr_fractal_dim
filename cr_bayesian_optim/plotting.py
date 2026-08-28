from .cr_fractal_dim import (
    load_cells_at_iteration,
    get_all_iterations,
    load_subdomains_at_iteration,
)

from pathlib import Path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mp
import itertools
from skopt.plots import plot_gaussian_process, plot_convergence, plot_objective

# Define colors
COLOR1 = "#0c457d"
COLOR2 = "#0ea7b5"
COLOR3 = "#6bd2db"
COLOR4 = "#ffbe4f"
COLOR5 = "#e8702a"


def my_cmap():
    cmap = matplotlib.colormaps["viridis"]
    colors = cmap(np.arange(cmap.N))
    colors[:, -1] = np.linspace(1, 0, cmap.N)
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap


def set_rcparams():
    matplotlib.rcParams["axes.facecolor"] = (0, 0, 0, 0)
    matplotlib.rcParams["figure.facecolor"] = (0, 0, 0, 0)
    matplotlib.rcParams["figure.edgecolor"] = (0, 0, 0, 0)
    matplotlib.rcParams["savefig.facecolor"] = (0, 0, 0, 0)
    matplotlib.rcParams["savefig.edgecolor"] = (0, 0, 0, 0)
    matplotlib.rcParams["legend.facecolor"] = (0, 0, 0, 0)
    matplotlib.rcParams["legend.framealpha"] = None


def plot_iteration(
    iteration: int,
    intra_bounds: tuple[float, float],
    extra_bounds: tuple[float, float],
    output_path: Path,
    save_figure: bool = True,
    figsize: int = 32,
) -> matplotlib.figure.Figure | None:
    set_rcparams()

    cells = load_cells_at_iteration(output_path, iteration)
    subdomains = load_subdomains_at_iteration(output_path, iteration)

    # Set size of the image
    sbd0 = list(subdomains.values())[0]
    domain_min: tuple[float, float] = sbd0["subdomain"]["domain_min"]
    domain_max: tuple[float, float] = sbd0["subdomain"]["domain_max"]
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim((domain_min[0], domain_max[0]))
    ax.set_ylim((domain_min[1], domain_max[1]))

    # Plot background
    # max_size = np.max([dfsi["index_max"] for _, dfsi in dfs.iterrows()], axis=0)
    max_size = np.max([sbd["index_max"] for sbd in subdomains.values()], axis=0)
    all_values = np.zeros(max_size)
    for sbd in subdomains.values():
        values = np.array(sbd["extracellular"]["data"]).reshape(
            sbd["extracellular"]["dim"]
        )[:, :, 0]
        filt = np.array(sbd["ownership_array"]["data"]).reshape(
            sbd["ownership_array"]["dim"]
        )
        filt = filt[1:-1, 1:-1]

        index_min = np.array(sbd["index_min"])
        slow = index_min
        shigh = index_min + np.array(values.shape)
        all_values[slow[0] : shigh[0], slow[1] : shigh[1]] += values * filt

    ax.imshow(
        all_values.T,
        vmin=extra_bounds[0],
        vmax=extra_bounds[1],
        extent=(domain_min[0], domain_max[0], domain_min[1], domain_max[1]),
        origin="lower",
        cmap=my_cmap(),
    )

    # Plot cells
    points = np.array([c[0].mechanics.pos for c in cells.values()])
    radii = np.array([c[0].interaction.radius for c in cells.values()])
    radii_div = np.array([c[0].division_radius for c in cells.values()])
    s = np.clip(
        (radii / radii_div - intra_bounds[0]) / (intra_bounds[1] - intra_bounds[0]),
        0,
        1,
    )

    color_high = np.array([233, 170, 242]) / 255
    color_low = np.array([129, 12, 145]) / 255
    color = np.tensordot((1 - s), color_low, 0) + np.tensordot(s, color_high, 0)

    # Plot cells as circles
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    collection = PatchCollection(
        [
            Circle(
                points[i, :],
                radius=radii[i],
            )
            for i in range(points.shape[0])
        ],
        facecolors=color,
    )
    ax.add_collection(collection)
    ax.text(
        0.05,
        0.05,
        "Agents: {:9}".format(points.shape[0]),
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="center",
        bbox=dict(boxstyle="square", facecolor="#FFFFFF"),
    )

    ax.set_axis_off()
    if save_figure:
        os.makedirs(output_path / "images", exist_ok=True)
        fig.savefig(
            output_path / "images/cells_at_iter_{:010}".format(iteration),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
        return None
    else:
        return fig


def __plot_all_iterations_helper(args_kwargs):
    iteration, kwargs = args_kwargs
    plot_iteration(iteration, **kwargs)


def plot_all_iterations(
    intra_bounds: tuple[float, float],
    extra_bounds: tuple[float, float],
    output_path: Path,
    n_threads: int | None = None,
    **kwargs,
):
    pool = mp.Pool(n_threads)
    kwargs["intra_bounds"] = intra_bounds
    kwargs["extra_bounds"] = extra_bounds
    kwargs["output_path"] = output_path
    iterations = get_all_iterations(output_path)
    args = zip(
        iterations,
        itertools.repeat(kwargs),
    )
    print("Plotting Results")
    _ = list(
        tqdm.tqdm(pool.imap(__plot_all_iterations_helper, args), total=len(iterations))
    )


def generate_movie(opath: Path, play_movie: bool = True):
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{opath}/images/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {opath}/movie.mp4"
    os.system(bashcmd)

    if play_movie:
        print("Playing Movie")
        bashcmd2 = f"firefox ./{opath}/movie.mp4"
        os.system(bashcmd2)


# Plotting for bacteria rods optimization:
def plot_optimization_convergence_bayes(res, path="", add_name=""):
    fig, ax = plt.subplots()
    plot_convergence(res)
    plt.savefig(path + "skopt_convergence" + add_name + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_1D_cost_approximation_bayes(res, path="", add_name=""):
    fig, ax = plt.subplots()
    _ = plot_gaussian_process(res)
    plt.savefig(path + "cost_approximation" + add_name + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_objective_projection_bayes(res, path="", add_name=""):
    fig, ax = plt.subplots()
    _ = plot_objective(res, n_points=10)
    plt.savefig(path + "objective_projection" + add_name + ".png", bbox_inches="tight")
    plt.close(fig)
