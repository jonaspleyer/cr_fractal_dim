from .cr_bayesian_optim import (
    Options,
    load_cells,
    load_cells_at_iteration,
    get_all_iterations,
    run_sim_branching,
)
from pathlib import Path
from glob import glob
import numpy as np
import scipy as sp
from tqdm import tqdm
from PIL import Image


def check_exists(options: Options) -> Path | None:
    for file in glob(str(options.storage_location / "**/options.toml")):
        file_path = Path(file)
        opt_loaded = Options.load_from_toml(file_path)
        if opt_loaded == options:
            return file_path.parent


def load_or_compute_full(options):
    """
    Obtains a result for a given combination of options by either loading from existing files or
    performing a fully new numerical simulation.

    .. seealso::
        The :meth:`load_or_compute_last_iter` function is cheaper when only loading the last
        iteration.
    """
    out_path = check_exists(options)
    if out_path is not None:
        return load_cells(out_path), out_path
    else:
        print("Running Simulation")
        cells, out_path = run_sim_branching(options)
        return cells, out_path


def load_or_compute_last_iter(options):
    out_path = check_exists(options)
    if out_path is not None:
        last_iter = get_all_iterations(out_path)[-1]
        return load_cells_at_iteration(out_path, last_iter), out_path
    else:
        print("Running Simulation")
        cells, out_path = run_sim_branching(options)
        last_iter = sorted(cells.keys())[-1]
        return cells[last_iter], out_path


def calculate_discretization(
    positions: np.ndarray,
    n_voxels: int,
    options: Options,
    raw: bool = False,
):
    radius = options.bacteria.cell_radius / options.domain.domain_size
    voxels = positions * n_voxels / options.domain.domain_size
    # Calculate padding
    dist = 2 * n_voxels * radius

    y = np.zeros((n_voxels, n_voxels))
    for vox in voxels:
        xmin = max(vox[0] - dist, 0)
        xmax = min(vox[0] + dist + 1, n_voxels)
        ymin = max(vox[1] - dist, 0)
        ymax = min(vox[1] + dist + 1, n_voxels)
        for i in range(int(xmin), np.ceil(xmax).astype(int)):
            for j in range(int(ymin), np.ceil(ymax).astype(int)):
                if i == int(vox[0]) and j == int(vox[1]):
                    y[i, j] += 1
                elif ((i - vox[0]) ** 2 + (j - vox[1]) ** 2) ** 0.5 <= dist:
                    y[i, j] += 1
    if raw:
        return y
    else:
        return y > 0


def plot_discretizations(last_pos, n_voxels_list, options, out_path: Path):
    # Plot Snapshots of Discretization to calculate Fractal Dimension
    n = len(n_voxels_list)
    indices = np.round(np.linspace(0, n - 1, min(10, n), endpoint=True)).astype(int)
    counts = []
    for n_voxels in tqdm(n_voxels_list[indices], desc="Plotting Discretizations"):
        y = calculate_discretization(
            last_pos,
            n_voxels,
            options,
            raw=True,
        )
        counts.append((n_voxels, y))

    max_overall = np.max([np.max(c[1]) for c in counts])
    for n_voxels, y in counts:
        img = Image.fromarray(
            np.round(y / max_overall * 125 + (y > 0) * 130).astype(np.uint8).T[::-1]
        )
        img.save(out_path / f"discretization-nvoxels-{n_voxels:06}.png")


def calculate_fractal_dim_for_pos(pos, options: Options, out_path: Path | None = None):
    x = np.linspace(
        options.bacteria.cell_radius / 2.0,
        options.domain.domain_size / 10,
        10,
        dtype=float,
    )
    n_voxels_list = np.floor(options.domain.domain_size / x).astype(int)
    fields = [calculate_discretization(pos, n, options) for n in n_voxels_list]
    count_boxes = np.array([np.sum(yi > 0) for yi in fields]).astype(int)

    if out_path is not None:
        plot_discretizations(pos, n_voxels_list, options, out_path)

    popt, pcov = sp.optimize.curve_fit(
        lambda x, a, b: a * x + b, np.log(x), np.log(count_boxes)
    )
    return x, count_boxes, popt, pcov
