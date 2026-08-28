"""
>>> import cr_bayesian_optim as crb
>>> options = crb.Options()
>>> cells, out_path = crb.run_sim_branching(options)
"""

from .cr_bayesian_optim import (
    run_sim_branching,
    load_cells,
    load_cells_at_iteration,
    load_subdomains_at_iteration,
    get_all_iterations,
    Options,
    BacterialParameters,
    DomainParameters,
    TimeParameters,
    BacteriaBranching,
)

type CellIdentifier = tuple[int, int]
type CellOutput = dict[
    int, dict[CellIdentifier, tuple[BacteriaBranching, CellIdentifier | None]]
]
type SingleIterCells = dict[
    CellIdentifier, tuple[BacteriaBranching, CellIdentifier | None]
]
type SingleIterSubDomains = dict[int, dict]

import cr_bayesian_optim.sim_branching as sim_branching
import cr_bayesian_optim.plotting as plotting

from .fractal_dim import fractal_dim_main
