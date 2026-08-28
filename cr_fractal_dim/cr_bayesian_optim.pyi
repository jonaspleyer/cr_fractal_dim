import numpy as np
from pathlib import Path

# From branching part
class Options:
    bacteria: BacterialParameters
    domain: DomainParameters
    time: TimeParameters
    show_progressbar: bool = False
    n_threads: int = 1
    storage_location: Path = Path("out")

    @staticmethod
    def __new__(cls, **kwargs): ...
    def save_to_toml(self, path: Path): ...
    @staticmethod
    def load_from_toml(path: Path) -> Options: ...

class BacterialParameters:
    n_bacteria_initial: int = 5
    cell_radius: float = 6.0
    division_threshold: float = 2.0
    potential_stiffness: float = 0.15
    potential_strength: float = 2.0
    damping_constant: float = 1.0
    uptake_rate: float = 1.0
    growth_rate: float = 13.0

    @staticmethod
    def __new__(cls, **kwargs): ...

class DomainParameters:
    domain_size: float = 3000.0
    voxel_size: float = 30.0
    domain_starting_size: float = 100.0
    reactions_dx: float = 20.0
    diffusion_constant: float = 80.0
    initial_concentration: float = 10.0

    @staticmethod
    def __new__(cls, **kwargs): ...

class TimeParameters:
    dt: float = 0.1
    t_max: float = 2000.0
    save_interval: int = 200

    @staticmethod
    def __new__(cls, **kwargs): ...

type CellIdentifier = tuple[int, int]
type SingleIterCells = dict[CellIdentifier, tuple[BacteriaBranching, CellIdentifier | None]]
type SingleIterSubDomains = dict[int, dict]
type CellOutput = dict[int, SingleIterCells]

class BacteriaBranching:
    mechanics: NewtonDamped2D
    interaction: MorsePotential
    uptake_rate: float
    division_radius: float
    growth_rate: float

def run_sim_branching(options: Options) -> tuple[CellOutput, Path]: ...
def load_cells(path: Path) -> CellOutput: ...
def load_cells_at_iteration(path: Path, iteration: int) -> SingleIterCells: ...
def load_subdomains_at_iteration(path: Path, iteartion: int) -> SingleIterSubDomains: ...
def get_all_iterations(path: Path) -> list[int]: ...

# From cellular_raza
class NewtonDamped2D:
    pos: np.typing.NDArray[np.float64]
    vel: np.typing.NDArray[np.float64]
    damping_constant: float
    mass: float

class MorsePotential:
    radius: float
    potential_stiffness: float
    cutoff: float
    strength: float
