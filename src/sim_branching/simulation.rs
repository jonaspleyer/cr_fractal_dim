use approxim::AbsDiffEq;
use cellular_raza::prelude as cr;
use nalgebra::Vector2;
use num::Zero;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, num::NonZeroUsize};

use super::agent::BacteriaBranching;

pub type ReactionVector = nalgebra::DVector<f64>;

macro_rules! opt (
    ($name:ident $($ti:tt)*) => {
        short_default::default! {
            #[pyclass(get_all, set_all)]
            #[derive(Clone, Debug, AbsDiffEq, PartialEq, Serialize, Deserialize)]
            #[approx(epsilon_type = f64)]
            pub struct $name $($ti)*
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (**kwargs))]
            pub fn new(
                py: Python,
                kwargs: Option<&Bound<pyo3::types::PyDict>>
            ) -> PyResult<Py<$name>> {
                let new = Py::new(py, $name ::default())?;
                if let Some(kwds) = kwargs {
                    for (key, value) in kwds.iter() {
                        let key: Py<pyo3::types::PyString> = key.extract()?;
                        new.setattr(py, &key, value)?;
                    }
                }
                Ok(new)
            }

            pub fn __repr__(&self) -> String {
                format!("{:#?}", self)
            }
        }
    }
);

opt! { BacterialParameters {
    #[approx(equal)]
    pub n_bacteria_initial: u32 = 5,
    pub cell_radius: f64 = 6.0,
    pub division_threshold: f64 = 2.0,
    pub potential_stiffness: f64 = 0.15,
    pub potential_strength: f64 = 2.0,
    pub damping_constant: f64 = 1.0,
    pub uptake_rate: f64 = 1.0,
    pub growth_rate: f64 = 13.0,
}}

opt! { DomainParameters {
    /// Overall size of the domain
    pub domain_size: f64 = 3000.0,
    pub voxel_size: f64 = 30.0,
    /// Size of the square for initlal placement of bacteria
    pub domain_starting_size: f64 = 100.0,
    /// Discretization of the diffusion process
    pub reactions_dx: f64 = 20.0,
    pub diffusion_constant: f64 = 80.0,
    pub initial_concentration: f64 = 10.0,
}}

opt! { TimeParameters {
    pub dt: f64 = 0.1,
    pub t_max: f64 = 2000.0,
    #[approx(equal)]
    pub save_interval: usize = 200,
}}

fn get_inner<T>(ptp: &Py<T>, py: Python) -> T
where
    T: for<'a, 'py> pyo3::conversion::FromPyObjectBound<'a, 'py>,
{
    ptp.extract(py).unwrap()
}

#[pyclass(get_all, set_all)]
#[derive(Clone, Serialize, Deserialize, AbsDiffEq)]
#[approx(epsilon_type = f64)]
pub struct Options {
    /// Holds bacteria parameters. See :class:`BacterialParameters`.
    #[approx(map = |b| Python::attach(|py| Some(get_inner(b, py))))]
    pub bacteria: Py<BacterialParameters>,
    /// Holds domain parameters. See :class:`DomainParameters`.
    #[approx(map = |b| Python::attach(|py| Some(get_inner(b, py))))]
    pub domain: Py<DomainParameters>,
    /// Holds time parameters. See :class:`TimeParameters`.
    #[approx(map = |b| Python::attach(|py| Some(get_inner(b, py))))]
    pub time: Py<TimeParameters>,
    /// Show or hide the progress bar during the simulation.
    #[approx(skip)]
    pub progressbar: bool,
    /// Specify how many threads to use for the simulation.
    /// Must be a positive integer value.
    #[approx(equal)]
    pub n_threads: NonZeroUsize,
    /// If this variable is set to `None`, no output will be produced and results will only be
    /// returned in memory.
    /// If a path is given, results will be stored there with a timestamp of the beginning of the
    /// simulation.
    #[approx(skip)]
    pub storage_location: Option<std::path::PathBuf>,
}

#[pymethods]
impl Options {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python, kwargs: Option<&Bound<pyo3::types::PyDict>>) -> PyResult<Py<Self>> {
        let new = Py::new(
            py,
            Self {
                bacteria: Py::new(py, <BacterialParameters as Default>::default())?,
                domain: Py::new(py, <DomainParameters as Default>::default())?,
                time: Py::new(py, <TimeParameters as Default>::default())?,
                progressbar: false,
                n_threads: 1.try_into().unwrap(),
                storage_location: Some("out".into()),
            },
        )?;
        if let Some(kwds) = kwargs {
            for (key, value) in kwds.iter() {
                let key: Py<pyo3::types::PyString> = key.extract()?;
                new.setattr(py, &key, value)?;
            }
        }
        Ok(new)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        use std::io::Write;
        let mut out = Vec::new();
        write!(out, "Options {{")?;
        let fields = [
            format!("{}", self.bacteria.call_method0(py, "__repr__")?),
            format!("{}", self.bacteria.call_method0(py, "__repr__")?),
            format!("{}", self.domain.call_method0(py, "__repr__")?),
            format!("{}", self.time.call_method0(py, "__repr__")?),
        ];
        for field in fields {
            for line in field.lines() {
                writeln!(out)?;
                write!(out, "    {}", line)?;
            }
            write!(out, ",")?;
        }
        writeln!(out, "\n}}")?;
        Ok(String::from_utf8(out)?)
    }

    fn save_to_toml(&self, path: std::path::PathBuf) -> PyResult<()> {
        use std::io::prelude::*;
        let serialized_toml = toml::to_string_pretty(&self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(serialized_toml.as_bytes())?;
        Ok(())
    }

    #[staticmethod]
    fn load_from_toml(path: std::path::PathBuf) -> PyResult<Self> {
        let contents = std::fs::read_to_string(path)?;
        toml::from_str(&contents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.abs_diff_eq(other, Self::default_epsilon())
    }
}

impl PartialEq for Options {
    fn eq(&self, other: &Self) -> bool {
        Python::attach(|py| {
            self.bacteria.borrow(py).eq(&other.bacteria.borrow(py))
                && self.domain.borrow(py).eq(&other.domain.borrow(py))
                && self.time.borrow(py).eq(&other.time.borrow(py))
                && self.n_threads.eq(&other.n_threads)
        })
    }
}

pub type SingleIterCells =
    BTreeMap<cr::CellIdentifier, (BacteriaBranching, Option<cr::CellIdentifier>)>;
pub type SingleIterSubDomains<'a> = BTreeMap<cr::SubDomainPlainIndex, Bound<'a, PyAny>>;
pub type CellOutput = BTreeMap<u64, SingleIterCells>;

/// Performs a full numerical simulation with the given :class:`Options`.
///
/// .. warning::
///     Although this function is fully working it is highly recommended to use the
///     :meth:`load_or_compute_full` function instead which will check first if a result with
///     the given options has already been produced and thus reuse these existing results.
#[pyfunction]
pub fn run_sim_branching(
    py: Python,
    options: Options,
) -> Result<(CellOutput, std::path::PathBuf), cr::SimulationError> {
    use cr::*;

    let BacterialParameters {
        n_bacteria_initial,
        cell_radius,
        division_threshold,
        potential_stiffness,
        potential_strength,
        damping_constant,
        uptake_rate,
        growth_rate,
    } = *options.bacteria.borrow(py);
    let DomainParameters {
        domain_size,
        voxel_size,
        domain_starting_size,
        reactions_dx,
        diffusion_constant,
        initial_concentration,
    } = *options.domain.borrow(py);
    let TimeParameters {
        dt,
        t_max,
        save_interval,
    } = *options.time.borrow(py);
    let n_threads = options.n_threads;

    let ds = domain_size / 2.0;
    let dx = domain_starting_size / 2.0;

    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let cells = (0..n_bacteria_initial)
        .map(|_| {
            let x = rng.random_range(ds - dx..ds + dx);
            let y = rng.random_range(ds - dx..ds + dx);

            let pos = Vector2::from([x, y]);
            BacteriaBranching {
                mechanics: NewtonDamped2D {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant,
                    mass: 1.0,
                },
                interaction: MorsePotential {
                    radius: cell_radius,
                    potential_stiffness,
                    cutoff: 2.0 * division_threshold * cell_radius,
                    strength: potential_strength,
                },
                uptake_rate,
                division_radius: division_threshold * cell_radius,
                growth_rate,
            }
        })
        .collect::<Vec<_>>();

    let cond = dt - 0.5 * reactions_dx / diffusion_constant;
    if cond >= 0.0 {
        println!(
            "❗❗❗WARNING❗❗❗\n\
            The stability condition \
            dt <= 0.5 dx^2/D for the integration \
            method is not satisfied. This can \
            lead to solving errors and inaccurate \
            results."
        );
    }

    if voxel_size < division_threshold * cell_radius {
        println!(
            "❗❗❗WARNING❗❗❗\n\
            The voxel_size {voxel_size} has been chosen \
            smaller than the length of the interaction {}. This \
            will probably yield incorrect results.",
            division_threshold * cell_radius,
        );
    }

    let domain = CartesianDiffusion2D {
        domain: CartesianCuboid::from_boundaries_and_interaction_range(
            [0.0; 2],
            [domain_size, domain_size],
            voxel_size,
        )?,
        reactions_dx: [reactions_dx; 2].into(),
        diffusion_constant,
        initial_value: ReactionVector::from(vec![initial_concentration]),
    };

    let storage = match &options.storage_location {
        Some(loc) => StorageBuilder::new()
            .priority([StorageOption::Memory, StorageOption::SerdeJson])
            .location(&loc),
        None => StorageBuilder::new().priority([StorageOption::Memory]),
    };
    let time = FixedStepsize::from_partial_save_freq(0.0, dt, t_max, save_interval)?;
    let settings = Settings {
        n_threads,
        time,
        storage,
        progressbar: if options.progressbar {
            Some("".to_string())
        } else {
            None
        },
    };

    let storager = run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, ReactionsExtra, Cycle],
        parallelizer: Rayon,
        zero_reactions_default: |_| nalgebra::DVector::zeros(1),
    )?;
    if let Err(e) = options.save_to_toml(storager.get_path()?.join("options.toml")) {
        eprintln!("Encountered error when saving simulation Options to file:");
        eprintln!("{e}");
    }
    let cells = storager
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            (
                iteration,
                cells
                    .into_iter()
                    .map(|(ident, (cbox, _))| (ident, (cbox.cell, cbox.parent)))
                    .collect(),
            )
        })
        .collect();
    Ok((cells, storager.get_path()?))
}
