use cellular_raza::prelude as cr;
use pyo3::prelude::*;

mod sim_branching;

use sim_branching::*;

#[pymodule]
fn cr_fractal_dim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sim_branching, m)?)?;
    m.add_function(wrap_pyfunction!(load_cells, m)?)?;
    m.add_function(wrap_pyfunction!(load_cells_at_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(load_subdomains_at_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_iterations, m)?)?;
    m.add_class::<Options>()?;
    m.add_class::<BacterialParameters>()?;
    m.add_class::<DomainParameters>()?;
    m.add_class::<TimeParameters>()?;
    m.add_class::<BacteriaBranching>()?;
    m.add_class::<cr::NewtonDamped2D>()?;
    m.add_class::<cr::MorsePotential>()?;
    m.add_class::<cr::CellIdentifier>()?;
    m.add_class::<cr::SubDomainPlainIndex>()?;
    Ok(())
}
