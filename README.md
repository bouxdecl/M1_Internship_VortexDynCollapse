# athena_collapse_analysis

The relevant code to analyse the output of ATHENA++ vortex simulation for hydrodynamics in a spherically collapsing local box.

**Author:** Leo Boux de Casson
**Project start date:** October 2025

---

## Scientific Context

This project focuses on the **local dynamics of vortices embedded in a spherically-collapsing background flow**, motivated by:

- Spherical (or quasi-spherical) gravitational collapse in astrophysical contexts
- Vortex deformation, advection, and merging in time-dependent strain flows
- Comparison between **2D and 3D analytical vortex models** (Kirchhoff dynamics) and numerical simulations

The simulations are run in a **local Cartesian box with imposed collapse**, while the analysis is performed *a posteriori* using this package.


![A 3-day drifter/field comparison](save_plots/L3_Scenes/L3_Scenes_GOODBAD_best_filtering/GOOD_norma_scene_13.png)

---

## Features

#### Visualization
- Vorticity and density field visualization and animations
- Collapse profile diagnostics
- Phase-space trajectory comparisons
- Multi-time ellipse profile plots

#### Simulation Analysis
- **ATHENA++ I/O utilities** for HDF5 and VTK formats
- **Conservation law diagnostics** (mass, momentum, energy)
- **Vorticity and Mach number analysis** in collapsing coordinates
- **Elliptical vortex fitting** with streamfunction-based contour extraction
- **Shape diagnostics** (aspect ratio, orientation profiles)

#### Analytical Models
- **2D Hamiltonian Kirchhoff vortex model** with time-dependent strain
- **3D vortex tube advection model** (full deformation gradient evolution)
- **Direct comparison tools** between analytical theories and simulations
- **2D vs 3D theoretical cross-validation**
- **Efficient data caching** for large simulation datasets



---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/athena_collapse_analysis.git
cd athena_collapse_analysis
pip install -e .
```

### Dependencies

Standard scientific Python packages:
- `numpy`
- `scipy`
- `matplotlib`
- `h5py` (for HDF5 files)
- `pytest` (for testing)

ATHENA++ itself is **not required** to run the analysis.




---

## Usage

Example notebooks are provided, using almost all functionnalities.

### Quick Start

Import the package:

```python
import athena_collapse_analysis as aca
```

### Reading ATHENA++ Output

```python
from aca.io.ath_io import open_hdf_files_with_collapse

data = open_hdf_files_with_collapse(
    path_simu="./simulation_output/",
    files=["output.00100.athdf"]
)

rho = data["rho"]
vel = data["vel"]
time = data["time"]
```

### Conservation Law Diagnostics

```python
from aca.analysis.conservation_laws import compute_conservation_laws, plot_conservation_laws

time, Mtot, Omega_phys = compute_conservation_laws(
        path_simu,
        files,
        nz_slice=None,   # or e.g. nz_slice=0
        verbose=True
    )

plot_conservation_laws(time, Mtot, Omega_phys)
```

### Vorticity Visualization

```python
from aca.visu.plot_vorticity import make_vorticity_movie

make_vorticity_movie(path_simu, outname="physical_vorticity_animation.mp4",
                         vort_type="physical")
```



## Contact

For questions or feedback, email me at [leo.bouxdecasson@example.com](mailto:leo.bouxdecasson@example.com).


## License

This project is licensed under the BSD 3-Clause, Copyright (c) 2016, PrincetonUniversity - see the [LICENSE](LICENSE) file for details.
