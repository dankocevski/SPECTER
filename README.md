# SPECTER

The Spectral and Temporal analysis for Energetic Radiation (SPECTER) toolkit is a Python library for **Fermi/GBM gamma-ray burst (GRB) analysis**, built on top of the **Gamma-ray Data Tools (GDT)**. It provides a single high-level `GRB` class that wraps data I/O, background modeling, light-curve and spectral analysis, and common GRB diagnostics. Optional PyQt-based GUI windows support interactive selections and visualization.

## Key capabilities
- Load GBM **TTE**, **CSPEC**, and **CTIME** products
- Manage detector collections (NaI/BGO) and response files (RSP2)
- Background fitting and interactive background/source selection
- Light-curve plotting and stacking
- Spectral fitting with common models and statistics (PG-Stat/C-Stat/Chi-square)
- T90 estimation, Bayesian blocks, structure function, and hardness ratios
- Orbital/geometry utilities (detector angles, SAA timing, localization plots)
- Quasi-periodic search utilities
- Spectral line search utilities
- Energetics calculations: **$E_{\rm iso}$** and **$L_{\rm iso}$**

## Installation
SPECTER is a lightweight repo without a packaging setup yet. Install the required dependencies in your environment first, then use the module directly.

Minimum dependencies used by the code:
- Gamma-ray Data Tools (GDT): `gdt-core`, `gdt-fermi`
- `numpy`, `scipy`, `matplotlib`
- `astropy`, `platformdirs`
- `PyQt6` (optional, for GUI tools)

Example (pip):

```bash
pip install gdt-core gdt-fermi numpy scipy matplotlib astropy platformdirs PyQt6
```
or simply:

```bash
pip install -r requirements.txt
```

### Optional: create a custom environment
You can isolate dependencies using a virtual environment before installing requirements.

**venv (Python built-in):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**conda:**

```bash
conda create -n specter python=3.10
conda activate specter
pip install -r requirements.txt
```

**uv:**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data directory configuration
SPECTER stores data using `platformdirs` by default. Default `platformdirs` locations (app name: `specter`):
- macOS: `~/Library/Application Support/specter`
- Windows: `C:\\Users\\<user>\\AppData\\Roaming\\specter`
- Linux: `~/.local/share/specter`

You can override the data directory in two ways:

1. **Environment variable**
   - `SPECTER_DATA_DIR=/path/to/data`
2. **Config file**
   - `~/.config/specter/config.json`
   - Example:
     ```json
     {"data_dir": "/path/to/data"}
     ```

## Quick start

```python
import specter

# Create a GRB object
grb = specter.GRB("160509374")

# Load TTE data
grb.load_tte(bin=True)

# Define time ranges
grb.src_range = (14, 17)
grb.bkgd_range = [[(-150, -25), (100, 200)]]

# Fit background and spectra
grb.fit_backgrounds(order=1)
grb.fit_spectra(models=["band"], stat="PG-Stat")

# Compute T90 and energetics
grb.calc_t90()
eiso = grb.calc_eiso(redshift=1.23)
```

> Tip: Many methods are interactive and may open Qt or Matplotlib windows for selection and visualization.

## GUI example

```python
import specter

grb = specter.GRB("160509374")
grb.load_tte(bin=True)

# Launch GUI windows (PHA viewers; FitPlotter if a spectral fitter exists)
grb.show_gui()

# Or open a single detector window
grb.show_gui(detector="n3")
```

## Bayesian blocks example

```python
import specter

grb = specter.GRB("160509374")

# Load and bin TTE data (Bayesian blocks expects evenly binned data)
grb.load_tte(bin=True, resolution=0.064)

# Run Bayesian blocks on a detector and show the plot
grb.bayesian_blocks(detector="n3", p0=0.05, show_plot=True)
```

## Fermi orbit plot example

```python
import specter

grb = specter.GRB("160509374")

# Plot the spacecraft orbit around the trigger time
grb.plot_orbit()

# Optionally save the plot to the data directory
grb.plot_orbit(save=True)
```

## Repository layout
- **specter.py** — primary `GRB` class and core analysis routines
- **analysis/** — algorithms (Bayesian blocks, structure function, FFT tools, energetics)
- **gui/** — PyQt widgets for interactive fitting/selection

## Notes
- The code relies on GDT’s Fermi/GBM support for file handling and response data.
- Some plots and GUI tools require a working Qt backend.

## License
TBD
