# Rayleigh-Bénard convection

Dedalus script simulating 2D horizontally-periodic Rayleigh-Bénard convection.

## Python environment (Linux)

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

## Run

See parameter file `params_rayleigh_benard.yaml`. To run and plot using e.g. 4 processes:

    mpiexec -n 4 python3 rayleigh_benard.py
    mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5