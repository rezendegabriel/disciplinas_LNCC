#%% LIBRARIES

from argparse import ArgumentParser

import dedalus.public as d3
import logging
import numpy as np
import yaml

logger = logging.getLogger(__name__)

#%% LOAD YAML PARAMETERS FILE

def load_params():
    with open(f"params_rayleigh_benard.yaml") as stream:
        params = yaml.safe_load(stream)

    return params

def prepare_parser(params):
    usage = "Parser for the rayleigh_benard.py"
    parser = ArgumentParser(description=usage)

    # Geometry and resolution parameters (discretization)
    parser.add_argument(
        "--Lx", type=float, default=params["Lx"],
        help="Horizontal length (y-axis) of the non-dimensionalized domain. Default: %(default)s."
    )
    parser.add_argument(
        "--Lz", type=float, default=params["Lz"],
        help="Vertical height (z-axis) of the non-dimensionalized domain. Rayleigh-Bénard occurs between z = 0 and z = Lz. Default: %(default)s."
    )
    parser.add_argument(
        "--Nx", type=int, default=params["Nx"],
        help="Number of points (Fourier coefficients) used to discretize the y-axis. Determines the horizontal resolution. Default: %(default)s."
    )
    parser.add_argument(
        "--Nz", type=int, default=params["Nz"],
        help="Number of points (Chebyshev coefficients) used to discretize the z-axis. Determines the vertical resolution. Default: %(default)s."
    )
    parser.add_argument(
        "--dealias", type=float, default=params["dealias"],
        help="Dealiasing factor (antialiasing). It`s used in spectral methods to avoid aliasing errors caused by the multiplication of nonlinear fields in Fourier space. A value of 1.5 is common and means that one third of the high-frequency coefficients are discarded. Default: %(default)s."
    )

    # Physical parameters (non-dimensionalized)
    parser.add_argument(
        "--rayleigh", type=float, default=params["rayleigh"],
        help="The Rayleigh number is the main parameter that measures the intensity of the thermal forcing (the heating at the base) in relation to heat and momentum diffusion. Default: %(default)s."
    )
    parser.add_argument(
        "--prandtl", type=float, default=params["prandtl"],
        help="The Prandtl number is a fluid property that compares the momentum diffusivity (kinematic viscosity) with thermal diffusivity. Default: %(default)s."
    )

    # Solver parameters
    parser.add_argument(
        "--stop_sim_time", type=int, default=params["stop_sim_time"],
        help="Simulation time (non-dimensional) at which execution should stop. Default: %(default)s."
    )
    parser.add_argument(
        "--time_stepper", type=str, default=params["time_stepper"],
        help="Temporary integrator. Default: %(default)s."
    )
    parser.add_argument(
        "--max_time_step", type=float, default=params["max_time_step"],
        help="The maximum time step (non-dimensional) that the solver can use. This ensures stability and accuracy in the integration. Default: %(default)s."
    )
    parser.add_argument(
        "--dtype", type=str, default=params["dtype"],
        help="Numerical precision. Default: %(default)s."
    )

    return params

#%% RAYLEIGH-BENARD SOLVER

def rayleigh_benard(args):
    # Geometry and resolution parameters (discretization)
    Lx, Lz = args.Lx, args.Lz # horizontal length and vertical height (boundary conditions)
    Nx, Nz = args.Nx, args.Nz # number of points used to discretization (Fourier and Chebyshev coefficients, respectively)
    dealias = args.dealias # antialiasing factor

    # Physical parameters (non-dimensionalized)
    rayleigh = args.rayleigh # Rayleigh number
    prandtl = args.prandtl # Prandtl number

    # Solver parameters
    stop_sim_time = args.stop_sim_time # simulation time
    time_stepper = getattr(d3, args.time_stepper) # temporary integrator
    max_time_step = args.max_time_step # maximum time step
    dtype = getattr(np, args.dtype) # numerical precision

    # Base functions
    coords = d3.CartesianCoordinates("x", "z") # define the cartesian coordinates system
    dist = d3.Distributor(coords, dtype=dtype) # parallelizes processing between cores
    x_basis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
    z_basis = d3.ChebyshevT(coords["z"], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Main field variables
    p = dist.Field(name="p", bases=(x_basis, z_basis)) # pressure (scalar field)
    b = dist.Field(name="b", bases=(x_basis, z_basis)) # temperature (scalar field)
    u = dist.VectorField(coords, name="u", bases=(x_basis, z_basis)) # velocity (vector field)

    # Auxiliary variables (used to impose the boundary conditions)
    tau_p = dist.Field(name="tau_p") # pressure gauge
    tau_b1 = dist.Field(name="tau_b1", bases=x_basis) # b(z=0)=0
    tau_b2 = dist.Field(name="tau_b2", bases=x_basis) # b(z=Lz)=0
    tau_u1 = dist.VectorField(coords, name="tau_u1", bases=x_basis) # u(z=0)=0
    tau_u2 = dist.VectorField(coords, name="tau_u2", bases=x_basis) # u(z=Lz)=0

    # Diffusion coefficients
    kappa = (rayleigh*prandtl)**(-1/2)
    nu = (rayleigh/prandtl)**(-1/2)

    # Auxiliary geometrical elements
    x, z = dist.local_grids(x_basis, z_basis) # grid
    ex, ez = coords.unit_vector_fields(dist) # unit vectors

    # Order reduction terms
    lift_basis = z_basis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # first-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # first-order reduction

    """
    Problem:
        First-order form: "div(f)" becomes "trace(grad_f)"
        First-order form: "lap(f)" becomes "div(grad_f)"
    """
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals()) # initial value problem
    problem.add_equation("trace(grad_u) + tau_p = 0") # continuity equation
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)") # energy equation
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)") # momentum equation

    # Buoyancy boundary conditions
    problem.add_equation("b(z=0) = Lz") # top plate (cold)
    problem.add_equation("b(z=Lz) = 0") # bottom plate (hot)

    # Vorticity boundary conditions (no-slip)
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("u(z=Lz) = 0")

    problem.add_equation("integ(p) = 0") # pressure gauge

    # Solver
    solver = problem.build_solver(time_stepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions (only b, u=0 and p=0)
    b.fill_random("g", seed=42, distribution="normal", scale=1e-3) # random noise
    b["g"] *= z*(Lz - z) # damp noise at walls
    b["g"] += Lz - z # add linear background

    # Analysis
    snapshots = solver.evaluator.add_file_handler("snapshots", sim_dt=0.25, max_writes=50)
    snapshots.add_task(b, name="buoyancy")
    snapshots.add_task(-d3.div(d3.skew(u)), name="vorticity")

    # Courant–Friedrichs–Lewy (CFL) condition
    CFL = d3.CFL(solver, initial_dt=max_time_step, cadence=10, safety=0.5, threshold=0.05,
                 max_change=1.5, min_change=0.5, max_dt=max_time_step)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name="Re") # Reynolds number

    try:
        logger.info("Starting main loop")
        while solver.proceed:
            timestep = CFL.compute_timestep() # 1. Determines the time step
            solver.step(timestep) # 2. Advance the solution in time
            if (solver.iteration-1) % 10 == 0: # 3. Status report
                max_Re = flow.max("Re")
                logger.info("Iteration=%i, Time=%e, dt=%e, max(Re)=%f" %(solver.iteration, solver.sim_time, timestep, max_Re))
    except:
        logger.error("Exception raised, triggering end of main loop.")
        raise
    finally:
        solver.log_stats()

#%% MAIN

if __name__ == "__main__":
    params = load_params()
    parser = prepare_parser(params)
    args = parser.parse_args()
    print("rayleigh_benard.py arguments: " + f"{args}")

    rayleigh_benard(args)