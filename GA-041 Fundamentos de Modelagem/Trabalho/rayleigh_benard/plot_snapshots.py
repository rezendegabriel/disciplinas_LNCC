"""
Plot 2D cartesian snapshots.

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

#%% LIBRARIES

from dedalus.extras import plot_tools
from dedalus.tools import post
from dedalus.tools.parallel import Sync
from docopt import docopt

import h5py
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import yaml

matplotlib.use("Agg")

def load_params():
    with open(f"params_rayleigh_benard.yaml") as stream:
        params = yaml.safe_load(stream)

    return params

#%% PLOT SNAPSHOTS

def plot_snapshots(filename, start, count, output):
    params = load_params() # Rayleigh-Bénard parameters

    # Plot settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r'\usepackage{amsfonts}',
        "text.usetex": True
    })
    tasks = ["buoyancy", "vorticity"]
    titles = ["Empuxo", "Vorticidade"]
    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: "t = {:.2f}".format(sim_time)
    savename_func = lambda write: "write_{:03}.png".format(write)

    # Layout
    nrows, ncols = 2, 1
    image = plot_tools.Box(params["Lx"], params["Lz"])
    pad = plot_tools.Frame(0.3, 0.3, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode="r") as file:
        for index in range(start, start + count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])

                # Call 3D plotting helper, slicing in time
                dset = file["tasks"][task]
                axes, colorbar = plot_tools.plot_bot_3d(dset, 0, index, axes=axes, visible_axes=True, title=titles[n], cmap="RdBu_r")
                axes.set_xlabel(r"$y$")
                axes.set_ylabel(r"$z$")
                axes.set_xticks([0, params["Lx"]/2, params["Lx"]])
                axes.set_yticks([0, params["Lz"]/2, params["Lz"]])
                v_min, v_max = colorbar.get_xlim()
                colorbar.set_xticks([v_min, v_max])
                colorbar.set_xticklabels([f"{v_min:.2f}", f"{v_max:.2f}"])

            # Add time title
            title = title_func(file["scales/sim_time"][index])
            fig.suptitle(title)

            # Save figure
            savename = savename_func(file["scales/write_number"][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight", pad_inches=0)
            fig.clear()

    plt.close(fig)

#%% MAIN

if __name__ == "__main__":
    args = docopt(__doc__)
    output_path = pathlib.Path(args["--output"]).absolute()

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args["<files>"], plot_snapshots, output=output_path)