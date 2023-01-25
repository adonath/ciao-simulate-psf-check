import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from gammapy.estimators import ImageProfile, ImageProfileEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.maps import Map
from gpxray.chandra.config import ChandraConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PATH_BASE = Path(__file__).parent


def measure_radial_profile(filename, threshold=0.01):
    """Measure radial profile of a map."""
    log.info(f"Reading {filename}")
    map_ = Map.read(filename)
    map_ /= map_.data.sum()
    x_edges = Angle(np.linspace(0, 6, 60), "arcsec")

    peaks = find_peaks(map_, threshold=threshold)[:1]
    center = SkyCoord(peaks["ra"], peaks["dec"])

    est = ImageProfileEstimator(
        x_edges=x_edges, axis="radial", center=center, method="mean"
    )

    profile = est.run(map_)
    return profile.normalize("integral")


def read_profile(filename):
    """Read radial profile from file."""
    log.info(f"Reading {filename}")
    table = Table.read(filename)
    return ImageProfile(table=table)


def measure_radial_profiles():
    """Measure radial profiles for all PSF maps in the data folder."""
    path_results = PATH_BASE / "results/profiles"
    path_results.mkdir(exist_ok=True)

    filenames = PATH_BASE.glob("config*.yaml")

    for filename in filenames:
        config = ChandraConfig.read(filename)

        path_input = PATH_BASE / f"{config.sub_name}" / f"{config.obs_id_ref}"

        for name, irf_config in config.irfs.items():
            filename_psf = path_input / f"psf-{config.psf_simulator}-{name}.fits.gz"
            profile = measure_radial_profile(filename_psf)

            meta_dict = irf_config.psf.dict()
            profile.table.meta.update(meta_dict)
            profile.table.meta["simulator"] = config.psf_simulator

            path = path_results / f"psf-{name}.fits.gz"
            log.info(f"Writing {path}")
            profile.table.write(path, overwrite=True)

    profile_counts = measure_radial_profile(path_input / "counts.fits.gz")
    path = path_results / "counts-profile.fits.gz"
    log.info(f"Writing {path}")
    profile_counts.table.write(path, overwrite=True)


def plot_profiles(filenames, title):
    path_results = PATH_BASE / "results"
    path_figures = path_results / "figures"
    path_figures.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()

    profile_counts = read_profile("results/profiles/counts-profile.fits.gz")
    profile_counts.plot(ax=ax, label="Counts profile", lw=2)

    for filename in filenames:
        profile_psf = read_profile(filename=filename)
        blur = profile_psf.table.meta["BLUR"]
        pileup = profile_psf.table.meta["PILEUP"]
        simulator = profile_psf.table.meta["simulator"]

        ls = "-" if simulator == "marx" else "--"

        profile_psf.plot(
            ax=ax,
            label=f"PSF profile {simulator} (blur={blur}, pileup={pileup})",
            alpha=0.4,
            ls=ls,
        )

    ax.set_xlim(0, 3.5)
    ax.set_yscale("log")
    ax.set_xlabel("Offset / arcsec")
    ax.set_ylabel("Profile / A.U.")
    plt.legend()

    filename = path_figures / f"{title}.png"
    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=300)


def plot_marx_vs_saotrace():
    filenames = PATH_BASE.glob("results/profiles/psf-*.fits.gz")
    plot_profiles(filenames, "marx-vs-saotrace-no-pileup")


def plot_images():
    counts = Map.read("broadband/19692/counts.fits")
    psf = Map.read("broadband/19692/psf-marx-pks-0637.psf")

    # interpolate on the same WCS
    psf = psf.interp_to_geom(counts.geom)

    # normalize
    counts = counts / counts.data.sum()
    psf = psf / psf.data.sum()

    fig, axes = plt.subplots(
        nrows=1, ncols=3, subplot_kw={"projection": counts.geom.wcs}, figsize=(12, 3)
    )

    counts.plot(axes[0], stretch="log")
    axes[0].set_title("Counts")

    psf.plot(axes[1], stretch="log")
    axes[1].set_title("PSF")

    difference = counts - psf

    val = 1e-3
    difference.plot(axes[2], cmap="RdBu", vmin=-val, vmax=val)
    axes[2].set_title("Difference")

    plt.savefig("images.png", dpi=300)


if __name__ == "__main__":
    measure_radial_profiles()
    plot_marx_vs_saotrace()
    # plot_images()
