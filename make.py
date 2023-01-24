from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from gammapy.estimators import ImageProfileEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.maps import Map


def measure_radial_profile(filename, threshold):
    data = Map.read(filename)
    x_edges = Angle(np.linspace(0, 6, 60), "arcsec")

    peaks = find_peaks(data, threshold=threshold)[:1]
    center = SkyCoord(peaks["ra"], peaks["dec"])

    est = ImageProfileEstimator(
        x_edges=x_edges, axis="radial", center=center, method="mean"
    )

    profile = est.run(data)
    return profile.normalize("integral")


def plot_profiles():
    profile_counts = measure_radial_profile(
        "broadband/19692/counts.fits", threshold=100
    )

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()

    profile_counts.plot(ax=ax, label="Counts profile", lw=2)

    for filename in Path("broadband/19692/").glob("*.psf"):

        profile_psf = measure_radial_profile(filename=filename, threshold=0.0001)

        try:
            blur = float(filename.name.split("-")[-1].replace(".psf", ""))
            pileup = False
        except ValueError:
            blur = float(filename.name.split("-")[-3].replace(".psf", ""))
            pileup = True

        ls = "-" if not pileup else "--"
        profile_psf.plot(
            ax=ax, label=f"PSF profile (blur={blur}, pilup={pileup})", alpha=0.4, ls=ls
        )

        # profile_psf_scaled = deepcopy(profile_psf)

        # for column in ["x_min", "x_max", "x_ref"]:
        #     profile_psf_scaled.table[column] /= 2.

        # profile_psf_scaled.plot(ax=ax, label="PSF profile, scaled")

        ax.set_xlim(0, 3.5)
        ax.set_yscale("log")
        ax.set_xlabel("Offset / arcsec")
        ax.set_ylabel("Profile / A.U.")
        plt.legend()
        plt.savefig("profiles-int-normed.png", dpi=300)


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
    plot_profiles()
    plot_images()
