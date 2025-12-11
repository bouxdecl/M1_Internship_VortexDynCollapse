import os
import numpy as np
import pytest

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files,
    get_collapse_profile,
    add_collapse_param,
    pressure_from_conservatives,
    load_cons_with_collapse,
    collapse_param_decomposition,
)



# ======================================================================
# Tests: get_hdf_files
# ======================================================================

def test_get_hdf_files(sample_dir):
    files = get_hdf_files(sample_dir, verbose=False)
    assert len(files) > 0
    assert all(f.endswith(".athdf") for f in files)


# ======================================================================
# Tests: open_hdf_files
# ======================================================================

def test_open_hdf_files_basic(hdf_files):
    data = open_hdf_files(hdf_files, read_every=1, adia=True)

    # Required keys
    for key in ["x1", "x2", "x3", "time", "rho", "v1", "v2", "v3", "Etot"]:
        assert key in data

    # Dimensions consistent
    Nt = len(hdf_files)
    assert data["time"].shape == (Nt,)
    assert data["rho"].shape[0] == Nt
    assert data["v1"].shape == data["rho"].shape


def test_open_hdf_files_no_adiabatic(hdf_files):
    data = open_hdf_files(hdf_files, adia=False)
    assert "Etot" not in data or data["Etot"] is None


# ======================================================================
# Tests: get_collapse_profile
# ======================================================================

def test_get_collapse_profile_success(sample_dir):
    t, R, Lz = get_collapse_profile(sample_dir)

    assert t.ndim == 1
    assert R.shape == t.shape
    assert Lz.shape == t.shape
    assert np.all(R > 0)
    assert np.all(Lz > 0)


# ======================================================================
# Tests: add_collapse_param
# ======================================================================

def test_add_collapse_param_interpolation(hdf_files, sample_dir):
    # Load time series
    dic = open_hdf_files_cons(hdf_files)

    # Load HST values
    t_hst, R_hst, Lz_hst = get_collapse_profile(sample_dir)

    dic2 = add_collapse_param(dic, t_hst, R_hst, Lz_hst)

    assert "Rglobal" in dic2 and "Lzglobal" in dic2
    t = dic["time"]
    assert np.allclose(dic2["Rglobal"], np.interp(t, t_hst, R_hst))
    assert np.allclose(dic2["Lzglobal"], np.interp(t, t_hst, Lz_hst))


def test_add_collapse_param_out_of_bounds():
    dic = {"time": np.array([-0.1, 0.5, 2.0])}  # -0.1 is out-of-bounds
    t_hst = np.array([0, 1, 2])
    R = np.array([1, 2, 3])
    Lz = np.array([10, 20, 30])

    with pytest.raises(ValueError, match="outside the .hst time range"):
        add_collapse_param(dic, t_hst, R, Lz)


# ======================================================================
# Tests: load_cons_with_collapse
# ======================================================================

def test_load_cons_with_collapse(sample_dir, hdf_files):
    dic = load_cons_with_collapse(sample_dir, hdf_files)
    assert "Rglobal" in dic and "Lzglobal" in dic


# ======================================================================
# Tests: pressure_from_conservatives
# ======================================================================

def test_pressure_from_conservatives():
    rho = np.array([1.0, 2.0, 3.0])
    Etot = np.array([10.0, 12.0, 14.0])
    v1 = np.array([1.0, 0.0, 1.0])
    v2 = np.zeros(3)
    v3 = np.zeros(3)

    p = pressure_from_conservatives(rho, Etot, v1, v2, v3, gamma=5/3)

    # Manual computation
    KE = 0.5 * rho * v1**2
    e = Etot - KE
    expected = (5/3 - 1) * e

    assert np.allclose(p, expected)


# ======================================================================
# Tests: collapse_param_decomposition
# ======================================================================

def test_collapse_param_decomposition_valid():
    R = np.array([1.0, 2.0, 4.0])
    Lz = np.array([2.0, 4.0, 8.0])

    S, alpha = collapse_param_decomposition(R, Lz)

    assert S.shape == R.shape
    assert alpha.shape == R.shape
    assert np.allclose(
        S, (R / R[0]**2 * Lz / Lz[0]) ** (1/3)
    )
    assert np.allclose(
        alpha, (R / R[0]) / (Lz / Lz[0]) ** (1/3)
    )


def test_collapse_param_decomposition_shape_error():
    R = np.array([1, 2])
    Lz = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        collapse_param_decomposition(R, Lz)
