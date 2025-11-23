import sys
import os
import pytest

# ----------------------------------------------------------------------
# Add src folder to sys.path so Python can find the package
# ----------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ----------------------------------------------------------------------
# Add athena_read path if it's local and not installed
# ----------------------------------------------------------------------
athena_read_path = os.path.join(project_root, "athena_collapsingbox", "vis", "python")
if athena_read_path not in sys.path:
    sys.path.insert(0, athena_read_path)

# ----------------------------------------------------------------------
# Fixtures for test data
# ----------------------------------------------------------------------

@pytest.fixture
def sample_dir(request):
    """
    Path to a test data subdirectory.

    Can be overridden by a parametrized test using `request.param`.
    Defaults to 'one_vortex_collapse'.
    """
    subdir = getattr(request, "param", "one_vortex_collapse")
    path = os.path.join(os.path.dirname(__file__), "sample_data", subdir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample data directory {path} does not exist")
    return path


@pytest.fixture
def hdf_files(sample_dir):
    """
    Sorted list of .athdf files in the sample_dir.
    """
    files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".athdf")])
    if len(files) == 0:
        raise FileNotFoundError(f"No .athdf files in {sample_dir}")
    return files


@pytest.fixture
def hst_file(sample_dir):
    """
    Path to the .hst file in the sample_dir.
    """
    f = os.path.join(sample_dir, "OneVortex.hst")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing OneVortex.hst in {sample_dir}")
    return f
