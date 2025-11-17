#!/usr/bin/env python3
"""
Binary wrapper module for IOWarp Core command-line tools.

This module provides Python entry points that call the actual C++ binaries
installed in the package's bin/ directory. All binaries are prefixed with 'py_'
to avoid conflicts with system-installed versions.
"""

import os
import sys
import subprocess
from pathlib import Path


def _get_bin_path(bin_name):
    """
    Get the full path to a binary in the iowarp_core package.

    Args:
        bin_name: Name of the binary (without py_ prefix)

    Returns:
        Path to the binary
    """
    # Get the virtual environment's bin directory
    # The binaries are installed by CMake to {prefix}/bin/
    venv_bin = Path(sys.executable).parent
    bin_path = venv_bin / bin_name

    if not bin_path.exists():
        # Fallback: try the package bin directory
        package_dir = Path(__file__).parent
        bin_path = package_dir / "bin" / bin_name

        if not bin_path.exists():
            raise FileNotFoundError(
                f"Binary '{bin_name}' not found in {venv_bin} or {package_dir / 'bin'}. "
                "Please ensure iowarp-core is properly installed."
            )

    return str(bin_path)


def _run_binary(bin_name):
    """
    Run a binary with the provided command-line arguments.

    Args:
        bin_name: Name of the binary to run (without py_ prefix)
    """
    try:
        bin_path = _get_bin_path(bin_name)
        # Pass through all command-line arguments (excluding script name)
        result = subprocess.run([bin_path] + sys.argv[1:])
        sys.exit(result.returncode)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running {bin_name}: {e}", file=sys.stderr)
        sys.exit(1)


# Entry point functions for each binary
def chimaera_start_runtime():
    """Start the Chimaera runtime."""
    _run_binary("chimaera_start_runtime")


def chimaera_stop_runtime():
    """Stop the Chimaera runtime."""
    _run_binary("chimaera_stop_runtime")


def chimaera_compose():
    """Compose cluster configuration."""
    _run_binary("chimaera_compose")


def chi_refresh_repo():
    """Refresh the repository."""
    _run_binary("chi_refresh_repo")


def wrp_cae_omni():
    """Run the CAE OMNI processor."""
    _run_binary("wrp_cae_omni")


def test_binary_assim():
    """Run binary assimilation test."""
    _run_binary("test_binary_assim")


def test_error_handling():
    """Run error handling test."""
    _run_binary("test_error_handling")


def test_hdf5_assim():
    """Run HDF5 assimilation test."""
    _run_binary("test_hdf5_assim")


def test_range_assim():
    """Run range assimilation test."""
    _run_binary("test_range_assim")


if __name__ == "__main__":
    print("This module provides entry points for IOWarp Core binaries.")
    print("Available commands (with py_ prefix):")
    print("  - py_chimaera_start_runtime")
    print("  - py_chimaera_stop_runtime")
    print("  - py_chimaera_compose")
    print("  - py_chi_refresh_repo")
    print("  - py_wrp_cae_omni")
    print("  - py_test_binary_assim")
    print("  - py_test_error_handling")
    print("  - py_test_hdf5_assim")
    print("  - py_test_range_assim")
