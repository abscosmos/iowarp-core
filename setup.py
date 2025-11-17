#!/usr/bin/env python3
"""
Setup script for iowarp-core package.
Builds and installs C++ components using CMake in the correct order.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


class CMakeExtension(Extension):
    """Extension class for CMake-based C++ projects."""

    def __init__(self, name, sourcedir="", repo_url="", **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)
        self.repo_url = repo_url


class CustomSDist(sdist):
    """Custom sdist command that ensures git submodules are fully included."""

    def run(self):
        """Ensure git submodules are initialized before creating source distribution."""
        # Check if we're in a git repository
        if os.path.exists(".git"):
            print("\n" + "="*60)
            print("Ensuring git submodules are included in source distribution")
            print("="*60 + "\n")

            try:
                # Initialize and update submodules to ensure they're present
                subprocess.check_call(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                print("✓ Git submodules initialized successfully")

                # List submodules to verify
                result = subprocess.run(
                    ["git", "submodule", "status"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout:
                    print("\nSubmodules found:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"  {line}")

                print("\nNote: MANIFEST.in will include submodule files in the tarball")
                print("")

            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not initialize git submodules: {e}")
                print("Source distribution may be incomplete!")
        else:
            print("Not a git repository - skipping submodule initialization")

        # Call parent sdist command to create the distribution
        super().run()


class CMakeBuild(build_ext):
    """Custom build command that builds IOWarp core using CMake presets."""

    # Single repository for all components
    REPO_URL = "https://github.com/iowarp/core"

    def run(self):
        """Build IOWarp core following the quick installation steps."""
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build iowarp-core. "
                "Install with: pip install cmake"
            )

        # Create build directory
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Build the unified core
        self.build_iowarp_core(build_temp)

    def build_iowarp_core(self, build_temp):
        """Build IOWarp core using install.sh script."""
        print(f"\n{'='*60}")
        print(f"Building IOWarp Core using install.sh")
        print(f"{'='*60}\n")

        # Determine install prefix based on whether we're bundling binaries
        # For IOWarp Core, install directly to the Python environment (sys.prefix)
        # This is the standard approach for packages with C++ libraries that need to be
        # found by CMake and linked by other applications.
        # Libraries → {sys.prefix}/lib/
        # Headers → {sys.prefix}/include/
        # CMake configs → {sys.prefix}/lib/cmake/
        bundle_binaries = os.environ.get("IOWARP_BUNDLE_BINARIES", "OFF").upper() == "ON"
        if bundle_binaries:
            # Install to a staging directory that we'll copy into the wheel
            # (Not recommended for IOWarp - use IOWARP_BUNDLE_BINARIES=ON to enable)
            install_prefix = build_temp / "install"
        else:
            # Install directly to Python environment prefix
            # This ensures libraries/headers are in standard locations
            install_prefix = Path(sys.prefix).absolute()

        print(f"Install prefix: {install_prefix}")

        # Find install.sh in package root
        package_root = Path(__file__).parent.absolute()
        install_script = package_root / "install.sh"

        if not install_script.exists():
            raise RuntimeError(f"install.sh not found at {install_script}")

        # Make install.sh executable
        install_script.chmod(0o755)

        # Prepare environment for install.sh
        env = os.environ.copy()
        env["INSTALL_PREFIX"] = str(install_prefix)
        # INSTALL_PREFIX points to venv root, e.g., /path/to/venv
        # This ensures standard layout:
        #   - Binaries:    {venv}/bin/
        #   - Libraries:   {venv}/lib/
        #   - Headers:     {venv}/include/
        #   - CMake:       {venv}/lib/cmake/
        # Python bindings install to site-packages via nanobind

        # Determine number of parallel jobs
        if hasattr(self, "parallel") and self.parallel:
            env["BUILD_JOBS"] = str(self.parallel)
        else:
            import multiprocessing
            env["BUILD_JOBS"] = str(multiprocessing.cpu_count())

        print(f"\nRunning install.sh with:")
        print(f"  INSTALL_PREFIX={env['INSTALL_PREFIX']}")
        print(f"  BUILD_JOBS={env['BUILD_JOBS']}\n")

        # Run install.sh and capture output for debugging
        result = subprocess.run(
            [str(install_script)],
            cwd=package_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Print all install.sh output
        if result.stdout:
            print(result.stdout)

        # Check for errors
        if result.returncode != 0:
            print(f"\nERROR: install.sh failed with exit code {result.returncode}\n")
            sys.exit(result.returncode)

        print(f"\nIOWarp core built and installed successfully!\n")


# Create extensions list
# Always include the CMake build extension so that source distributions work correctly.
# The IOWARP_BUNDLE_BINARIES flag controls whether binaries are bundled into the wheel
# or installed to the system prefix (for source installs).
ext_modules = [
    CMakeExtension(
        "iowarp_core._native",
        sourcedir=".",
    )
]
cmdclass = {
    "build_ext": CMakeBuild,
    "sdist": CustomSDist,
}


if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
