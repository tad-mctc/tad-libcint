"""
Setup for C Extension.
"""

from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

module_name: str = "dxtblibs"
ext_name: str = "dxtblibs.pyscflibs"

file_dir: Path = Path(__file__).resolve().parent

# Arguments for CMake
CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DDISABLE_DFT=ON",
    "-DBUILD_LIBCINT=ON",
    "-DWITH_F12=OFF",
    "-DPYPZPX=OFF",
    # '-DWITH_RANGE_COULOMB=OFF', # no variable in CMakeLists.txt
]


class CMakeExtension(Extension):
    """
    Extension module for CMake.
    """

    def __init__(self, name: str, sourcedir: str = ""):
        """
        Initialize a new CMakeExtension.

        Parameters
        ----------
        name : str
            The name of the extension.
        sourcedir : str, optional
            The source directory of the extension.
        """
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


class CMakeBuildExt(build_ext):
    """
    Build extension module for CMake.
    """

    def run(self):
        """
        Run the extension building process.
        """
        extension = self.extensions[0]
        assert extension.name == ext_name
        self.build_extension(self.extensions[0])

    def build_extension(self, ext: Extension):
        """
        Build an extension.

        Parameters
        ----------
        ext : Extension
            The extension to build.
        """
        self.construct_extension(ext)

    def construct_extension(self, ext: Extension):
        """
        Construct an extension.

        Parameters
        ----------
        ext : Extension
            The extension to construct.
        """
        # libraries from PySCF
        lib_dir = file_dir / "src" / module_name / "libs"
        build_dir = Path(self.build_temp)
        self.announce(
            f"Compiling libraries from PySCF from {lib_dir} to {build_dir}", level=3
        )
        self.build_cmake(ext, lib_dir, build_dir)

    def build_cmake(self, ext: Extension, lib_dir: Path, build_dir: Path):
        """
        Build an extension with CMake.

        Parameters
        ----------
        ext : Extension
            The extension to build.
        lib_dir : Path
            The library directory of the extension.
        build_dir : Path
            The build directory for CMake.
        """
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        self.announce("Configuring cmake", level=3)
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extdir)] + CMAKE_ARGS
        cmd = ["cmake", f"-S{lib_dir}", f"-B{build_dir}"] + cmake_args
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ["cmake", "--build", build_dir, "-j"]
        self.spawn(cmd)


setup(
    ext_modules=[CMakeExtension(ext_name)],
    cmdclass={"build_ext": CMakeBuildExt},
)
