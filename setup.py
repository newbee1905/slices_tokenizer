import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
	def __init__(self, name: str, sourcedir: str = "") -> None:
		super().__init__(name, sources=[])
		self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
	def build_extension(self, ext: CMakeExtension) -> None:
		ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
		extdir = ext_fullpath.parent.resolve()

		debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
		cfg = "Debug" if debug else "Release"

		cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

		cmake_args = [
			f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
			f"-DPYTHON_EXECUTABLE={sys.executable}",
			f"-DCMAKE_BUILD_TYPE={cfg}",
			f"-DVERSION_INFO={self.distribution.get_version()}",
		]
		build_args = []

		if "CMAKE_ARGS" in os.environ:
			cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

		if self.compiler.compiler_type == "msvc":
			cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
			build_args += ["--config", cfg]
		else:
			cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

		if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
			if hasattr(self, "parallel") and self.parallel:
				build_args += [f"-j{self.parallel}"]

		build_temp = Path(self.build_temp) / ext.name
		if not build_temp.exists():
			build_temp.mkdir(parents=True)

		print("-" * 10, "Configuring CMake project", "-" * 10)
		subprocess.run(
			["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
		)

		print("-" * 10, "Building CMake project", "-" * 10)
		subprocess.run(
			["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
		)

setup(
	name="SLICESTokenizer",
	version="0.1.0",
	author="newbee1905",
	author_email="beenewminh@outlook.com",
	description="A C++ tokenizer for the SLICES format, with Python bindings.",
	long_description="",
	ext_modules=[CMakeExtension("slices_tokenizer")],
	cmdclass={"build_ext": CMakeBuild},
	zip_safe=False,
	python_requires=">=3.8",
)
