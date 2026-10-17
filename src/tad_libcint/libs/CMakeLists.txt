# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
# Modifications for tad-libcint by Grimme Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared libraries for tad-libcint:
#
# - libcint (git submodule): the integrals
# - libcgto: PySCF's driver that fills a whole two-centre integral matrix
#
# Both get a suffix, so that their file names and SONAMEs do not collide with
# PySCF's (or a system) libcint/libcgto when loaded into the same process.

cmake_minimum_required(VERSION 3.18)
project(tad_libcint C)

# fixed, since the Python side loads the libraries by these names
set(TAD_LIBRARY_SUFFIX "_tad")

set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

# The sources call functions through pointers declared without a prototype,
# e.g. `int (*intor)()`, which C23 (the default of GCC >= 15) rejects.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)

option(ENABLE_OPENMP "Parallelize over shell pairs with OpenMP" ON)

# the libraries find each other next to themselves
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Intel's runtime libraries (libimf, libiomp5, ...) are only on the library
# path inside the compiler environment (e.g. `module load intel`); find them in
# the compiler installation instead, so that the libraries also load without it
if(CMAKE_C_COMPILER_ID MATCHES "^Intel")
  find_library(
    INTEL_RUNTIME_LIBRARY imf HINTS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  )
  if(INTEL_RUNTIME_LIBRARY)
    get_filename_component(intel_lib_dir "${INTEL_RUNTIME_LIBRARY}" DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH "${intel_lib_dir}")
  endif()
endif()


# libcint (git submodule), configured through its options; fixed, since the
# Python side relies on them (e.g. spherical p functions as (py, pz, px))

set(PYPZPX ON)
set(WITH_F12 OFF)
set(WITH_RANGE_COULOMB OFF)
set(WITH_FORTRAN OFF)
set(WITH_CINT2_INTERFACE OFF)
set(MIN_EXPCUTOFF 20)
set(KEEP_GOING OFF)
add_subdirectory(libcint EXCLUDE_FROM_ALL)

# a single unversioned file (`libcint_tad.so`) instead of a symlink chain,
# which wheels would store as three copies
set_property(TARGET cint PROPERTY VERSION)
set_property(TARGET cint PROPERTY SOVERSION)


# PySCF's two-centre driver (`GTOint2c`) and its helpers

add_library(
  cgto SHARED
  gto/fill_int2c.c
  gto/fill_int2e.c # GTOmax_cache_size
  np_helper/np_helper.c # NPdcopy
  np_helper/pack_tril.c # NPdsymm_triu
)
target_include_directories(cgto PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(cgto PRIVATE cint m)

if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_C_FOUND AND MINGW)
    # The imported target links the import library of libgomp's DLL, which
    # bypasses `-static` (see below); the flag alone links libgomp statically.
    target_compile_options(cgto PRIVATE ${OpenMP_C_FLAGS})
    target_link_options(cgto PRIVATE ${OpenMP_C_FLAGS})
  elseif(OpenMP_C_FOUND)
    target_link_libraries(cgto PRIVATE OpenMP::OpenMP_C)
  endif()
endif()


foreach(target cint cgto)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${target}${TAD_LIBRARY_SUFFIX}")

  # `.so` also on macOS (instead of `.dylib`), like Python's own extensions
  if(NOT WIN32)
    set_target_properties(${target} PROPERTIES SUFFIX ".so")
  endif()

  # Python (>= 3.8) does not search the PATH for the DLLs a DLL depends on,
  # so MinGW's runtime (libgcc, libgomp, libwinpthread) is linked statically
  if(MINGW)
    target_link_options(${target} PRIVATE -static)
  endif()
endforeach()
