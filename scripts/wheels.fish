#!/usr/bin/env fish

# environments are created with:
# mamba create -n wheel-<version> --yes python=<version> auditwheel c-compiler fortran-compiler cmake meson numpy python pip python-build pkgconfig patchelf unzip wheel

set -l VERSIONS "38" "39" "310" "311"

# Trap SIGINT and SIGTERM signals to exit the whole script
function handle_exit
    echo "Exited!"
    exit
end
trap handle_exit SIGINT SIGTERM

for VERSION in $VERSIONS
    conda activate wheel-$VERSION
    echo "Activate wheel-$VERSION"
    python -m build --wheel
    auditwheel repair -w wheels --plat manylinux_2_12_x86_64 dist/dxtblibs-*-cp$VERSION-cp$VERSION-linux_x86_64.whl
    conda deactivate
    rm -rf build/ dist/
end
