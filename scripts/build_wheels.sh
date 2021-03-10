#!/bin/bash
# Adapted from github.com/pypa/python-manylinux
cd io
echo "Current dir=$(pwd)"
echo "Platform=${PLAT}"
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Compile wheels
for PYBIN in /opt/python/*3*/bin; do
    "${PYBIN}/pip" install -r /io/devreqs.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*3*/bin/; do
    "${PYBIN}/pip" install nphil --no-index -f /io/wheelhouse
done
