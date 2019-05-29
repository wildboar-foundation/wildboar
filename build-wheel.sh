#!/bin/bash
cmd="wheel /io/ -w /io/wheelhouse/"

echo "Building for Python 3.6"
/opt/python/cp36-cp36m/bin/pip $cmd

echo "Building for Python 3.7"
/opt/python/cp37-cp37m/bin/pip $cmd

for whl in /io/wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat manylinux2010_x86_64 -w /io/wheelhouse/
done


