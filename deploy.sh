#!/bin/bash
sudo rm -rf "$(pwd)/wheelhouse/"
sudo rm -r "$(pwd)/dist"
sudo docker build -t wheelbuild "$(pwd)"
sudo docker run -v "$(pwd)":/io wheelbuild
python setup.py sdist
twine upload  wheelhouse/*-manylinux{1,2010}_x86_64.whl dist/*

