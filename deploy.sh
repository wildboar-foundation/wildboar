#!/bin/bash
sudo rm -rf "$(pwd)/wheelhouse/"
sudo rm -r "$(pwd)/dist"
sudo docker run -v "$(pwd)":/io wheelbuild
python setup.py sdist
