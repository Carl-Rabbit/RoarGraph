#!/bin/bash
rm -rf build tmp dist roargraph.egg-info
python setup.py bdist_wheel
pip uninstall roargraph -y 
pip install dist/roargraph-1.0.0-cp310-cp310-linux_x86_64.whl
rm -rf build tmp roargraph.egg-info