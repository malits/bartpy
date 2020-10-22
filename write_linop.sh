#!/bin/sh

# OLD Setup
# swig -python swig_scripts/linops_scripts/linop.i
# mv swig_scripts/linops_scripts/linop_wrap.c wrappers/linops_wrappers/
# mv swig_scripts/linops_scripts/linop.py setup_scripts/linops_setup/
# cd setup_scripts/linops_setup
# python3 setup_linop.py install
# cd ..

swig -python linop.i
python3 setup_linop.py install