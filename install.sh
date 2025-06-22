#!/bin/bash

python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt

echo -e "\e[31m Installing shelf_gym\e[0m"
pip install -e .
cd shelf_gym/utils/scikit-geometry
echo -e "\e[31m Installing scikit-geometry\e[0m"
sed -i 's/pybind11>=2.3,<2.8/pybind11>=2.3/g' setup.py
mkdir wheel
python -m pip wheel . -w wheel
python -m pip install -e .
echo -e "\e[31m Everything installed\e[0m"
cd ../../..