#!/bin/bash
git pull
echo "Installing ART..."
cd ./dist
pip uninstall -y adversarial_robustness_toolbox-1.16.0-py3-none-any.whl
cd ..
python setup.py bdist_wheel
cd ./dist
pip install adversarial_robustness_toolbox-1.16.0-py3-none-any.whl
echo "ART installed successfully"


