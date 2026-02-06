#!/bin/bash

PYTHON_PATH=$(which python3.10)
echo "Using Python 3.10 from $PYTHON_PATH..."

# Flushes and rebuilds venv
rm -rf venv
$PYTHON_PATH -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# TODO not sure if this is necessary
cat <<EOF > venv/lib/python3.10/site-packages/cmeel.pth
$(pwd)/venv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages
EOF