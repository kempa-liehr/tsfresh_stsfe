#! /bin/bash

# Boilerplate
set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
IFS=$'\n\t'
# End of boilerplate


# Workflow

# 1. Create a venv with specific python version "x"
# 2. Install all dependancies
# 3. Run pytest
# 4. Write results to a specific log file for python version "x"

declare -a python_versions=("3.9.5" "3.8.5")

# Uncomment this line for all prod versions of python
declare -a available_python_versions=($(pyenv install -l | grep -v '[A-Za-z]'))

echo available python versions:\n

for version in ${available_python_versions[@]}
do
echo ${version}\n
done


for version in ${python_versions[@]}
do
echo Installing ${version}\n
# install python version
pyenv install ${version}
done


# Install every version of python
