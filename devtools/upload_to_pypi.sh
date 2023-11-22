
# Convert README.md to README_pypi.md
python devtools/readme_for_pypi.py -i README.md

# Update version on pyproject.toml
# !! Don't do this, removes comments
# python devtools/update_version.py -i pyproject.toml -o pyproject_test.toml

# Remove old build
rm -r dist

# BUILD
python -m build

# Require API token if two-factor was enabled 
# provide API token as password
twine upload --repository testpypi dist/* --username __token__

# UPLOAD TO REAL PyPi
twine upload dist/* --username __token__

# use this in a venv
pip install -i https://test.pypi.org/simple/ plotastic