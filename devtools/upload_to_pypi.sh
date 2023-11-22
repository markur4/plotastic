# pip install build

python -m build

# Require API token if two-factor was enabled 
# provide API token as password
twine upload --repository testpypi dist/* --username __token__

# use this in a venv
pip install -i https://test.pypi.org/simple/ plotastic==0.1.0