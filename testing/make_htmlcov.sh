

### From within project root
#' sadly coveragerc can't be in a different directory
pytest tests -n 3 --cov --cov-report html:testing/htmlcov --cov-config .coveragerc 
