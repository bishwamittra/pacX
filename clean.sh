find . -type d -name  "__pycache__" -exec rm -r {} +
rm -r temp*
find . -type d -name  ".ipynb_checkpoints" -exec rm -r {} +