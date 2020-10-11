# Setup file

# Setup a python3 virtual environment
PYTHONPATH=""
virtualenv -p /usr/bin/python3.5 env
source env/bin/activate

# Install requirements
pip install -r requirements.txt


# Make shortest path cache files
python floyd_warshall.py 

# Make particle measurement cache files
python lut.py
