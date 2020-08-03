# Setup file

# Setup a python3 virtual environment
PYTHONPATH=""
virtualenv -p /usr/bin/python3 env
source env/bin/activate

# Install requirements
pip install -r requirements.txt

