pipx install pre-commit ruff
pre-commit install

echo "Setup precommit"

mkdir data/datafiles
git clone https://github.com/brendenlake/SCAN.git data/datafiles

echo "Installed the dataset"

pip install -r requirements.txt
echo "Installed requirements"
