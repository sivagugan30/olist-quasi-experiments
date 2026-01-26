# Olist Quasi-Experiments Makefile
# Provides commands for reproducible setup and execution

.PHONY: all setup venv install download data notebooks clean help streamlit jupyter run-all run-eda run-rd run-did run-ship run-iv

# Python version requirement
PYTHON_VERSION = 3.12
VENV_NAME = olist-qe
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip

# Default target
all: setup data

# Help message
help:
	@echo "Olist Quasi-Experiments - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup      - Create venv, install deps, download data"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make download   - Download Olist dataset from Kaggle"
	@echo "  make data       - Download and preprocess data"
	@echo ""
	@echo "Analysis Commands (Scripts):"
	@echo "  make run-all    - Run all analysis scripts"
	@echo "  make run-eda    - Run EDA script"
	@echo "  make run-rd     - Run Deadline RD analysis"
	@echo "  make run-did    - Run Truckers Strike DiD analysis"
	@echo "  make run-ship   - Run Shipping Threshold RD analysis"
	@echo "  make run-iv     - Run Installments IV analysis"
	@echo ""
	@echo "Interactive Commands:"
	@echo "  make streamlit  - Start Streamlit dashboard"
	@echo "  make jupyter    - Start JupyterLab server"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean      - Remove generated files"
	@echo "  make clean-all  - Remove venv and all generated files"
	@echo "  make lint       - Run code linters"
	@echo "  make format     - Format code with black"
	@echo "  make test       - Run tests"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV_NAME)
	@echo "Virtual environment created at $(VENV_NAME)/"
	@echo "Activate with: source $(VENV_NAME)/bin/activate"

# Install dependencies
install: venv
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .
	@echo "Dependencies installed successfully!"

# Download Olist dataset from Kaggle
download:
	@echo "Downloading Olist dataset..."
	@$(PYTHON) -m src.data.download
	@echo "Download complete!"

# Create processed dataset
data: download
	@echo "Creating analysis dataset..."
	@$(PYTHON) -c "from src.data import load_all_tables, load_or_create_analysis_dataset; \
		tables = load_all_tables(exclude=['geolocation']); \
		df = load_or_create_analysis_dataset(tables, force_rebuild=True); \
		print(f'Created dataset with {len(df):,} rows')"
	@echo "Data processing complete!"

# Full setup: venv + install + data
setup: install data
	@echo ""
	@echo "=========================================="
	@echo "Setup complete!"
	@echo "=========================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate venv: source $(VENV_NAME)/bin/activate"
	@echo "  2. Start Jupyter: make jupyter"
	@echo "  3. Open notebooks/01_eda.ipynb"
	@echo ""

# Start JupyterLab
jupyter:
	@echo "Starting JupyterLab..."
	@$(VENV_NAME)/bin/jupyter lab --notebook-dir=notebooks

# Run EDA notebook
eda:
	@echo "Running EDA notebook..."
	@$(PYTHON) -m jupyter nbconvert --to notebook --execute \
		--output 01_eda_executed.ipynb \
		notebooks/01_eda.ipynb
	@echo "EDA notebook executed!"

# Run all notebooks
notebooks:
	@echo "Running all notebooks..."
	@for nb in notebooks/*.ipynb; do \
		echo "Running $$nb..."; \
		$(PYTHON) -m jupyter nbconvert --to notebook --execute \
			--output "$$(basename $$nb .ipynb)_executed.ipynb" \
			"$$nb" || true; \
	done
	@echo "All notebooks executed!"

# Start Streamlit dashboard
streamlit:
	@echo "Starting Streamlit dashboard..."
	@$(VENV_NAME)/bin/streamlit run app.py

# Run all analysis scripts
run-all:
	@echo "Running all analysis scripts..."
	@$(PYTHON) scripts/run_all.py

# Run individual analysis scripts
run-eda:
	@echo "Running EDA script..."
	@$(PYTHON) scripts/run_eda.py

run-rd:
	@echo "Running Deadline RD analysis..."
	@$(PYTHON) scripts/run_deadline_rd.py

run-did:
	@echo "Running Truckers Strike DiD analysis..."
	@$(PYTHON) scripts/run_truckers_strike_did.py

run-ship:
	@echo "Running Shipping Threshold RD analysis..."
	@$(PYTHON) scripts/run_shipping_threshold_rd.py

run-iv:
	@echo "Running Installments IV analysis..."
	@$(PYTHON) scripts/run_installments_iv.py

# Run linters
lint:
	@echo "Running linters..."
	@$(VENV_NAME)/bin/ruff check src/ notebooks/ || true
	@$(VENV_NAME)/bin/mypy src/ --ignore-missing-imports || true

# Format code
format:
	@echo "Formatting code..."
	@$(VENV_NAME)/bin/black src/ notebooks/
	@$(VENV_NAME)/bin/isort src/ notebooks/

# Run tests
test:
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf data/processed/*.parquet
	@rm -rf data/processed/*.csv
	@rm -rf reports/figures/*.png
	@rm -rf reports/figures/*.html
	@rm -rf reports/figures/*.json
	@rm -rf notebooks/*_executed.ipynb
	@rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	@rm -rf .pytest_cache .mypy_cache
	@echo "Clean complete!"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_NAME)
	@rm -rf *.egg-info
	@rm -rf data/raw/*.csv
	@rm -rf data/raw/*.zip
	@echo "Full clean complete!"
