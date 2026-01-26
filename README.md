# Olist Brazilian E-Commerce: Quasi-Experimental Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive causal inference project using the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle. This project implements four quasi-experimental analyses to identify causal effects in e-commerce operations.

## Research Questions

| Analysis | Method | Treatment | Outcome |
|----------|--------|-----------|---------|
| **1. Deadline RD** | Sharp RD | Late vs On-time Delivery | Review Score |
| **2. Truckers Strike** | Diff-in-Diff | May 21, 2018 Strike | Delivery Time, Cancellations |
| **3. Shipping Threshold** | Fuzzy RD | Free Shipping Eligibility | AOV, Items per Order |
| **4. Installments** | Fuzzy RD + IV | Payment Installments | Conversion, AOV |

## Project Structure

```
olist-quasi-experiments/
├── data/
│   ├── raw/              # Original Kaggle data (downloaded via script)
│   └── processed/        # Processed parquet files
├── scripts/              # Analysis scripts (recommended)
│   ├── run_eda.py        # Exploratory Data Analysis
│   ├── run_deadline_rd.py        # Deadline RD analysis
│   ├── run_truckers_strike_did.py # Strike DiD analysis
│   ├── run_shipping_threshold_rd.py # Shipping threshold RD
│   ├── run_installments_iv.py    # Installments IV analysis
│   └── run_all.py        # Run all analyses
├── notebooks/            # Jupyter notebooks (alternative)
│   └── 01_eda.ipynb      # Exploratory Data Analysis
├── reports/
│   ├── figures/          # Plotly charts (HTML, PNG, JSON)
│   └── decision_memo.md  # Summary of findings
├── src/
│   ├── data/             # Data loading & preprocessing
│   ├── analysis/         # RD, DiD, IV estimation methods
│   └── visualization/    # Plotly chart utilities
├── app.py                # Streamlit dashboard
├── tests/                # Unit tests
├── Makefile              # Automation commands
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+ (or 3.12)
- Kaggle dataset (download manually or via kagglehub)

### Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv olist-qe
source olist-qe/bin/activate  # On Windows: olist-qe\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 3. Download data (choose one method):

# Option A: Via kagglehub (recommended - no credentials needed)
python -c "import kagglehub; kagglehub.dataset_download('olistbr/brazilian-ecommerce')"

# Option B: Manual download from Kaggle
# Download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Extract CSV files to data/raw/
```

### Run Analyses (Python Scripts)

```bash
# Run individual analyses
python scripts/run_eda.py              # Exploratory Data Analysis
python scripts/run_deadline_rd.py      # Deadline RD Analysis
python scripts/run_truckers_strike_did.py  # Truckers Strike DiD
python scripts/run_shipping_threshold_rd.py # Shipping Threshold RD
python scripts/run_installments_iv.py  # Installments IV Analysis

# Or run all at once
python scripts/run_all.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides an interactive interface to explore all four analyses with customizable parameters.

### Using Makefile (Alternative)

```bash
make setup      # Full setup: venv + deps + data
make jupyter    # Start JupyterLab for notebooks
make streamlit  # Start Streamlit dashboard
```

## Data

The project uses the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), which contains:

- **100k+ orders** from 2016-2018
- Order details, payments, reviews, products, sellers
- Customer and seller geolocation
- Delivery timestamps and estimates

Data is downloaded automatically via the Kaggle API.

## Methodology

### 1. Deadline Regression Discontinuity

**Question**: Does late delivery causally affect review scores?

- **Running Variable**: Days from promised delivery date
- **Cutoff**: 0 (delivered on deadline)
- **Design**: Sharp RD (deterministic treatment assignment)
- **Key Assumption**: No manipulation around cutoff (McCrary test)

### 2. Truckers Strike Difference-in-Differences

**Question**: What was the causal impact of the 2018 truckers strike?

- **Treatment**: Strike period (May 21 - June 1, 2018)
- **Outcomes**: Delivery time, cancellation rate
- **Control Group**: Pre-strike period
- **Key Assumption**: Parallel trends in pre-period

### 3. Shipping Threshold Fuzzy RD

**Question**: Does free shipping affect order composition?

- **Running Variable**: Order value (R$)
- **Cutoff**: Common thresholds (R$99, R$149, R$199)
- **Design**: Fuzzy RD (probabilistic treatment)
- **Key Assumption**: Discontinuity in freight, no bunching

### 4. Installments IV Analysis

**Question**: Do payment installments causally increase spending?

- **Endogenous Variable**: Installment usage
- **Instrument**: Credit card availability / payment method
- **Design**: IV/2SLS with sensitivity analysis
- **Key Assumption**: Exclusion restriction

## Visualizations

All visualizations are created with **Plotly** for interactivity and are saved in multiple formats:

- **HTML**: Interactive charts for web viewing
- **PNG**: High-resolution static images
- **JSON**: Plotly JSON for Streamlit dashboard

Charts are saved to `reports/figures/` for easy reuse.

## Available Commands

```bash
# Setup
make setup      # Full setup: venv + deps + data
make venv       # Create virtual environment only
make install    # Install dependencies only
make download   # Download Kaggle data only
make data       # Download and process data

# Development
make jupyter    # Start JupyterLab
make notebooks  # Execute all notebooks
make eda        # Execute EDA notebook only
make streamlit  # Start Streamlit dashboard

# Code Quality
make lint       # Run linters (ruff, mypy)
make format     # Format code (black, isort)
make test       # Run unit tests

# Cleanup
make clean      # Remove generated files
make clean-all  # Remove everything including venv

make help       # Show all available commands
```

## Python Scripts (Recommended)

| Script | Description | Output |
|--------|-------------|--------|
| `scripts/run_eda.py` | Exploratory Data Analysis | Summary stats, figures |
| `scripts/run_deadline_rd.py` | Deadline RD Analysis | RD estimates, sensitivity |
| `scripts/run_truckers_strike_did.py` | Strike DiD Analysis | DiD estimates, event study |
| `scripts/run_shipping_threshold_rd.py` | Shipping Threshold RD | Bunching analysis |
| `scripts/run_installments_iv.py` | Installments IV Analysis | IV/2SLS estimates |
| `scripts/run_all.py` | Run all analyses | Combined results JSON |

All scripts save results to `reports/` and figures to `reports/figures/`.

## Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Features:
- **Home**: Overview of all analyses and key findings
- **Deadline RD**: Interactive RD plot with adjustable bandwidth
- **Truckers Strike DiD**: Parallel trends and event study visualization
- **Shipping Threshold**: Bunching analysis with adjustable window
- **Installments IV**: First stage diagnostics and IV estimates

## Notebooks (Alternative)

| Notebook | Description | Status |
|----------|-------------|--------|
| `01_eda.ipynb` | Exploratory Data Analysis | Ready |

## Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_analysis.py -v
```

## Dependencies

### Core
- pandas, numpy, scipy
- plotly, kaleido
- statsmodels, linearmodels
- rdrobust, scikit-learn

### Development
- jupyter, jupyterlab
- pytest, black, ruff

See `requirements.txt` for full list with versions.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Make your changes
4. Run tests (`make test`)
5. Format code (`make format`)
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Olist](https://olist.com/) for making the dataset public
- [Kaggle](https://www.kaggle.com/) for hosting the data
- The causal inference community for methodological guidance

## References

- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly harmless econometrics*.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2019). *A Practical Introduction to Regression Discontinuity Designs*.
