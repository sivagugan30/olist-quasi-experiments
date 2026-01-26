# Decision Memo: Olist Quasi-Experimental Analysis

**Date**: [Auto-generated after running analyses]  
**Author**: [Your Name]  
**Status**: Draft

---

## Executive Summary

This memo summarizes the causal findings from four quasi-experimental analyses on the Olist Brazilian E-Commerce dataset. These findings can inform business decisions around delivery operations, payment options, and pricing strategies.

---

## 1. Deadline Regression Discontinuity: Late Delivery Impact

### Research Question
Does late delivery (missing the promised delivery date) causally affect customer review scores?

### Methodology
- **Design**: Sharp Regression Discontinuity
- **Running Variable**: Days from promised delivery date
- **Cutoff**: 0 (delivery deadline)
- **Bandwidth**: [Optimal bandwidth calculated]
- **Sample Size**: [N observations]

### Key Findings

| Metric | Estimate | 95% CI | p-value |
|--------|----------|--------|---------|
| RD Effect | [TBD] | [TBD] | [TBD] |

**Interpretation**: [To be filled after analysis]

### Robustness Checks
- [ ] McCrary density test (manipulation)
- [ ] Bandwidth sensitivity
- [ ] Polynomial order sensitivity
- [ ] Covariate balance at cutoff

### Business Recommendation
[To be filled after analysis]

---

## 2. 2018 Truckers Strike: Difference-in-Differences

### Research Question
What was the causal impact of the May 2018 truckers strike on delivery times and cancellations?

### Methodology
- **Design**: Difference-in-Differences
- **Treatment Period**: May 21 - June 1, 2018
- **Pre-Period**: [Dates]
- **Post-Period**: [Dates]
- **Sample Size**: [N observations]

### Key Findings

| Outcome | DiD Estimate | 95% CI | p-value |
|---------|--------------|--------|---------|
| Delivery Time | [TBD] | [TBD] | [TBD] |
| Cancellation Rate | [TBD] | [TBD] | [TBD] |

**Interpretation**: [To be filled after analysis]

### Robustness Checks
- [ ] Parallel trends test
- [ ] Placebo tests (fake treatment dates)
- [ ] Event study (dynamic effects)
- [ ] Alternative control groups

### Business Recommendation
[To be filled after analysis]

---

## 3. Shipping Threshold: Fuzzy RD

### Research Question
Does free shipping eligibility (crossing price thresholds) causally affect order composition (AOV, items per order)?

### Methodology
- **Design**: Fuzzy Regression Discontinuity
- **Running Variable**: Order subtotal (R$)
- **Cutoffs**: R$99, R$149, R$199
- **First Stage**: Freight = 0 above threshold
- **Sample Size**: [N observations]

### Key Findings

| Threshold | First Stage | Reduced Form | Fuzzy RD | p-value |
|-----------|-------------|--------------|----------|---------|
| R$99 | [TBD] | [TBD] | [TBD] | [TBD] |
| R$149 | [TBD] | [TBD] | [TBD] | [TBD] |
| R$199 | [TBD] | [TBD] | [TBD] | [TBD] |

**Interpretation**: [To be filled after analysis]

### Robustness Checks
- [ ] Bunching analysis
- [ ] First stage strength (F-statistic)
- [ ] Bandwidth sensitivity
- [ ] Donut-hole RD

### Business Recommendation
[To be filled after analysis]

---

## 4. Installments: IV Analysis

### Research Question
Do payment installments causally increase spending (AOV)?

### Methodology
- **Design**: Instrumental Variables / 2SLS
- **Endogenous Variable**: Installment usage
- **Instrument**: [Credit card availability / payment type]
- **Sample Size**: [N observations]

### Key Findings

| Estimator | Estimate | 95% CI | First Stage F |
|-----------|----------|--------|---------------|
| OLS | [TBD] | [TBD] | - |
| 2SLS | [TBD] | [TBD] | [TBD] |

**Interpretation**: [To be filled after analysis]

### Robustness Checks
- [ ] First stage diagnostics (F > 10)
- [ ] Hausman test (OLS vs IV)
- [ ] Overidentification test (if multiple instruments)
- [ ] Sensitivity to exclusion restriction

### Business Recommendation
[To be filled after analysis]

---

## Summary of Recommendations

| Analysis | Finding | Confidence | Recommendation |
|----------|---------|------------|----------------|
| Late Delivery | [TBD] | [TBD] | [TBD] |
| Strike Impact | [TBD] | [TBD] | [TBD] |
| Free Shipping | [TBD] | [TBD] | [TBD] |
| Installments | [TBD] | [TBD] | [TBD] |

---

## Limitations

1. **External Validity**: Results from 2016-2018 Brazilian e-commerce may not generalize
2. **Data Limitations**: [List specific limitations]
3. **Identification Concerns**: [List any remaining concerns]

---

## Appendix

### A. Data Description
- Source: Olist Brazilian E-Commerce (Kaggle)
- Period: [Date range]
- Total Orders: [N]

### B. Code and Reproducibility
All analysis code is available in the `notebooks/` directory. 
Run `make notebooks` to reproduce all results.

### C. Figures
All figures are saved to `reports/figures/` in HTML, PNG, and JSON formats.
