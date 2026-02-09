# Interview Prep: Difference-in-Differences & Event Study

## Quick Reference Card

| Concept | One-Liner |
|---------|-----------|
| **DiD** | Compare (Treated After - Before) vs (Control After - Before) to remove time trends |
| **Parallel Trends** | Without treatment, both groups would have followed the same trajectory |
| **Event Study** | Week-by-week DiD to see when effects kick in and validate parallel trends |
| **Interaction Term** | `Treated × Post` captures the causal effect |

---

## Why DiD? (Interview Question)

### Q: "Why did you use Difference-in-Differences instead of just comparing delivery times before vs after the strike?"

**Answer:**
> "Simply comparing before vs after would conflate the strike effect with other time trends. For example, Olist was improving their logistics throughout 2018, so delivery times were naturally decreasing. If I just compared before vs after, I'd underestimate the strike's impact because the general improvement would mask it.
>
> DiD solves this by using a control group (unaffected states) to estimate what the counterfactual trend would have been. The control group experienced the same time trends but wasn't affected by the strike. By subtracting the control group's change from the treated group's change, I isolate the causal effect of the strike."

### Q: "Why not just use a simple t-test comparing affected vs unaffected states?"

**Answer:**
> "A simple t-test would compare levels, not changes. But affected and unaffected states might have had different baseline delivery times for reasons unrelated to the strike (geography, infrastructure, seller density). DiD accounts for these pre-existing differences by looking at *changes* within each group, then comparing those changes."

---

## Why Event Study? (Interview Question)

### Q: "Why did you run an event study in addition to the standard DiD?"

**Answer:**
> "The event study serves two purposes:
>
> 1. **Validate parallel trends**: If the pre-treatment coefficients (weeks -8 to -1) are all close to zero and statistically insignificant, it supports the parallel trends assumption. If they're significantly different from zero, it raises concerns about the DiD validity.
>
> 2. **Understand dynamics**: Instead of a single 'average effect,' I can see *when* the effect kicks in and how it evolves. In my analysis, the strike effect peaked around weeks 5-8, suggesting a delayed impact as delivery backlogs accumulated."

### Q: "Your event study showed some significant pre-treatment effects. Does that invalidate your analysis?"

**Answer:**
> "It's a concern but not necessarily fatal. A few points:
>
> 1. The effects immediately before treatment (weeks -2, -1) were close to zero, suggesting trends converged near the treatment date.
>
> 2. The pre-treatment effects were negative (treated doing *better*), while post-treatment effects were positive (treated doing *worse*). This pattern is consistent with a real treatment effect rather than a spurious trend.
>
> 3. I'd acknowledge this limitation: the parallel trends assumption isn't perfectly satisfied, so the exact magnitude should be interpreted with some caution. However, the qualitative finding—that the strike caused delays—remains robust."

---

## Technical Deep-Dive Questions

### Q: "Walk me through the DiD regression equation."

**Answer:**
```
Y = β₀ + β₁(Treated) + β₂(Post) + β₃(Treated × Post) + ε
```

| Coefficient | Interpretation |
|-------------|----------------|
| β₀ | Baseline: Control group, pre-treatment |
| β₁ | Pre-existing difference between treated and control |
| β₂ | Time trend (change from pre to post for control) |
| **β₃** | **DiD effect** = causal impact of treatment |

> "The key is β₃, the interaction term. It captures how much *more* the treated group changed compared to the control group. In my analysis, β₃ = +2.3 days, meaning the strike caused 2.3 extra days of delivery delay."

### Q: "How do you calculate DiD from group means?"

**Answer:**
```
DiD = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)
    = (Change in Treated) - (Change in Control)
```

> "In my analysis:
> - Treated changed by -4.6 days (got faster)
> - Control changed by -6.9 days (got even faster)
> - DiD = (-4.6) - (-6.9) = +2.3 days
>
> The strike prevented treated states from improving as much as they should have."

### Q: "Why use robust standard errors?"

**Answer:**
> "Standard OLS assumes homoskedasticity—constant variance across observations. In practice, this rarely holds. Robust standard errors (HC1) are consistent even when variance differs across groups or time periods. This gives more reliable confidence intervals and p-values."

---

## Assumption Questions

### Q: "What are the key assumptions of DiD?"

**Answer:**

| Assumption | Description | How I Validated |
|------------|-------------|-----------------|
| **Parallel Trends** | Without treatment, both groups would follow same trend | Event study, visual inspection |
| **No Spillovers (SUTVA)** | Treatment of one unit doesn't affect others | Reasonable for geographic separation |
| **No Anticipation** | Treatment group didn't change behavior before treatment | Event study pre-trends ~0 |
| **Stable Composition** | Same units in pre and post periods | Used same time window for all |

### Q: "What if parallel trends is violated?"

**Answer:**
> "If parallel trends is violated, the DiD estimate is biased. Options include:
>
> 1. **Add covariates**: Control for time-varying confounders
> 2. **Use different control groups**: Find better comparison units
> 3. **Synthetic control method**: Construct a weighted combination of control units
> 4. **Acknowledge limitations**: Report the concern and interpret results cautiously
>
> In my analysis, I added covariates (price, freight) and got similar results, suggesting some robustness."

---

## Interpretation Questions

### Q: "Your data shows both groups got faster over time. How can the strike have caused delays?"

**Answer:**
> "Great observation! DiD measures *relative* change, not absolute change. Both groups improved, but the control group improved *more*. The strike held back the treated states from improving as much as they would have otherwise.
>
> Think of it as: 'Without the strike, treated states would have improved by 7 days (like control). They only improved by 4.6 days. The strike cost them 2.3 days of improvement.'"

### Q: "What's the difference between statistical and practical significance?"

**Answer:**
> "Statistical significance (p < 0.001) means the effect is unlikely due to random chance. Practical significance asks: 'Is 2.3 days meaningful for the business?'
>
> In e-commerce, 2.3 extra days of delivery delay is substantial—it can affect customer satisfaction, reviews, and repeat purchases. So yes, this is both statistically and practically significant."

---

## Business Application Questions

### Q: "What would you recommend to the business based on this analysis?"

**Answer:**
> 1. **Build buffer time**: During supply chain disruptions, add 2-3 days to delivery estimates
> 2. **Regional contingency plans**: Develop alternative logistics routes for affected regions
> 3. **Proactive communication**: Alert customers immediately when disruptions occur
> 4. **Inventory positioning**: Pre-position inventory in regions likely to be affected by disruptions

### Q: "How would you design a system to detect future disruptions early?"

**Answer:**
> "I'd set up monitoring for:
> 1. **Daily delivery time tracking** by region (detect anomalies)
> 2. **External data feeds**: News APIs, traffic data, fuel prices
> 3. **Automated DiD**: Compare regions in real-time to detect divergence
> 4. **Alert thresholds**: Trigger warnings when delivery times deviate significantly"

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong |
|---------|----------------|
| Comparing levels instead of changes | Ignores pre-existing differences |
| Ignoring pre-trends | May violate parallel trends |
| Using post-treatment covariates | Can introduce bias (bad controls) |
| Claiming causation without checking assumptions | DiD requires parallel trends |
| Forgetting about spillovers | Treatment may affect control group |

---

## Quick Formulas

```
# DiD from means
DiD = (Y_T1 - Y_T0) - (Y_C1 - Y_C0)

# DiD regression
Y = β₀ + β₁(Treat) + β₂(Post) + β₃(Treat×Post) + ε
    └─ DiD effect = β₃

# Event study (simplified)
Y = β₀ + Σₜ βₜ(Treat × 1{time=t}) + γ(Treat) + δₜ(Time FE) + ε
    └─ βₜ = effect at time t relative to reference period

# Standard error of DiD (simplified)
SE(DiD) ≈ √(SE₁² + SE₂² + SE₃² + SE₄²)
```

---

## Your Project Stats (Quick Reference)

| Metric | Value |
|--------|-------|
| DiD Estimate | +2.30 days |
| Standard Error | 0.21 |
| 95% CI | [1.89, 2.71] |
| p-value | < 1e-28 |
| N (total) | 52,614 |
| N (treated) | 44,894 |
| N (control) | 7,720 |
| Treatment | Orders in SP, MG, PR, SC, RS, RJ, GO, MT, MS |
| Event | May 21 - June 2, 2018 truckers strike |
