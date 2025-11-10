# Problem Set 1, Question 2: Hopenhayn-Rogerson (1993) Model

## Overview

This Python program implements the Hopenhayn-Rogerson (1993) model of industry dynamics with firing costs, as requested in PS1 Question 2.

## Model Description

The model extends Hopenhayn (1992) by adding:
- **Firing costs**: g(n', n) = τ × max{0, n - n'}
- **General equilibrium**: Household chooses consumption and labor supply
- **Two state variables**: Productivity (z) and lagged employment (n)
- **Dynamic employment decisions**: Firms face intertemporal tradeoffs due to firing costs

## Files

- `hopenhayn_rogerson_1993.py`: Main implementation file
- `hopenhayn_1992.py`: Reference implementation (simpler model)
- `lecture2and3.pdf`: Lecture notes with model details
- `PS1.pdf`: Problem set with questions

## Key Features

### Model Components

1. **Household**:
   - Utility: Σ β^t (θ ln C_t - N_t)
   - FOC: C = θ/p
   - Labor supply: N = θ - Π - T

2. **Firms**:
   - Production: y = z × n^α
   - Fixed cost: c_f
   - Firing cost: g(n', n) = τ × max{0, n - n'}
   - AR(1) productivity: log z' = (1-ρ)μ + ρ log z + σε

3. **Equilibrium**:
   - Free entry: V_e = c_e
   - Labor market clearing
   - Goods market clearing

### Computational Approach

The algorithm follows these steps:

**Step 1**: Solve for equilibrium price p*
- Use bisection to find p* satisfying free-entry condition
- For each price guess, solve firm's dynamic problem via VFI

**Step 2**: Compute stationary distribution μ(z, n)
- Iterate on law of motion until convergence
- Distribution has two dimensions (productivity and employment)

**Step 3**: Find mass of entrants m*
- Scale distribution to clear goods market
- Compute equilibrium statistics

### Implementation Details

- Productivity discretization: Tauchen (1986) method with 33 grid points
- Employment discretization: 500 grid points (linear spacing)
- Value function iteration with tolerance 1e-8
- Price bisection with tolerance 1e-6

## How to Run

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run the model
python hopenhayn_rogerson_1993.py
```

## Computational Notes

**Warning**: This model is computationally intensive due to:
- Two-dimensional state space (z, n)
- Value function iteration over ~16,500 states (33 × 500)
- Nested loops for computing continuation values
- Multiple equilibrium iterations

**Expected runtime**:
- With full grids (33 × 500): ~10-30 minutes per τ value
- With reduced grids (20 × 100): ~2-5 minutes per τ value

**Optimization suggestions**:
1. Reduce grid sizes for faster testing
2. Use parallel computing for VFI
3. Implement Howard's policy iteration
4. Use sparse matrices for transition probabilities

## Results Structure

The program solves the model for three values of τ:

### Question 2.2: Baseline (τ = 0.2)
Computes:
- (i) Average firm size
- (ii) Exit/entry rates
- (iii) Job destruction and creation rates

### Question 2.3: Comparative Statics (τ = 0, 0.2, 0.5)
Analyzes:
- Labor productivity Y/N across different firing costs
- How misallocation arises from firing taxes
- Trade-offs between employment protection and efficiency

## Key Insights

The model demonstrates that:

1. **Firing costs create misallocation**: The inaction region means firms don't adjust employment optimally, so MPL is not equalized across firms.

2. **Higher τ reduces productivity**: More friction → worse allocation → lower Y/N

3. **Labor market dynamics affected**: Higher τ → lower entry/exit rates, less job churning

4. **Trade-off**: Job security (less firing) vs. allocative efficiency

## Differences from Hopenhayn (1992)

The Hopenhayn-Rogerson model is **harder to solve** because:

1. **Two state variables** (z, n) instead of one (z)
   - Computational complexity increases dramatically
   - Need to discretize employment in addition to productivity

2. **Adjustment costs** create dynamic linkages
   - Current employment affects future costs
   - Firms must solve intertemporal optimization problem

3. **General equilibrium** with household
   - Must solve for equilibrium price AND entry mass
   - Labor supply is endogenous

4. **Policy functions are more complex**
   - Inaction regions emerge
   - Employment policy n'(z, n) depends on both states

## Extensions

Possible extensions to explore:
- Different adjustment cost specifications (convex costs)
- Hiring costs in addition to firing costs
- Aggregate shocks
- Firm-level capital accumulation
- Financial frictions

## References

- Hopenhayn, H. A. (1992). "Entry, Exit, and Firm Dynamics in Long Run Equilibrium." *Econometrica*, 60(5), 1127-1150.

- Hopenhayn, H., & Rogerson, R. (1993). "Job Turnover and Policy Evaluation: A General Equilibrium Analysis." *Journal of Political Economy*, 101(5), 915-938.

---

**Author**: Problem Set 1 Solution
**Course**: ECON8862 - Producer Heterogeneity in Macro
**Instructor**: Lukas Freund
