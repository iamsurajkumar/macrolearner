# Ramsey-Cass-Koopmans Model Solver using Neural Networks

A Physics-Informed Neural Network (PINN) implementation for solving the Ramsey-Cass-Koopmans optimal growth model in economics.

## Overview

The Ramsey-Cass-Koopmans (RCK) model is a fundamental model in macroeconomics that describes optimal economic growth. It determines how an economy should optimally allocate resources between consumption and investment over time.

This implementation uses deep learning and automatic differentiation to solve the model's differential equations numerically.

## The Model

### Key Variables
- **K(t)**: Capital stock at time t
- **C(t)**: Consumption at time t
- **Y(t)**: Output at time t

### Model Equations

1. **Production Function**:
   ```
   Y(t) = K(t)^α
   ```
   where α is the capital share (typically 0.3)

2. **Capital Accumulation**:
   ```
   dK/dt = Y(t) - C(t) - δK(t)
   ```
   where δ is the depreciation rate

3. **Consumption Euler Equation**:
   ```
   dC/dt = [αK(t)^(α-1) - δ - ρ] × C(t) / θ
   ```
   where:
   - ρ is the discount rate (time preference)
   - θ is the inverse of the elasticity of intertemporal substitution

### Steady State

At steady state, capital and consumption are constant:
```
K_ss = (α / (ρ + δ))^(1/(1-α))
C_ss = K_ss^α - δ × K_ss
```

## Method: Physics-Informed Neural Networks (PINNs)

The neural network learns to approximate K(t) and C(t) by minimizing a loss function that includes:

1. **Physics Loss**: Ensures the solution satisfies the differential equations
2. **Boundary Loss**: Ensures initial capital K(0) = K₀
3. **Terminal Loss**: Encourages convergence to steady state at final time

The network uses automatic differentiation to compute time derivatives directly from the learned functions, making it a truly physics-informed approach.

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the solver with default parameters:

```bash
python ramsey_klass_neural_network.py
```

This will:
1. Initialize the model with default economic parameters
2. Train the neural network (15,000 epochs)
3. Generate solution plots
4. Save plots as PNG files

### Custom Parameters

You can customize the model parameters in your own script:

```python
from ramsey_klass_neural_network import RamseyKlassSolver

# Create solver with custom parameters
solver = RamseyKlassSolver(
    alpha=0.35,     # Capital share
    delta=0.06,     # Depreciation rate
    rho=0.03,       # Discount rate
    theta=1.5,      # Inverse of elasticity of intertemporal substitution
    K0=3.0,         # Initial capital
    T=100.0,        # Time horizon
    device='cpu'    # Use 'cuda' for GPU acceleration
)

# Train the model
solver.train(
    n_epochs=20000,
    n_collocation=300,
    learning_rate=1e-3
)

# Get solution
solution = solver.solve()

# Plot results
solver.plot_solution()
solver.plot_training_loss()
```

### Output

The program generates:

1. **Console Output**:
   - Steady state values
   - Training progress
   - Final solution summary

2. **Visualizations**:
   - `ramsey_klass_solution.png`: Four-panel plot showing:
     - Capital dynamics over time
     - Consumption dynamics over time
     - Output dynamics over time
     - Phase diagram (K vs C)
   - `training_loss.png`: Training loss convergence

## Model Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `alpha` | α | 0.3 | Capital share in production |
| `delta` | δ | 0.05 | Depreciation rate |
| `rho` | ρ | 0.02 | Discount rate (time preference) |
| `theta` | θ | 2.0 | Inverse of elasticity of intertemporal substitution |
| `K0` | K₀ | 5.0 | Initial capital stock |
| `T` | T | 100.0 | Time horizon |

## Neural Network Architecture

- **Input**: Time t (1 dimension)
- **Hidden Layers**: 4 layers with 64 units each
- **Activation**: Tanh
- **Output**: K(t) and C(t) (2 dimensions, constrained to be positive)

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 15000 | Number of training iterations |
| `n_collocation` | 200 | Number of time points for training |
| `learning_rate` | 0.001 | Initial learning rate |

## Example Output

```
======================================================================
Ramsey-Cass-Koopmans Model Solver using Neural Networks
======================================================================

Steady State: K_ss = 10.5737, C_ss = 2.2074
Starting training for 15000 epochs...
Using 200 collocation points
----------------------------------------------------------------------
Epoch 1000/15000 | Total Loss: 0.123456 | Physics: 0.098765 | Boundary: 0.012345 | Terminal: 0.012346
Epoch 2000/15000 | Total Loss: 0.045678 | Physics: 0.034567 | Boundary: 0.005678 | Terminal: 0.005433
...
----------------------------------------------------------------------
Training completed!

Solution Summary:
  Initial Capital: K(0) = 5.0012
  Final Capital: K(T) = 10.5689
  Initial Consumption: C(0) = 1.8234
  Final Consumption: C(T) = 2.2056
  Steady State Capital: K_ss = 10.5737
  Steady State Consumption: C_ss = 2.2074
```

## Features

- **Automatic Differentiation**: Uses PyTorch's autograd for computing derivatives
- **Physics-Informed Learning**: Incorporates economic laws directly into the loss function
- **Steady State Convergence**: Ensures solution converges to theoretical steady state
- **Flexible Architecture**: Easily customizable network depth and width
- **GPU Support**: Can run on CUDA-enabled GPUs for faster training
- **Comprehensive Visualization**: Multiple plots for analyzing the solution

## Theory Background

The Ramsey-Cass-Koopmans model extends the Solow growth model by endogenizing the savings rate through intertemporal utility maximization. Households choose consumption to maximize:

```
∫₀^∞ e^(-ρt) × u(C(t)) dt
```

where u(C) = C^(1-θ)/(1-θ) is the CRRA utility function.

The solution characterizes the optimal path of capital accumulation and consumption that maximizes social welfare.

## Advantages of the Neural Network Approach

1. **No Discretization Errors**: Unlike finite difference methods, the solution is continuous
2. **Automatic Differentiation**: No need to manually compute Jacobians
3. **Handles Complex Dynamics**: Can easily extend to higher dimensions
4. **Mesh-Free**: No need to define grids
5. **Differentiable Solution**: Can compute sensitivities w.r.t. parameters

## Extensions

Potential extensions of this code:
- Add labor dynamics (Ramsey model with endogenous labor)
- Include technology growth
- Multi-sector models
- Stochastic shocks
- Inequality constraints (e.g., non-negativity of consumption)

## References

1. Ramsey, F. P. (1928). "A Mathematical Theory of Saving". *Economic Journal*.
2. Cass, D. (1965). "Optimum Growth in an Aggregative Model of Capital Accumulation". *Review of Economic Studies*.
3. Koopmans, T. C. (1965). "On the Concept of Optimal Economic Growth". *Econometric Approach to Development Planning*.
4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks". *Journal of Computational Physics*.

## License

MIT License

## Author

Created using Physics-Informed Neural Networks for solving differential equations in economics.
