"""
Ramsey-Cass-Koopmans Model Solver using Neural Networks

This module implements a Physics-Informed Neural Network (PINN) approach
to solve the Ramsey-Cass-Koopmans optimal growth model.

The model consists of:
1. Production function: Y = K^α
2. Capital accumulation: dK/dt = K^α - C - δK
3. Consumption Euler equation: dC/dt = (αK^(α-1) - δ - ρ) * C / θ

where:
- K: capital stock
- C: consumption
- α: capital share in production
- δ: depreciation rate
- ρ: discount rate (time preference)
- θ: inverse of elasticity of intertemporal substitution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class RamseyKlassNN(nn.Module):
    """
    Neural network to approximate the solution to the Ramsey-Cass-Koopmans model.

    The network takes time t as input and outputs capital K(t) and consumption C(t).
    """

    def __init__(self, hidden_layers: int = 4, hidden_units: int = 64):
        """
        Initialize the neural network.

        Args:
            hidden_layers: Number of hidden layers
            hidden_units: Number of units in each hidden layer
        """
        super(RamseyKlassNN, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(1, hidden_units))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())

        # Output layer (K and C)
        layers.append(nn.Linear(hidden_units, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            t: Time points

        Returns:
            K: Capital stock
            C: Consumption
        """
        output = self.network(t)
        K = torch.abs(output[:, 0:1])  # Ensure K > 0
        C = torch.abs(output[:, 1:2])  # Ensure C > 0
        return K, C


class RamseyKlassSolver:
    """
    Solver for the Ramsey-Cass-Koopmans model using neural networks.
    """

    def __init__(
        self,
        alpha: float = 0.3,      # Capital share
        delta: float = 0.05,     # Depreciation rate
        rho: float = 0.02,       # Discount rate
        theta: float = 2.0,      # Inverse of elasticity of intertemporal substitution
        K0: float = 5.0,         # Initial capital
        T: float = 100.0,        # Time horizon
        device: str = 'cpu'
    ):
        """
        Initialize the Ramsey-Cass-Koopmans solver.

        Args:
            alpha: Capital share in production
            delta: Depreciation rate
            rho: Discount rate (time preference)
            theta: Inverse of elasticity of intertemporal substitution
            K0: Initial capital stock
            T: Time horizon
            device: Device to run on ('cpu' or 'cuda')
        """
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        self.theta = theta
        self.K0 = K0
        self.T = T
        self.device = device

        # Compute steady state
        self.K_ss = ((self.alpha) / (self.rho + self.delta)) ** (1 / (1 - self.alpha))
        self.C_ss = self.K_ss ** self.alpha - self.delta * self.K_ss

        print(f"Steady State: K_ss = {self.K_ss:.4f}, C_ss = {self.C_ss:.4f}")

        self.model = RamseyKlassNN().to(device)
        self.optimizer = None
        self.loss_history = []

    def production(self, K: torch.Tensor) -> torch.Tensor:
        """Production function Y = K^α"""
        return K ** self.alpha

    def marginal_product(self, K: torch.Tensor) -> torch.Tensor:
        """Marginal product of capital: dY/dK = αK^(α-1)"""
        return self.alpha * K ** (self.alpha - 1)

    def compute_derivatives(
        self,
        t: torch.Tensor,
        K: torch.Tensor,
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time derivatives of K and C using automatic differentiation.

        Args:
            t: Time points
            K: Capital stock
            C: Consumption

        Returns:
            dK_dt: Time derivative of capital
            dC_dt: Time derivative of consumption
        """
        # Compute gradients
        dK_dt = torch.autograd.grad(
            K, t,
            grad_outputs=torch.ones_like(K),
            create_graph=True,
            retain_graph=True
        )[0]

        dC_dt = torch.autograd.grad(
            C, t,
            grad_outputs=torch.ones_like(C),
            create_graph=True,
            retain_graph=True
        )[0]

        return dK_dt, dC_dt

    def physics_loss(
        self,
        t: torch.Tensor,
        K: torch.Tensor,
        C: torch.Tensor,
        dK_dt: torch.Tensor,
        dC_dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the physics-informed loss based on the model equations.

        The equations are:
        1. dK/dt = K^α - C - δK
        2. dC/dt = (αK^(α-1) - δ - ρ) * C / θ

        Args:
            t: Time points
            K: Capital stock
            C: Consumption
            dK_dt: Time derivative of capital
            dC_dt: Time derivative of consumption

        Returns:
            Total physics loss
        """
        # Capital accumulation equation: dK/dt = K^α - C - δK
        Y = self.production(K)
        capital_residual = dK_dt - (Y - C - self.delta * K)

        # Consumption Euler equation: dC/dt = (αK^(α-1) - δ - ρ) * C / θ
        mpk = self.marginal_product(K)
        euler_residual = dC_dt - (mpk - self.delta - self.rho) * C / self.theta

        # MSE loss
        loss_capital = torch.mean(capital_residual ** 2)
        loss_euler = torch.mean(euler_residual ** 2)

        return loss_capital + loss_euler

    def boundary_loss(self, K_init: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for initial boundary condition.

        Args:
            K_init: Initial capital from network

        Returns:
            Boundary condition loss
        """
        return (K_init - self.K0) ** 2

    def terminal_loss(self, K_final: torch.Tensor, C_final: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for terminal condition (steady state).

        Args:
            K_final: Final capital from network
            C_final: Final consumption from network

        Returns:
            Terminal condition loss
        """
        loss_K_terminal = (K_final - self.K_ss) ** 2
        loss_C_terminal = (C_final - self.C_ss) ** 2
        return loss_K_terminal + loss_C_terminal

    def train(
        self,
        n_epochs: int = 10000,
        n_collocation: int = 200,
        learning_rate: float = 1e-3,
        print_every: int = 1000
    ):
        """
        Train the neural network to solve the Ramsey-Cass-Koopmans model.

        Args:
            n_epochs: Number of training epochs
            n_collocation: Number of collocation points
            learning_rate: Learning rate for optimizer
            print_every: Print loss every N epochs
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, verbose=False
        )

        print(f"Starting training for {n_epochs} epochs...")
        print(f"Using {n_collocation} collocation points")
        print("-" * 70)

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            # Sample collocation points
            t_collocation = torch.linspace(0, self.T, n_collocation).reshape(-1, 1)
            t_collocation = t_collocation.to(self.device)
            t_collocation.requires_grad = True

            # Initial and terminal time points
            t_init = torch.zeros(1, 1).to(self.device)
            t_init.requires_grad = True

            t_terminal = torch.ones(1, 1).to(self.device) * self.T
            t_terminal.requires_grad = True

            # Forward pass
            K_collocation, C_collocation = self.model(t_collocation)
            K_init, C_init = self.model(t_init)
            K_terminal, C_terminal = self.model(t_terminal)

            # Compute derivatives
            dK_dt, dC_dt = self.compute_derivatives(t_collocation, K_collocation, C_collocation)

            # Compute losses
            loss_physics = self.physics_loss(t_collocation, K_collocation, C_collocation, dK_dt, dC_dt)
            loss_boundary = self.boundary_loss(K_init)
            loss_terminal = self.terminal_loss(K_terminal, C_terminal)

            # Total loss with weights
            total_loss = loss_physics + 10.0 * loss_boundary + 5.0 * loss_terminal

            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            scheduler.step(total_loss)

            # Record loss
            self.loss_history.append(total_loss.item())

            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Total Loss: {total_loss.item():.6f} | "
                      f"Physics: {loss_physics.item():.6f} | "
                      f"Boundary: {loss_boundary.item():.6f} | "
                      f"Terminal: {loss_terminal.item():.6f}")

        print("-" * 70)
        print("Training completed!")

    def solve(self, t_points: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Solve the model and return the solution.

        Args:
            t_points: Time points to evaluate (if None, uses default)

        Returns:
            Dictionary containing time, capital, consumption, and output
        """
        if t_points is None:
            t_points = np.linspace(0, self.T, 1000)

        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.FloatTensor(t_points.reshape(-1, 1)).to(self.device)
            K, C = self.model(t_tensor)
            K = K.cpu().numpy().flatten()
            C = C.cpu().numpy().flatten()
            Y = K ** self.alpha

        return {
            'time': t_points,
            'capital': K,
            'consumption': C,
            'output': Y
        }

    def plot_solution(self, solution: Dict[str, np.ndarray] = None, save_path: str = None):
        """
        Plot the solution of the Ramsey-Cass-Koopmans model.

        Args:
            solution: Solution dictionary (if None, computes it)
            save_path: Path to save the figure (if None, displays it)
        """
        if solution is None:
            solution = self.solve()

        t = solution['time']
        K = solution['capital']
        C = solution['consumption']
        Y = solution['output']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot Capital
        axes[0, 0].plot(t, K, 'b-', linewidth=2, label='K(t)')
        axes[0, 0].axhline(y=self.K_ss, color='r', linestyle='--', label=f'K_ss = {self.K_ss:.2f}')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Capital Stock')
        axes[0, 0].set_title('Capital Dynamics')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot Consumption
        axes[0, 1].plot(t, C, 'g-', linewidth=2, label='C(t)')
        axes[0, 1].axhline(y=self.C_ss, color='r', linestyle='--', label=f'C_ss = {self.C_ss:.2f}')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Consumption')
        axes[0, 1].set_title('Consumption Dynamics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot Output
        axes[1, 0].plot(t, Y, 'm-', linewidth=2, label='Y(t)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Output')
        axes[1, 0].set_title('Output Dynamics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Phase diagram (K vs C)
        axes[1, 1].plot(K, C, 'k-', linewidth=2, label='Trajectory')
        axes[1, 1].plot(self.K_ss, self.C_ss, 'ro', markersize=10, label='Steady State')
        axes[1, 1].plot(K[0], C[0], 'go', markersize=8, label='Initial State')
        axes[1, 1].set_xlabel('Capital (K)')
        axes[1, 1].set_ylabel('Consumption (C)')
        axes[1, 1].set_title('Phase Diagram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Ramsey-Cass-Koopmans Model Solution (Neural Network)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

    def plot_training_loss(self, save_path: str = None):
        """
        Plot the training loss history.

        Args:
            save_path: Path to save the figure (if None, displays it)
        """
        if not self.loss_history:
            print("No training history available. Train the model first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()


def main():
    """
    Main function to demonstrate the Ramsey-Cass-Koopmans solver.
    """
    print("=" * 70)
    print("Ramsey-Cass-Koopmans Model Solver using Neural Networks")
    print("=" * 70)
    print()

    # Initialize solver with model parameters
    solver = RamseyKlassSolver(
        alpha=0.3,      # Capital share
        delta=0.05,     # Depreciation rate
        rho=0.02,       # Discount rate
        theta=2.0,      # Inverse of EIS
        K0=5.0,         # Initial capital (below steady state)
        T=100.0,        # Time horizon
        device='cpu'
    )

    # Train the neural network
    solver.train(
        n_epochs=15000,
        n_collocation=200,
        learning_rate=1e-3,
        print_every=1000
    )

    print()
    print("=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # Solve and plot
    solution = solver.solve()
    solver.plot_solution(solution, save_path='ramsey_klass_solution.png')
    solver.plot_training_loss(save_path='training_loss.png')

    # Print final values
    print()
    print("Solution Summary:")
    print(f"  Initial Capital: K(0) = {solution['capital'][0]:.4f}")
    print(f"  Final Capital: K(T) = {solution['capital'][-1]:.4f}")
    print(f"  Initial Consumption: C(0) = {solution['consumption'][0]:.4f}")
    print(f"  Final Consumption: C(T) = {solution['consumption'][-1]:.4f}")
    print(f"  Steady State Capital: K_ss = {solver.K_ss:.4f}")
    print(f"  Steady State Consumption: C_ss = {solver.C_ss:.4f}")
    print()


if __name__ == "__main__":
    main()
