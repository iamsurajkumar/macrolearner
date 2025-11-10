# =========================================================================
# Hopenhayn-Rogerson (1993) Model - Python Implementation
# Problem Set 1, Question 2
# =========================================================================

import numpy as np
from scipy.special import erf
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =========================================================================
# Helper Functions
# =========================================================================

def normcdf_local(x):
    """Standard normal CDF without scipy.stats"""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def fn_tauchen_discrete(n, rho, sigma_eps, mu, m):
    """
    Tauchen (1986) discretization for AR(1) in logs
    log z' = (1-rho)*mu + rho*log z + sigma_eps*eps, eps~N(0,1)
    """
    sigma_y = sigma_eps / np.sqrt(1 - rho**2)
    x = np.linspace(mu - m*sigma_y, mu + m*sigma_y, n)
    step = x[1] - x[0]
    P = np.zeros((n, n))

    for i in range(n):
        mu_i = (1 - rho)*mu + rho*x[i]

        # Transitioning to the lowest state
        P[i, 0] = normcdf_local((x[0] - mu_i + step/2) / sigma_eps)

        # Transitioning to the highest state
        P[i, n-1] = 1 - normcdf_local((x[n-1] - mu_i - step/2) / sigma_eps)

        # Interior states
        for j in range(1, n-1):
            zhi = (x[j] + step/2 - mu_i) / sigma_eps
            zlo = (x[j] - step/2 - mu_i) / sigma_eps
            P[i, j] = normcdf_local(zhi) - normcdf_local(zlo)

    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)

    return x, P


def fn_stationary_row_dist(P, tol=1e-14, maxiter=1000000):
    """Find stationary distribution of Markov chain"""
    n = P.shape[0]
    s = np.ones(n) / n
    it = 0

    while it < maxiter:
        it = it + 1
        s_new = np.dot(s, P)
        if np.max(np.abs(s_new - s)) < tol:
            return s_new
        s = s_new

    warnings.warn("stationary_row_dist: no convergence")
    return s


# =========================================================================
# Hopenhayn-Rogerson Model Class
# =========================================================================

class HopenhaynRogerson:
    """
    Hopenhayn-Rogerson (1993) Model of Industry Dynamics with Firing Costs
    """

    def __init__(self,
                 alpha=2/3,           # output elasticity of labor
                 beta=0.8,            # discount factor (5-year period)
                 c_f=20,              # per-period fixed operating cost
                 ce=40,               # sunk entry cost
                 tau=0.2,             # firing tax rate
                 theta=100,           # household utility parameter
                 zbar_log=1.4,        # mean of log productivity
                 rho=0.9,             # AR(1) persistence in logs
                 sigma_eps=0.2,       # std dev of innovations in log z
                 num_nz=33,           # number of grid points for z
                 num_nn=500,          # number of grid points for n (employment)
                 num_maxiter_vfi=1000,    # max iterations for VFI
                 num_tol_vfi=1e-8,        # tolerance for VFI
                 num_maxiter_p=100,       # max iterations for price search
                 num_tol_p=1e-6,          # tolerance for price
                 num_sd_gridbound=6,      # number of std devs for grid bounds
                 ):

        # Model parameters
        self.alpha = alpha
        self.beta = beta
        self.c_f = c_f
        self.ce = ce
        self.tau = tau
        self.theta = theta

        # AR(1) parameters for productivity
        self.zbar_log = zbar_log
        self.rho = rho
        self.sigma_eps = sigma_eps

        # Numerical parameters
        self.num_nz = num_nz
        self.num_nn = num_nn
        self.num_maxiter_vfi = num_maxiter_vfi
        self.num_tol_vfi = num_tol_vfi
        self.num_maxiter_p = num_maxiter_p
        self.num_tol_p = num_tol_p
        self.num_sd_gridbound = num_sd_gridbound

        # Discretize AR(1) productivity process
        self.logz, self.P_z = fn_tauchen_discrete(
            self.num_nz, self.rho, self.sigma_eps,
            self.zbar_log, self.num_sd_gridbound
        )
        self.z_grid = np.exp(self.logz)

        # Entrants' initial productivity distribution (stationary distribution)
        self.g_z = fn_stationary_row_dist(self.P_z)

        # Create employment grid (non-uniform, more points near zero)
        n_max = 1000  # Maximum employment
        self.n_grid = np.linspace(0, n_max, self.num_nn)

        # Initialize value function and policy functions
        self.V = None
        self.n_policy = None
        self.exit_policy = None

    def firing_cost(self, n_prime, n):
        """Firing cost function g(n', n) = tau * max{0, n - n'}"""
        return self.tau * np.maximum(0, n - n_prime)

    def static_profit(self, p, z, n_prime, n):
        """Static profits: p*z*n'^alpha - n' - p*c_f - g(n', n)"""
        output = p * z * (n_prime ** self.alpha)
        labor_cost = n_prime
        fixed_cost = p * self.c_f
        adj_cost = self.firing_cost(n_prime, n)
        return output - labor_cost - fixed_cost - adj_cost

    def exit_value(self, n):
        """Exit value: -g(0, n)"""
        return -self.firing_cost(0, n)


# =========================================================================
# Value Function Iteration
# =========================================================================

def fn_solve_incumbent_problem(p, hr, verbose=False):
    """
    Solve incumbent firm's value function via iteration
    Now with two state variables: (z, n)
    """
    nz = len(hr.z_grid)
    nn = len(hr.n_grid)

    # Initialize value function
    V = np.zeros((nz, nn))
    n_policy = np.zeros((nz, nn))
    exit_policy = np.zeros((nz, nn), dtype=bool)

    for iter in range(hr.num_maxiter_vfi):
        V_old = V.copy()

        # For each state (z, n)
        for iz in range(nz):
            z = hr.z_grid[iz]

            for in_idx in range(nn):
                n = hr.n_grid[in_idx]

                # Compute continuation value for all possible n'
                cont_values = np.zeros(nn)
                for in_prime in range(nn):
                    n_prime = hr.n_grid[in_prime]

                    # Expected value next period
                    EV = 0
                    for iz_prime in range(nz):
                        # Exit decision next period
                        exit_val_next = hr.exit_value(n_prime)
                        cont_val_next = V_old[iz_prime, in_prime]
                        EV += hr.P_z[iz, iz_prime] * max(exit_val_next, cont_val_next)

                    # Current period profit + discounted continuation
                    profit = hr.static_profit(p, z, n_prime, n)
                    cont_values[in_prime] = profit + hr.beta * EV

                # Find optimal n'
                best_in_prime = np.argmax(cont_values)
                best_value_continue = cont_values[best_in_prime]

                # Compare with exit
                exit_value = hr.exit_value(n)

                if exit_value > best_value_continue:
                    V[iz, in_idx] = exit_value
                    n_policy[iz, in_idx] = 0
                    exit_policy[iz, in_idx] = True
                else:
                    V[iz, in_idx] = best_value_continue
                    n_policy[iz, in_idx] = hr.n_grid[best_in_prime]
                    exit_policy[iz, in_idx] = False

        # Check convergence
        diff = np.max(np.abs(V - V_old))
        if diff < hr.num_tol_vfi:
            if verbose:
                print(f"VFI converged in {iter+1} iterations, max diff = {diff:.2e}")
            return V, n_policy, exit_policy

    warnings.warn(f"VFI did not converge after {hr.num_maxiter_vfi} iterations")
    return V, n_policy, exit_policy


# =========================================================================
# Equilibrium Price Finding
# =========================================================================

def fn_find_equilibrium_price(hr, pL=0.5, pH=2.0, verbose=True):
    """
    Find equilibrium price via bisection on the free-entry condition
    """

    for iter in range(hr.num_maxiter_p):
        p_test = (pL + pH) / 2

        # Solve incumbent problem
        V, n_policy, exit_policy = fn_solve_incumbent_problem(p_test, hr, verbose=False)

        # Compute entry value: entrants start with n=0
        # V_e = beta * E_z[V(z, 0)]
        Ve = hr.beta * np.sum(V[:, 0] * hr.g_z)

        if verbose:
            print(f'Price iteration {iter:3d}: p = {p_test:.6f}, Ve = {Ve:.6f}, '
                  f'ce = {hr.ce:.2f}, gap = {np.abs(Ve - hr.ce):.6f}')

        # Check free-entry condition
        if np.abs(Ve - hr.ce) < hr.num_tol_p:
            if verbose:
                print(f"\nConverged! Equilibrium Price = {p_test:.6f}")
            hr.V = V
            hr.n_policy = n_policy
            hr.exit_policy = exit_policy
            return p_test, V, n_policy, exit_policy
        elif Ve > hr.ce:
            pH = p_test  # Entry too attractive, lower price
        else:
            pL = p_test  # Entry not attractive enough, raise price

    warnings.warn("Price bisection did not converge")
    hr.V = V
    hr.n_policy = n_policy
    hr.exit_policy = exit_policy
    return p_test, V, n_policy, exit_policy


# =========================================================================
# Stationary Distribution
# =========================================================================

def fn_compute_stationary_distribution(hr, p_star, n_policy, exit_policy,
                                       m=1.0, maxiter=10000, tol=1e-8):
    """
    Compute stationary distribution of firms over (z, n)
    """
    nz = len(hr.z_grid)
    nn = len(hr.n_grid)

    # Initialize distribution
    mu = np.zeros((nz, nn))
    # Entrants start at n=0 with productivity drawn from g_z
    mu[:, 0] = m * hr.g_z

    for iter in range(maxiter):
        mu_new = np.zeros((nz, nn))

        # Entrants
        mu_new[:, 0] += m * hr.g_z

        # Incumbents
        for iz in range(nz):
            for in_idx in range(nn):
                if not exit_policy[iz, in_idx]:
                    # This firm continues
                    n_prime = n_policy[iz, in_idx]
                    # Find closest grid point for n'
                    in_prime = np.argmin(np.abs(hr.n_grid - n_prime))

                    # Add to next period distribution
                    for iz_prime in range(nz):
                        mu_new[iz_prime, in_prime] += hr.P_z[iz, iz_prime] * mu[iz, in_idx]

        # Check convergence
        diff = np.max(np.abs(mu_new - mu))
        if diff < tol:
            return mu_new

        mu = mu_new

    warnings.warn("Stationary distribution did not converge")
    return mu


# =========================================================================
# Market Clearing and Equilibrium
# =========================================================================

def fn_compute_equilibrium(hr, p_star, verbose=True):
    """
    Find mass of entrants that clears the labor market
    """
    # First compute distribution for m=1
    mu_normalized = fn_compute_stationary_distribution(
        hr, p_star, hr.n_policy, hr.exit_policy, m=1.0
    )

    # Compute total output and employment for m=1
    Y_normalized = 0
    N_demand_normalized = 0

    for iz in range(len(hr.z_grid)):
        for in_idx in range(len(hr.n_grid)):
            if mu_normalized[iz, in_idx] > 0:
                z = hr.z_grid[iz]
                n = hr.n_policy[iz, in_idx]
                Y_normalized += mu_normalized[iz, in_idx] * z * (n ** hr.alpha)
                N_demand_normalized += mu_normalized[iz, in_idx] * (n + hr.c_f)

    # Household supplies labor: N_supply = theta - Profits - Transfers
    # In equilibrium with balanced budget: Transfers = Tax Revenue
    # N_supply = theta - Profits
    # For now, iterate to find m* that clears labor market

    # Market clearing: p*C = N + Profits
    # C = theta/p (from household FOC)
    # Goods market clearing: Y = C
    C_demand = hr.theta / p_star

    # Scale to match goods demand
    m_star = C_demand / Y_normalized

    # Scale up distribution
    mu_star = m_star * mu_normalized

    if verbose:
        print(f"\nEquilibrium:")
        print(f"  Price p* = {p_star:.6f}")
        print(f"  Mass of entrants m* = {m_star:.6f}")
        print(f"  Aggregate output Y* = {m_star * Y_normalized:.6f}")
        print(f"  Aggregate labor demand N* = {m_star * N_demand_normalized:.6f}")

    return m_star, mu_star, mu_normalized


# =========================================================================
# Statistics Computation
# =========================================================================

def fn_compute_statistics(hr, p_star, m_star, mu_star):
    """
    Compute key statistics: average firm size, exit rate, job flows
    """
    nz = len(hr.z_grid)
    nn = len(hr.n_grid)

    # Total mass of firms
    total_firms = np.sum(mu_star)

    # Employment statistics
    total_employment = 0
    total_output = 0
    exiting_firms = 0

    # Job creation and destruction
    job_creation = 0
    job_destruction = 0

    for iz in range(nz):
        for in_idx in range(nn):
            if mu_star[iz, in_idx] > 0:
                z = hr.z_grid[iz]
                n_current = hr.n_grid[in_idx]
                n_next = hr.n_policy[iz, in_idx]

                # Employment and output
                total_employment += mu_star[iz, in_idx] * n_next
                total_output += mu_star[iz, in_idx] * z * (n_next ** hr.alpha)

                # Exit
                if hr.exit_policy[iz, in_idx]:
                    exiting_firms += mu_star[iz, in_idx]
                    if n_current > 0:
                        job_destruction += mu_star[iz, in_idx] * n_current
                else:
                    # Job flows
                    if n_next > n_current:
                        job_creation += mu_star[iz, in_idx] * (n_next - n_current)
                    elif n_next < n_current:
                        job_destruction += mu_star[iz, in_idx] * (n_current - n_next)

    # Average firm size (employment)
    avg_firm_size = total_employment / total_firms if total_firms > 0 else 0

    # Exit/entry rate
    exit_rate = exiting_firms / total_firms if total_firms > 0 else 0
    entry_rate = m_star / total_firms if total_firms > 0 else 0

    # Job creation/destruction rates
    jc_rate = job_creation / total_employment if total_employment > 0 else 0
    jd_rate = job_destruction / total_employment if total_employment > 0 else 0

    # Labor productivity
    labor_productivity = total_output / total_employment if total_employment > 0 else 0

    stats = {
        'avg_firm_size': avg_firm_size,
        'exit_rate': exit_rate,
        'entry_rate': entry_rate,
        'job_creation_rate': jc_rate,
        'job_destruction_rate': jd_rate,
        'labor_productivity': labor_productivity,
        'total_output': total_output,
        'total_employment': total_employment,
        'total_firms': total_firms,
        'price': p_star
    }

    return stats


# =========================================================================
# Main Execution
# =========================================================================

def solve_model(tau=0.2, verbose=True):
    """
    Solve the Hopenhayn-Rogerson model for a given firing tax tau
    """
    print(f"\n{'='*70}")
    print(f"Solving Hopenhayn-Rogerson Model with tau = {tau}")
    print(f"{'='*70}\n")

    # Create model instance
    hr = HopenhaynRogerson(tau=tau)

    # Find equilibrium price
    p_star, V, n_policy, exit_policy = fn_find_equilibrium_price(hr, verbose=verbose)

    # Compute equilibrium
    m_star, mu_star, mu_normalized = fn_compute_equilibrium(hr, p_star, verbose=verbose)

    # Compute statistics
    stats = fn_compute_statistics(hr, p_star, m_star, mu_star)

    print(f"\n{'='*70}")
    print(f"Results for tau = {tau}")
    print(f"{'='*70}")
    print(f"(i)   Average firm size: {stats['avg_firm_size']:.4f}")
    print(f"(ii)  Exit rate: {stats['exit_rate']:.4f}, Entry rate: {stats['entry_rate']:.4f}")
    print(f"(iii) Job destruction rate: {stats['job_destruction_rate']:.4f}")
    print(f"      Job creation rate: {stats['job_creation_rate']:.4f}")
    print(f"\nAdditional Statistics:")
    print(f"  Labor productivity (Y/N): {stats['labor_productivity']:.4f}")
    print(f"  Total output: {stats['total_output']:.4f}")
    print(f"  Total employment: {stats['total_employment']:.4f}")
    print(f"  Number of firms: {stats['total_firms']:.4f}")
    print(f"{'='*70}\n")

    return hr, stats


if __name__ == "__main__":

    # Question 2.2: Solve for tau = 0.2
    print("\n" + "="*70)
    print("QUESTION 2.2: Baseline Model (tau = 0.2)")
    print("="*70)
    hr_baseline, stats_baseline = solve_model(tau=0.2, verbose=True)

    # Question 2.3: Comparative statics
    print("\n" + "="*70)
    print("QUESTION 2.3: Comparative Statics")
    print("="*70)

    tau_values = [0.0, 0.2, 0.5]
    results = []

    for tau in tau_values:
        hr, stats = solve_model(tau=tau, verbose=False)
        results.append((tau, stats))

    # Display comparative statics
    print("\n" + "="*70)
    print("COMPARATIVE STATICS SUMMARY")
    print("="*70)
    print(f"{'tau':<10} {'Y/N':<15} {'Avg Size':<15} {'Exit Rate':<15} {'JD Rate':<15} {'JC Rate':<15}")
    print("-"*90)
    for tau, stats in results:
        print(f"{tau:<10.2f} {stats['labor_productivity']:<15.4f} {stats['avg_firm_size']:<15.4f} "
              f"{stats['exit_rate']:<15.4f} {stats['job_destruction_rate']:<15.4f} "
              f"{stats['job_creation_rate']:<15.4f}")

    print("\n" + "="*70)
    print("ANALYSIS:")
    print("="*70)
    print("\nWhy does misallocation arise?")
    print("-" * 70)
    print("1. Firing costs create an inaction region - firms don't adjust employment")
    print("   optimally in response to productivity shocks.")
    print("\n2. This means the marginal product of labor is NOT equalized across firms,")
    print("   leading to misallocation of labor.")
    print("\n3. Higher tau leads to:")
    print("   - Lower labor productivity (Y/N) due to misallocation")
    print("   - Lower exit/entry rates (less churning)")
    print("   - Lower job destruction AND creation rates (less labor reallocation)")
    print("\n4. The firing tax distorts the efficient allocation of labor across")
    print("   heterogeneous firms, reducing aggregate productivity.")
    print("="*70)

    print("\nDone! Analysis complete.")
