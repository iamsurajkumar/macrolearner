// =========================================================================
// Hopenhayn-Rogerson (1993) Model - C++ Implementation
// Problem Set 1, Question 2
// =========================================================================
//
// Compilation (with local Eigen):
//   g++ -std=c++17 -O3 -I./eigen-3.4.0 hopenhayn_rogerson_1993.cpp -o hopenhayn_rogerson
//
// Compilation (with system Eigen):
//   g++ -std=c++17 -O3 hopenhayn_rogerson_1993.cpp -o hopenhayn_rogerson
//
// Usage: ./hopenhayn_rogerson
//
// Dependencies: Eigen3 library for linear algebra (header-only)
// - Download: wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
//   Then: tar -xzf eigen-3.4.0.tar.gz
// - Or install system-wide: sudo apt-get install libeigen3-dev
// =========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// =========================================================================
// Helper Functions
// =========================================================================

/**
 * Standard normal CDF using error function
 */
double normcdf_local(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

/**
 * Tauchen (1986) discretization for AR(1) in logs
 * log z' = (1-rho)*mu + rho*log z + sigma_eps*eps, eps~N(0,1)
 *
 * @param n Number of grid points
 * @param rho AR(1) persistence parameter
 * @param sigma_eps Standard deviation of innovations
 * @param mu Mean of the process
 * @param m Number of standard deviations for grid bounds
 * @param x Output: grid points
 * @param P Output: transition probability matrix
 */
void fn_tauchen_discrete(int n, double rho, double sigma_eps, double mu, double m,
                         VectorXd& x, MatrixXd& P) {
    double sigma_y = sigma_eps / sqrt(1.0 - rho * rho);

    // Create grid
    x = VectorXd::LinSpaced(n, mu - m * sigma_y, mu + m * sigma_y);
    double step = x(1) - x(0);

    // Initialize transition matrix
    P = MatrixXd::Zero(n, n);

    for (int i = 0; i < n; i++) {
        double mu_i = (1.0 - rho) * mu + rho * x(i);

        // Transitioning to the lowest state
        P(i, 0) = normcdf_local((x(0) - mu_i + step / 2.0) / sigma_eps);

        // Transitioning to the highest state
        P(i, n-1) = 1.0 - normcdf_local((x(n-1) - mu_i - step / 2.0) / sigma_eps);

        // Interior states
        for (int j = 1; j < n - 1; j++) {
            double zhi = (x(j) + step / 2.0 - mu_i) / sigma_eps;
            double zlo = (x(j) - step / 2.0 - mu_i) / sigma_eps;
            P(i, j) = normcdf_local(zhi) - normcdf_local(zlo);
        }
    }

    // Normalize rows
    for (int i = 0; i < n; i++) {
        double row_sum = P.row(i).sum();
        P.row(i) /= row_sum;
    }
}

/**
 * Find stationary distribution of Markov chain
 */
VectorXd fn_stationary_row_dist(const MatrixXd& P, double tol = 1e-14, int maxiter = 1000000) {
    int n = P.rows();
    VectorXd s = VectorXd::Constant(n, 1.0 / n);

    for (int it = 0; it < maxiter; it++) {
        VectorXd s_new = s.transpose() * P;
        double diff = (s_new - s).cwiseAbs().maxCoeff();

        if (diff < tol) {
            return s_new;
        }
        s = s_new;
    }

    cerr << "Warning: stationary_row_dist did not converge" << endl;
    return s;
}

// =========================================================================
// Hopenhayn-Rogerson Model Class
// =========================================================================

class HopenhaynRogerson {
public:
    // Model parameters
    double alpha;           // output elasticity of labor
    double beta;            // discount factor
    double c_f;             // per-period fixed operating cost
    double ce;              // sunk entry cost
    double tau;             // firing tax rate
    double theta;           // household utility parameter

    // AR(1) parameters
    double zbar_log;        // mean of log productivity
    double rho;             // AR(1) persistence
    double sigma_eps;       // std dev of innovations

    // Numerical parameters
    int num_nz;             // number of z grid points
    int num_nn;             // number of n grid points
    int num_maxiter_vfi;    // max iterations for VFI
    double num_tol_vfi;     // tolerance for VFI
    int num_maxiter_p;      // max iterations for price search
    double num_tol_p;       // tolerance for price
    double num_sd_gridbound; // number of std devs for grid bounds

    // Grids and transition matrices
    VectorXd logz;          // log productivity grid
    VectorXd z_grid;        // productivity grid
    VectorXd n_grid;        // employment grid
    MatrixXd P_z;           // productivity transition matrix
    VectorXd g_z;           // entrants' productivity distribution

    // Value function and policies
    MatrixXd V;             // value function V(z, n)
    MatrixXd n_policy;      // employment policy n'(z, n)
    Matrix<bool, Dynamic, Dynamic> exit_policy;  // exit policy

    /**
     * Constructor with default parameters
     */
    HopenhaynRogerson(double alpha_ = 2.0/3.0,
                     double beta_ = 0.8,
                     double c_f_ = 20.0,
                     double ce_ = 40.0,
                     double tau_ = 0.2,
                     double theta_ = 100.0,
                     double zbar_log_ = 1.4,
                     double rho_ = 0.9,
                     double sigma_eps_ = 0.2,
                     int num_nz_ = 33,
                     int num_nn_ = 500,
                     int num_maxiter_vfi_ = 1000,
                     double num_tol_vfi_ = 1e-8,
                     int num_maxiter_p_ = 100,
                     double num_tol_p_ = 1e-6,
                     double num_sd_gridbound_ = 6.0)
        : alpha(alpha_), beta(beta_), c_f(c_f_), ce(ce_), tau(tau_), theta(theta_),
          zbar_log(zbar_log_), rho(rho_), sigma_eps(sigma_eps_),
          num_nz(num_nz_), num_nn(num_nn_), num_maxiter_vfi(num_maxiter_vfi_),
          num_tol_vfi(num_tol_vfi_), num_maxiter_p(num_maxiter_p_),
          num_tol_p(num_tol_p_), num_sd_gridbound(num_sd_gridbound_) {

        // Discretize AR(1) productivity process
        fn_tauchen_discrete(num_nz, rho, sigma_eps, zbar_log, num_sd_gridbound, logz, P_z);

        z_grid = logz.array().exp();

        // Entrants' initial productivity distribution (stationary)
        g_z = fn_stationary_row_dist(P_z);

        // Create employment grid
        double n_max = 1000.0;
        n_grid = VectorXd::LinSpaced(num_nn, 0.0, n_max);

        // Initialize value function and policies
        V = MatrixXd::Zero(num_nz, num_nn);
        n_policy = MatrixXd::Zero(num_nz, num_nn);
        exit_policy = Matrix<bool, Dynamic, Dynamic>::Zero(num_nz, num_nn);
    }

    /**
     * Firing cost function g(n', n) = tau * max{0, n - n'}
     */
    double firing_cost(double n_prime, double n) const {
        return tau * max(0.0, n - n_prime);
    }

    /**
     * Static profits: p*z*n'^alpha - n' - p*c_f - g(n', n)
     */
    double static_profit(double p, double z, double n_prime, double n) const {
        double output = p * z * pow(n_prime, alpha);
        double labor_cost = n_prime;
        double fixed_cost = p * c_f;
        double adj_cost = firing_cost(n_prime, n);
        return output - labor_cost - fixed_cost - adj_cost;
    }

    /**
     * Exit value: -g(0, n)
     */
    double exit_value(double n) const {
        return -firing_cost(0.0, n);
    }
};

// =========================================================================
// Value Function Iteration
// =========================================================================

/**
 * Solve incumbent firm's value function via iteration
 * State variables: (z, n)
 */
void fn_solve_incumbent_problem(double p, HopenhaynRogerson& hr, bool verbose = false) {
    int nz = hr.z_grid.size();
    int nn = hr.n_grid.size();

    // Initialize value function
    MatrixXd V = MatrixXd::Zero(nz, nn);
    MatrixXd n_policy = MatrixXd::Zero(nz, nn);
    Matrix<bool, Dynamic, Dynamic> exit_policy = Matrix<bool, Dynamic, Dynamic>::Zero(nz, nn);

    for (int iter = 0; iter < hr.num_maxiter_vfi; iter++) {
        MatrixXd V_old = V;

        // For each state (z, n)
        for (int iz = 0; iz < nz; iz++) {
            double z = hr.z_grid(iz);

            for (int in_idx = 0; in_idx < nn; in_idx++) {
                double n = hr.n_grid(in_idx);

                // Compute continuation value for all possible n'
                VectorXd cont_values(nn);

                for (int in_prime = 0; in_prime < nn; in_prime++) {
                    double n_prime = hr.n_grid(in_prime);

                    // Expected value next period
                    double EV = 0.0;
                    for (int iz_prime = 0; iz_prime < nz; iz_prime++) {
                        double exit_val_next = hr.exit_value(n_prime);
                        double cont_val_next = V_old(iz_prime, in_prime);
                        EV += hr.P_z(iz, iz_prime) * max(exit_val_next, cont_val_next);
                    }

                    // Current period profit + discounted continuation
                    double profit = hr.static_profit(p, z, n_prime, n);
                    cont_values(in_prime) = profit + hr.beta * EV;
                }

                // Find optimal n'
                int best_in_prime;
                double best_value_continue = cont_values.maxCoeff(&best_in_prime);

                // Compare with exit
                double exit_val = hr.exit_value(n);

                if (exit_val > best_value_continue) {
                    V(iz, in_idx) = exit_val;
                    n_policy(iz, in_idx) = 0.0;
                    exit_policy(iz, in_idx) = true;
                } else {
                    V(iz, in_idx) = best_value_continue;
                    n_policy(iz, in_idx) = hr.n_grid(best_in_prime);
                    exit_policy(iz, in_idx) = false;
                }
            }
        }

        // Check convergence
        double diff = (V - V_old).cwiseAbs().maxCoeff();
        if (diff < hr.num_tol_vfi) {
            if (verbose) {
                cout << "VFI converged in " << (iter + 1) << " iterations, max diff = "
                     << scientific << setprecision(2) << diff << endl;
            }
            hr.V = V;
            hr.n_policy = n_policy;
            hr.exit_policy = exit_policy;
            return;
        }
    }

    cerr << "Warning: VFI did not converge after " << hr.num_maxiter_vfi << " iterations" << endl;
    hr.V = V;
    hr.n_policy = n_policy;
    hr.exit_policy = exit_policy;
}

// =========================================================================
// Equilibrium Price Finding
// =========================================================================

/**
 * Find equilibrium price via bisection on the free-entry condition
 */
double fn_find_equilibrium_price(HopenhaynRogerson& hr, double pL = 0.5, double pH = 2.0,
                                 bool verbose = true) {
    for (int iter = 0; iter < hr.num_maxiter_p; iter++) {
        double p_test = (pL + pH) / 2.0;

        // Solve incumbent problem
        fn_solve_incumbent_problem(p_test, hr, false);

        // Compute entry value: entrants start with n=0
        // V_e = beta * E_z[V(z, 0)]
        double Ve = 0.0;
        for (int iz = 0; iz < hr.num_nz; iz++) {
            Ve += hr.g_z(iz) * hr.V(iz, 0);
        }
        Ve *= hr.beta;

        if (verbose) {
            cout << "Price iteration " << setw(3) << iter << ": p = " << fixed
                 << setprecision(6) << p_test << ", Ve = " << Ve
                 << ", ce = " << hr.ce << ", gap = " << abs(Ve - hr.ce) << endl;
        }

        // Check free-entry condition
        if (abs(Ve - hr.ce) < hr.num_tol_p) {
            if (verbose) {
                cout << "\nConverged! Equilibrium Price = " << p_test << endl;
            }
            return p_test;
        } else if (Ve > hr.ce) {
            pH = p_test;  // Entry too attractive, lower price
        } else {
            pL = p_test;  // Entry not attractive enough, raise price
        }
    }

    cerr << "Warning: Price bisection did not converge" << endl;
    return (pL + pH) / 2.0;
}

// =========================================================================
// Stationary Distribution
// =========================================================================

/**
 * Compute stationary distribution of firms over (z, n)
 */
MatrixXd fn_compute_stationary_distribution(const HopenhaynRogerson& hr, double m = 1.0,
                                            int maxiter = 10000, double tol = 1e-8) {
    int nz = hr.z_grid.size();
    int nn = hr.n_grid.size();

    // Initialize distribution
    MatrixXd mu = MatrixXd::Zero(nz, nn);

    // Entrants start at n=0 with productivity drawn from g_z
    for (int iz = 0; iz < nz; iz++) {
        mu(iz, 0) = m * hr.g_z(iz);
    }

    for (int iter = 0; iter < maxiter; iter++) {
        MatrixXd mu_new = MatrixXd::Zero(nz, nn);

        // Entrants
        for (int iz = 0; iz < nz; iz++) {
            mu_new(iz, 0) += m * hr.g_z(iz);
        }

        // Incumbents
        for (int iz = 0; iz < nz; iz++) {
            for (int in_idx = 0; in_idx < nn; in_idx++) {
                if (!hr.exit_policy(iz, in_idx)) {
                    // This firm continues
                    double n_prime = hr.n_policy(iz, in_idx);

                    // Find closest grid point for n'
                    int in_prime = 0;
                    double min_dist = abs(hr.n_grid(0) - n_prime);
                    for (int k = 1; k < nn; k++) {
                        double dist = abs(hr.n_grid(k) - n_prime);
                        if (dist < min_dist) {
                            min_dist = dist;
                            in_prime = k;
                        }
                    }

                    // Add to next period distribution
                    for (int iz_prime = 0; iz_prime < nz; iz_prime++) {
                        mu_new(iz_prime, in_prime) += hr.P_z(iz, iz_prime) * mu(iz, in_idx);
                    }
                }
            }
        }

        // Check convergence
        double diff = (mu_new - mu).cwiseAbs().maxCoeff();
        if (diff < tol) {
            return mu_new;
        }

        mu = mu_new;
    }

    cerr << "Warning: Stationary distribution did not converge" << endl;
    return mu;
}

// =========================================================================
// Market Clearing and Equilibrium
// =========================================================================

/**
 * Find mass of entrants that clears the labor market
 */
double fn_compute_equilibrium(const HopenhaynRogerson& hr, double p_star,
                              MatrixXd& mu_star, bool verbose = true) {
    // First compute distribution for m=1
    MatrixXd mu_normalized = fn_compute_stationary_distribution(hr, 1.0);

    // Compute total output and employment for m=1
    double Y_normalized = 0.0;
    double N_demand_normalized = 0.0;

    int nz = hr.z_grid.size();
    int nn = hr.n_grid.size();

    for (int iz = 0; iz < nz; iz++) {
        for (int in_idx = 0; in_idx < nn; in_idx++) {
            if (mu_normalized(iz, in_idx) > 0) {
                double z = hr.z_grid(iz);
                double n = hr.n_policy(iz, in_idx);
                Y_normalized += mu_normalized(iz, in_idx) * z * pow(n, hr.alpha);
                N_demand_normalized += mu_normalized(iz, in_idx) * (n + hr.c_f);
            }
        }
    }

    // Goods market clearing: Y = C, where C = theta/p
    double C_demand = hr.theta / p_star;

    // Scale to match goods demand
    double m_star = C_demand / Y_normalized;

    // Scale up distribution
    mu_star = m_star * mu_normalized;

    if (verbose) {
        cout << "\nEquilibrium:" << endl;
        cout << "  Price p* = " << fixed << setprecision(6) << p_star << endl;
        cout << "  Mass of entrants m* = " << m_star << endl;
        cout << "  Aggregate output Y* = " << (m_star * Y_normalized) << endl;
        cout << "  Aggregate labor demand N* = " << (m_star * N_demand_normalized) << endl;
    }

    return m_star;
}

// =========================================================================
// Statistics Computation
// =========================================================================

struct ModelStatistics {
    double avg_firm_size;
    double exit_rate;
    double entry_rate;
    double job_creation_rate;
    double job_destruction_rate;
    double labor_productivity;
    double total_output;
    double total_employment;
    double total_firms;
    double price;
};

/**
 * Compute key statistics: average firm size, exit rate, job flows
 */
ModelStatistics fn_compute_statistics(const HopenhaynRogerson& hr, double p_star,
                                      double m_star, const MatrixXd& mu_star) {
    int nz = hr.z_grid.size();
    int nn = hr.n_grid.size();

    ModelStatistics stats = {0};
    stats.price = p_star;

    // Total mass of firms
    stats.total_firms = mu_star.sum();

    // Job creation and destruction
    double job_creation = 0.0;
    double job_destruction = 0.0;
    double exiting_firms = 0.0;

    for (int iz = 0; iz < nz; iz++) {
        for (int in_idx = 0; in_idx < nn; in_idx++) {
            if (mu_star(iz, in_idx) > 0) {
                double z = hr.z_grid(iz);
                double n_current = hr.n_grid(in_idx);
                double n_next = hr.n_policy(iz, in_idx);

                // Employment and output
                stats.total_employment += mu_star(iz, in_idx) * n_next;
                stats.total_output += mu_star(iz, in_idx) * z * pow(n_next, hr.alpha);

                // Exit
                if (hr.exit_policy(iz, in_idx)) {
                    exiting_firms += mu_star(iz, in_idx);
                    if (n_current > 0) {
                        job_destruction += mu_star(iz, in_idx) * n_current;
                    }
                } else {
                    // Job flows
                    if (n_next > n_current) {
                        job_creation += mu_star(iz, in_idx) * (n_next - n_current);
                    } else if (n_next < n_current) {
                        job_destruction += mu_star(iz, in_idx) * (n_current - n_next);
                    }
                }
            }
        }
    }

    // Average firm size
    stats.avg_firm_size = (stats.total_firms > 0) ?
        stats.total_employment / stats.total_firms : 0.0;

    // Exit/entry rate
    stats.exit_rate = (stats.total_firms > 0) ? exiting_firms / stats.total_firms : 0.0;
    stats.entry_rate = (stats.total_firms > 0) ? m_star / stats.total_firms : 0.0;

    // Job creation/destruction rates
    stats.job_creation_rate = (stats.total_employment > 0) ?
        job_creation / stats.total_employment : 0.0;
    stats.job_destruction_rate = (stats.total_employment > 0) ?
        job_destruction / stats.total_employment : 0.0;

    // Labor productivity
    stats.labor_productivity = (stats.total_employment > 0) ?
        stats.total_output / stats.total_employment : 0.0;

    return stats;
}

// =========================================================================
// Main Execution
// =========================================================================

/**
 * Solve the Hopenhayn-Rogerson model for a given firing tax tau
 */
ModelStatistics solve_model(double tau = 0.2, bool verbose = true) {
    if (verbose) {
        cout << "\n" << string(70, '=') << endl;
        cout << "Solving Hopenhayn-Rogerson Model with tau = " << tau << endl;
        cout << string(70, '=') << "\n" << endl;
    }

    // Create model instance
    HopenhaynRogerson hr(2.0/3.0, 0.8, 20.0, 40.0, tau);

    // Find equilibrium price
    double p_star = fn_find_equilibrium_price(hr, 0.5, 2.0, verbose);

    // Compute equilibrium
    MatrixXd mu_star;
    double m_star = fn_compute_equilibrium(hr, p_star, mu_star, verbose);

    // Compute statistics
    ModelStatistics stats = fn_compute_statistics(hr, p_star, m_star, mu_star);

    if (verbose) {
        cout << "\n" << string(70, '=') << endl;
        cout << "Results for tau = " << tau << endl;
        cout << string(70, '=') << endl;
        cout << fixed << setprecision(4);
        cout << "(i)   Average firm size: " << stats.avg_firm_size << endl;
        cout << "(ii)  Exit rate: " << stats.exit_rate
             << ", Entry rate: " << stats.entry_rate << endl;
        cout << "(iii) Job destruction rate: " << stats.job_destruction_rate << endl;
        cout << "      Job creation rate: " << stats.job_creation_rate << endl;
        cout << "\nAdditional Statistics:" << endl;
        cout << "  Labor productivity (Y/N): " << stats.labor_productivity << endl;
        cout << "  Total output: " << stats.total_output << endl;
        cout << "  Total employment: " << stats.total_employment << endl;
        cout << "  Number of firms: " << stats.total_firms << endl;
        cout << string(70, '=') << "\n" << endl;
    }

    return stats;
}

int main() {
    // Question 2.2: Solve for tau = 0.2
    cout << "\n" << string(70, '=') << endl;
    cout << "QUESTION 2.2: Baseline Model (tau = 0.2)" << endl;
    cout << string(70, '=') << endl;
    ModelStatistics stats_baseline = solve_model(0.2, true);

    // Question 2.3: Comparative statics
    cout << "\n" << string(70, '=') << endl;
    cout << "QUESTION 2.3: Comparative Statics" << endl;
    cout << string(70, '=') << endl;

    vector<double> tau_values = {0.0, 0.2, 0.5};
    vector<ModelStatistics> results;

    for (double tau : tau_values) {
        ModelStatistics stats = solve_model(tau, false);
        results.push_back(stats);
    }

    // Display comparative statics
    cout << "\n" << string(70, '=') << endl;
    cout << "COMPARATIVE STATICS SUMMARY" << endl;
    cout << string(70, '=') << endl;
    cout << fixed << setprecision(4);
    cout << left << setw(10) << "tau"
         << setw(15) << "Y/N"
         << setw(15) << "Avg Size"
         << setw(15) << "Exit Rate"
         << setw(15) << "JD Rate"
         << setw(15) << "JC Rate" << endl;
    cout << string(90, '-') << endl;

    for (size_t i = 0; i < tau_values.size(); i++) {
        cout << left << setw(10) << tau_values[i]
             << setw(15) << results[i].labor_productivity
             << setw(15) << results[i].avg_firm_size
             << setw(15) << results[i].exit_rate
             << setw(15) << results[i].job_destruction_rate
             << setw(15) << results[i].job_creation_rate << endl;
    }

    cout << "\n" << string(70, '=') << endl;
    cout << "ANALYSIS:" << endl;
    cout << string(70, '=') << endl;
    cout << "\nWhy does misallocation arise?" << endl;
    cout << string(70, '-') << endl;
    cout << "1. Firing costs create an inaction region - firms don't adjust employment" << endl;
    cout << "   optimally in response to productivity shocks." << endl;
    cout << "\n2. This means the marginal product of labor is NOT equalized across firms," << endl;
    cout << "   leading to misallocation of labor." << endl;
    cout << "\n3. Higher tau leads to:" << endl;
    cout << "   - Lower labor productivity (Y/N) due to misallocation" << endl;
    cout << "   - Lower exit/entry rates (less churning)" << endl;
    cout << "   - Lower job destruction AND creation rates (less labor reallocation)" << endl;
    cout << "\n4. The firing tax distorts the efficient allocation of labor across" << endl;
    cout << "   heterogeneous firms, reducing aggregate productivity." << endl;
    cout << string(70, '=') << endl;

    cout << "\nDone! Analysis complete." << endl;

    return 0;
}
