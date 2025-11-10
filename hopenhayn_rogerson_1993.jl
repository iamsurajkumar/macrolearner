# =========================================================================
# Hopenhayn-Rogerson (1993) Model - Julia Implementation
# Problem Set 1, Question 2
# =========================================================================

using LinearAlgebra
using Distributions
using Printf

# =========================================================================
# Helper Functions
# =========================================================================

"""
Standard normal CDF
"""
function normcdf_local(x)
    return cdf(Normal(0, 1), x)
end


"""
Tauchen (1986) discretization for AR(1) in logs
log z' = (1-rho)*mu + rho*log z + sigma_eps*eps, eps~N(0,1)

Returns:
- x: grid points in log space
- P: transition matrix
"""
function fn_tauchen_discrete(n::Int, rho::Float64, sigma_eps::Float64,
                              mu::Float64, m::Float64)
    sigma_y = sigma_eps / sqrt(1 - rho^2)
    x = range(mu - m*sigma_y, mu + m*sigma_y, length=n)
    step = x[2] - x[1]
    P = zeros(n, n)

    for i in 1:n
        mu_i = (1 - rho) * mu + rho * x[i]

        # Transitioning to the lowest state
        P[i, 1] = normcdf_local((x[1] - mu_i + step/2) / sigma_eps)

        # Transitioning to the highest state
        P[i, n] = 1 - normcdf_local((x[n] - mu_i - step/2) / sigma_eps)

        # Interior states
        for j in 2:(n-1)
            zhi = (x[j] + step/2 - mu_i) / sigma_eps
            zlo = (x[j] - step/2 - mu_i) / sigma_eps
            P[i, j] = normcdf_local(zhi) - normcdf_local(zlo)
        end
    end

    # Normalize rows
    for i in 1:n
        P[i, :] ./= sum(P[i, :])
    end

    return collect(x), P
end


"""
Find stationary distribution of Markov chain
"""
function fn_stationary_row_dist(P::Matrix{Float64}; tol::Float64=1e-14,
                                  maxiter::Int=1000000)
    n = size(P, 1)
    s = ones(n) / n
    it = 0

    while it < maxiter
        it = it + 1
        s_new = s' * P
        s_new = vec(s_new)
        if maximum(abs.(s_new - s)) < tol
            return s_new
        end
        s = s_new
    end

    @warn "stationary_row_dist: no convergence"
    return s
end


# =========================================================================
# Hopenhayn-Rogerson Model Structure
# =========================================================================

"""
Hopenhayn-Rogerson (1993) Model of Industry Dynamics with Firing Costs
"""
mutable struct HopenhaynRogerson
    # Model parameters
    alpha::Float64           # output elasticity of labor
    beta::Float64            # discount factor (5-year period)
    c_f::Float64             # per-period fixed operating cost
    ce::Float64              # sunk entry cost
    tau::Float64             # firing tax rate
    theta::Float64           # household utility parameter

    # AR(1) parameters for productivity
    zbar_log::Float64        # mean of log productivity
    rho::Float64             # AR(1) persistence in logs
    sigma_eps::Float64       # std dev of innovations in log z

    # Numerical parameters
    num_nz::Int              # number of grid points for z
    num_nn::Int              # number of grid points for n (employment)
    num_maxiter_vfi::Int     # max iterations for VFI
    num_tol_vfi::Float64     # tolerance for VFI
    num_maxiter_p::Int       # max iterations for price search
    num_tol_p::Float64       # tolerance for price
    num_sd_gridbound::Float64  # number of std devs for grid bounds

    # Grids and distributions
    logz::Vector{Float64}    # log productivity grid
    P_z::Matrix{Float64}     # productivity transition matrix
    z_grid::Vector{Float64}  # productivity grid (levels)
    g_z::Vector{Float64}     # entrants' initial productivity distribution
    n_grid::Vector{Float64}  # employment grid

    # Value function and policy functions
    V::Union{Matrix{Float64}, Nothing}
    n_policy::Union{Matrix{Float64}, Nothing}
    exit_policy::Union{Matrix{Bool}, Nothing}

    function HopenhaynRogerson(;
                alpha::Float64=2/3,
                beta::Float64=0.8,
                c_f::Float64=20.0,
                ce::Float64=40.0,
                tau::Float64=0.2,
                theta::Float64=100.0,
                zbar_log::Float64=1.4,
                rho::Float64=0.9,
                sigma_eps::Float64=0.2,
                num_nz::Int=33,
                num_nn::Int=500,
                num_maxiter_vfi::Int=1000,
                num_tol_vfi::Float64=1e-8,
                num_maxiter_p::Int=100,
                num_tol_p::Float64=1e-6,
                num_sd_gridbound::Float64=6.0)

        # Discretize AR(1) productivity process
        logz, P_z = fn_tauchen_discrete(num_nz, rho, sigma_eps,
                                         zbar_log, num_sd_gridbound)
        z_grid = exp.(logz)

        # Entrants' initial productivity distribution (stationary distribution)
        g_z = fn_stationary_row_dist(P_z)

        # Create employment grid (non-uniform, more points near zero)
        n_max = 1000.0  # Maximum employment
        n_grid = range(0, n_max, length=num_nn) |> collect

        new(alpha, beta, c_f, ce, tau, theta,
            zbar_log, rho, sigma_eps,
            num_nz, num_nn, num_maxiter_vfi, num_tol_vfi,
            num_maxiter_p, num_tol_p, num_sd_gridbound,
            logz, P_z, z_grid, g_z, n_grid,
            nothing, nothing, nothing)
    end
end


"""
Firing cost function g(n', n) = tau * max{0, n - n'}
"""
function firing_cost(hr::HopenhaynRogerson, n_prime::Float64, n::Float64)
    return hr.tau * max(0, n - n_prime)
end


"""
Static profits: p*z*n'^alpha - n' - p*c_f - g(n', n)
"""
function static_profit(hr::HopenhaynRogerson, p::Float64, z::Float64,
                       n_prime::Float64, n::Float64)
    output = p * z * (n_prime ^ hr.alpha)
    labor_cost = n_prime
    fixed_cost = p * hr.c_f
    adj_cost = firing_cost(hr, n_prime, n)
    return output - labor_cost - fixed_cost - adj_cost
end


"""
Exit value: -g(0, n)
"""
function exit_value(hr::HopenhaynRogerson, n::Float64)
    return -firing_cost(hr, 0.0, n)
end


# =========================================================================
# Value Function Iteration
# =========================================================================

"""
Solve incumbent firm's value function via iteration
Now with two state variables: (z, n)
"""
function fn_solve_incumbent_problem(p::Float64, hr::HopenhaynRogerson;
                                      verbose::Bool=false)
    nz = length(hr.z_grid)
    nn = length(hr.n_grid)

    # Initialize value function
    V = zeros(nz, nn)
    n_policy = zeros(nz, nn)
    exit_policy = falses(nz, nn)

    for iter in 1:hr.num_maxiter_vfi
        V_old = copy(V)

        # For each state (z, n)
        for iz in 1:nz
            z = hr.z_grid[iz]

            for in_idx in 1:nn
                n = hr.n_grid[in_idx]

                # Compute continuation value for all possible n'
                cont_values = zeros(nn)
                for in_prime in 1:nn
                    n_prime = hr.n_grid[in_prime]

                    # Expected value next period
                    EV = 0.0
                    for iz_prime in 1:nz
                        # Exit decision next period
                        exit_val_next = exit_value(hr, n_prime)
                        cont_val_next = V_old[iz_prime, in_prime]
                        EV += hr.P_z[iz, iz_prime] * max(exit_val_next, cont_val_next)
                    end

                    # Current period profit + discounted continuation
                    profit = static_profit(hr, p, z, n_prime, n)
                    cont_values[in_prime] = profit + hr.beta * EV
                end

                # Find optimal n'
                best_in_prime = argmax(cont_values)
                best_value_continue = cont_values[best_in_prime]

                # Compare with exit
                exit_val = exit_value(hr, n)

                if exit_val > best_value_continue
                    V[iz, in_idx] = exit_val
                    n_policy[iz, in_idx] = 0.0
                    exit_policy[iz, in_idx] = true
                else
                    V[iz, in_idx] = best_value_continue
                    n_policy[iz, in_idx] = hr.n_grid[best_in_prime]
                    exit_policy[iz, in_idx] = false
                end
            end
        end

        # Check convergence
        diff = maximum(abs.(V - V_old))
        if diff < hr.num_tol_vfi
            if verbose
                println("VFI converged in $iter iterations, max diff = $(diff)")
            end
            return V, n_policy, exit_policy
        end
    end

    @warn "VFI did not converge after $(hr.num_maxiter_vfi) iterations"
    return V, n_policy, exit_policy
end


# =========================================================================
# Equilibrium Price Finding
# =========================================================================

"""
Find equilibrium price via bisection on the free-entry condition
"""
function fn_find_equilibrium_price(hr::HopenhaynRogerson;
                                    pL::Float64=0.5, pH::Float64=2.0,
                                    verbose::Bool=true)

    for iter in 1:hr.num_maxiter_p
        p_test = (pL + pH) / 2

        # Solve incumbent problem
        V, n_policy, exit_policy = fn_solve_incumbent_problem(p_test, hr,
                                                                verbose=false)

        # Compute entry value: entrants start with n=0
        # V_e = beta * E_z[V(z, 0)]
        Ve = hr.beta * sum(V[:, 1] .* hr.g_z)

        if verbose
            @printf("Price iteration %3d: p = %.6f, Ve = %.6f, ce = %.2f, gap = %.6f\n",
                    iter, p_test, Ve, hr.ce, abs(Ve - hr.ce))
        end

        # Check free-entry condition
        if abs(Ve - hr.ce) < hr.num_tol_p
            if verbose
                @printf("\nConverged! Equilibrium Price = %.6f\n", p_test)
            end
            hr.V = V
            hr.n_policy = n_policy
            hr.exit_policy = exit_policy
            return p_test, V, n_policy, exit_policy
        elseif Ve > hr.ce
            pH = p_test  # Entry too attractive, lower price
        else
            pL = p_test  # Entry not attractive enough, raise price
        end
    end

    @warn "Price bisection did not converge"
    hr.V = V
    hr.n_policy = n_policy
    hr.exit_policy = exit_policy
    return p_test, V, n_policy, exit_policy
end


# =========================================================================
# Stationary Distribution
# =========================================================================

"""
Compute stationary distribution of firms over (z, n)
"""
function fn_compute_stationary_distribution(hr::HopenhaynRogerson,
                                              p_star::Float64,
                                              n_policy::Matrix{Float64},
                                              exit_policy::Matrix{Bool};
                                              m::Float64=1.0,
                                              maxiter::Int=10000,
                                              tol::Float64=1e-8)
    nz = length(hr.z_grid)
    nn = length(hr.n_grid)

    # Initialize distribution
    mu = zeros(nz, nn)
    # Entrants start at n=0 with productivity drawn from g_z
    mu[:, 1] = m * hr.g_z

    for iter in 1:maxiter
        mu_new = zeros(nz, nn)

        # Entrants
        mu_new[:, 1] .+= m * hr.g_z

        # Incumbents
        for iz in 1:nz
            for in_idx in 1:nn
                if !exit_policy[iz, in_idx]
                    # This firm continues
                    n_prime = n_policy[iz, in_idx]
                    # Find closest grid point for n'
                    in_prime = argmin(abs.(hr.n_grid .- n_prime))

                    # Add to next period distribution
                    for iz_prime in 1:nz
                        mu_new[iz_prime, in_prime] += hr.P_z[iz, iz_prime] * mu[iz, in_idx]
                    end
                end
            end
        end

        # Check convergence
        diff = maximum(abs.(mu_new - mu))
        if diff < tol
            return mu_new
        end

        mu = mu_new
    end

    @warn "Stationary distribution did not converge"
    return mu
end


# =========================================================================
# Market Clearing and Equilibrium
# =========================================================================

"""
Find mass of entrants that clears the labor market
"""
function fn_compute_equilibrium(hr::HopenhaynRogerson, p_star::Float64;
                                  verbose::Bool=true)
    # First compute distribution for m=1
    mu_normalized = fn_compute_stationary_distribution(
        hr, p_star, hr.n_policy, hr.exit_policy, m=1.0
    )

    # Compute total output and employment for m=1
    Y_normalized = 0.0
    N_demand_normalized = 0.0

    for iz in 1:length(hr.z_grid)
        for in_idx in 1:length(hr.n_grid)
            if mu_normalized[iz, in_idx] > 0
                z = hr.z_grid[iz]
                n = hr.n_policy[iz, in_idx]
                Y_normalized += mu_normalized[iz, in_idx] * z * (n ^ hr.alpha)
                N_demand_normalized += mu_normalized[iz, in_idx] * (n + hr.c_f)
            end
        end
    end

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

    if verbose
        println("\nEquilibrium:")
        @printf("  Price p* = %.6f\n", p_star)
        @printf("  Mass of entrants m* = %.6f\n", m_star)
        @printf("  Aggregate output Y* = %.6f\n", m_star * Y_normalized)
        @printf("  Aggregate labor demand N* = %.6f\n", m_star * N_demand_normalized)
    end

    return m_star, mu_star, mu_normalized
end


# =========================================================================
# Statistics Computation
# =========================================================================

"""
Compute key statistics: average firm size, exit rate, job flows
"""
function fn_compute_statistics(hr::HopenhaynRogerson, p_star::Float64,
                                 m_star::Float64, mu_star::Matrix{Float64})
    nz = length(hr.z_grid)
    nn = length(hr.n_grid)

    # Total mass of firms
    total_firms = sum(mu_star)

    # Employment statistics
    total_employment = 0.0
    total_output = 0.0
    exiting_firms = 0.0

    # Job creation and destruction
    job_creation = 0.0
    job_destruction = 0.0

    for iz in 1:nz
        for in_idx in 1:nn
            if mu_star[iz, in_idx] > 0
                z = hr.z_grid[iz]
                n_current = hr.n_grid[in_idx]
                n_next = hr.n_policy[iz, in_idx]

                # Employment and output
                total_employment += mu_star[iz, in_idx] * n_next
                total_output += mu_star[iz, in_idx] * z * (n_next ^ hr.alpha)

                # Exit
                if hr.exit_policy[iz, in_idx]
                    exiting_firms += mu_star[iz, in_idx]
                    if n_current > 0
                        job_destruction += mu_star[iz, in_idx] * n_current
                    end
                else
                    # Job flows
                    if n_next > n_current
                        job_creation += mu_star[iz, in_idx] * (n_next - n_current)
                    elseif n_next < n_current
                        job_destruction += mu_star[iz, in_idx] * (n_current - n_next)
                    end
                end
            end
        end
    end

    # Average firm size (employment)
    avg_firm_size = total_firms > 0 ? total_employment / total_firms : 0.0

    # Exit/entry rate
    exit_rate = total_firms > 0 ? exiting_firms / total_firms : 0.0
    entry_rate = total_firms > 0 ? m_star / total_firms : 0.0

    # Job creation/destruction rates
    jc_rate = total_employment > 0 ? job_creation / total_employment : 0.0
    jd_rate = total_employment > 0 ? job_destruction / total_employment : 0.0

    # Labor productivity
    labor_productivity = total_employment > 0 ? total_output / total_employment : 0.0

    stats = Dict(
        :avg_firm_size => avg_firm_size,
        :exit_rate => exit_rate,
        :entry_rate => entry_rate,
        :job_creation_rate => jc_rate,
        :job_destruction_rate => jd_rate,
        :labor_productivity => labor_productivity,
        :total_output => total_output,
        :total_employment => total_employment,
        :total_firms => total_firms,
        :price => p_star
    )

    return stats
end


# =========================================================================
# Main Execution
# =========================================================================

"""
Solve the Hopenhayn-Rogerson model for a given firing tax tau
"""
function solve_model(tau::Float64=0.2; verbose::Bool=true)
    println("\n" * "="^70)
    println("Solving Hopenhayn-Rogerson Model with tau = $tau")
    println("="^70 * "\n")

    # Create model instance
    hr = HopenhaynRogerson(tau=tau)

    # Find equilibrium price
    p_star, V, n_policy, exit_policy = fn_find_equilibrium_price(hr, verbose=verbose)

    # Compute equilibrium
    m_star, mu_star, mu_normalized = fn_compute_equilibrium(hr, p_star, verbose=verbose)

    # Compute statistics
    stats = fn_compute_statistics(hr, p_star, m_star, mu_star)

    println("\n" * "="^70)
    println("Results for tau = $tau")
    println("="^70)
    @printf("(i)   Average firm size: %.4f\n", stats[:avg_firm_size])
    @printf("(ii)  Exit rate: %.4f, Entry rate: %.4f\n",
            stats[:exit_rate], stats[:entry_rate])
    @printf("(iii) Job destruction rate: %.4f\n", stats[:job_destruction_rate])
    @printf("      Job creation rate: %.4f\n", stats[:job_creation_rate])
    println("\nAdditional Statistics:")
    @printf("  Labor productivity (Y/N): %.4f\n", stats[:labor_productivity])
    @printf("  Total output: %.4f\n", stats[:total_output])
    @printf("  Total employment: %.4f\n", stats[:total_employment])
    @printf("  Number of firms: %.4f\n", stats[:total_firms])
    println("="^70 * "\n")

    return hr, stats
end


# =========================================================================
# Main script execution
# =========================================================================

function main()
    # Question 2.2: Solve for tau = 0.2
    println("\n" * "="^70)
    println("QUESTION 2.2: Baseline Model (tau = 0.2)")
    println("="^70)
    hr_baseline, stats_baseline = solve_model(0.2, verbose=true)

    # Question 2.3: Comparative statics
    println("\n" * "="^70)
    println("QUESTION 2.3: Comparative Statics")
    println("="^70)

    tau_values = [0.0, 0.2, 0.5]
    results = []

    for tau in tau_values
        hr, stats = solve_model(tau, verbose=false)
        push!(results, (tau, stats))
    end

    # Display comparative statics
    println("\n" * "="^70)
    println("COMPARATIVE STATICS SUMMARY")
    println("="^70)
    @printf("%-10s %-15s %-15s %-15s %-15s %-15s\n",
            "tau", "Y/N", "Avg Size", "Exit Rate", "JD Rate", "JC Rate")
    println("-"^90)
    for (tau, stats) in results
        @printf("%-10.2f %-15.4f %-15.4f %-15.4f %-15.4f %-15.4f\n",
                tau, stats[:labor_productivity], stats[:avg_firm_size],
                stats[:exit_rate], stats[:job_destruction_rate],
                stats[:job_creation_rate])
    end

    println("\n" * "="^70)
    println("ANALYSIS:")
    println("="^70)
    println("\nWhy does misallocation arise?")
    println("-" ^ 70)
    println("1. Firing costs create an inaction region - firms don't adjust employment")
    println("   optimally in response to productivity shocks.")
    println("\n2. This means the marginal product of labor is NOT equalized across firms,")
    println("   leading to misallocation of labor.")
    println("\n3. Higher tau leads to:")
    println("   - Lower labor productivity (Y/N) due to misallocation")
    println("   - Lower exit/entry rates (less churning)")
    println("   - Lower job destruction AND creation rates (less labor reallocation)")
    println("\n4. The firing tax distorts the efficient allocation of labor across")
    println("   heterogeneous firms, reducing aggregate productivity.")
    println("="^70)

    println("\nDone! Analysis complete.")
end

# Run main if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
