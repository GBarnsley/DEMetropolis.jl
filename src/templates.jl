"""
    deMC(model_wrapper, n_its; kwargs...)

Run the Differential Evolution Markov Chain (DE-MC) sampler proposed by ter Braak (2006).

This sampler uses differential evolution updates with optional switching between two
scaling factors (`γ₁` and `γ₂`) to enable mode switching. The algorithm runs for a
fixed number of iterations with optional burn-in.

This implementation varies slightly from the original: updates within a population
occur based on the previous positions to enable easy parallelization.

See doi.org/10.1007/s11222-006-8769-1 for more information.

# Arguments
- `model_wrapper`: LogDensityModel containing the target log-density function
- `n_its`: Number of sampling iterations per chain

# Keyword Arguments
- `n_burnin`: Number of burn-in iterations. Defaults to `n_its * 5`.
- `n_chains`: Number of chains. Defaults to `max(dimension(ld) * 2, 3)`.
- `initial_state`: Initial states for the chains. Defaults to random initialization.
- `N₀`: Size of initial population for memory-based sampling. Defaults to `n_chains`.
- `memory`: Use memory-based sampling (`true`) or memoryless (`false`). Defaults to `false`.
- `memory_size`: Maximum number of positions retained per chain in memory during initialization. Defaults to `n_its + n_burnin`.
- `memory_refill`: When memory is full, overwrite from the start (cyclic) if `true`. Defaults to `true` (forwarded via keyword arguments).
- `memory_thin_interval`: If > 0, only every `memory_thin_interval`-th accepted state is stored in memory. Defaults to `0` (forwarded via keyword arguments).
- `save_burnt`: Save burn-in samples in output. Defaults to `false`.
- `parallel`: Run chains in parallel using threading. Defaults to `false`.
- `silent`: Suppress informational initialization logs when `true`. Defaults to `false` (forwarded via keyword arguments).
- `rng`: Random number generator. Defaults to `default_rng()`.
- `thin`: Thinning interval for saved samples. Defaults to 1.
- `γ₁`: Primary scaling factor. Defaults to `2.38 / sqrt(2 * dim)`.
- `γ₂`: Secondary scaling factor for mode switching. Defaults to 1.0.
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.1.
- `β`: Noise distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `n_hot_chains`: Number of hot chains for parallel tempering. Defaults to 0 (no parallel tempering).
- `max_temp_pt`: Maximum temperature for parallel tempering. Defaults to 2*sqrt(dimension).
- `max_temp_sa`: Maximum temperature for simulated annealing. Defaults to `max_temp_pt`.
- `α`: Temperature ladder spacing parameter. Defaults to 1.0.
- `annealing`: Whether to use simulated annealing. Defaults to `false`.
- `annealing_steps`: Number of annealing steps. Defaults to 0 or the number of warmup-steps (when using AbstractMCMC.sample).
- `temperature_ladder`: Pre-defined temperature ladder. Defaults to automatic creation based on other parameters.
- `chain_type`: Type of chain to return (e.g., `Any`, `DifferentialEvolutionOutput`, `MCMCChains.Chains`). Defaults to `DifferentialEvolutionOutput`.
- `save_final_state`: Whether to return the final state along with samples, if true the output will be (samples::chain_type, final_state). Defaults to `false`.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` and the internal initialization step (e.g., `memory_refill`, `memory_thin_interval`, `silent`). See [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments).

# Returns
- Named tuple containing samples and optionally burn-in samples

# Example
```@example deMC
using DEMetropolis, Random, Distributions

# Define a simple log-density function
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Run differential evolution MCMC
result = deMC(model_wrapper, 1000; n_chains = 10, parallel = false)
```

See also [`deMCzs`](@ref), [`DREAMz`](@ref), [`setup_de_update`](@ref).
"""
function deMC(
        model_wrapper::LogDensityModel, n_its::Int;
        n_burnin::Int = n_its * 5,
        initial_state::Union{AbstractVector{<:AbstractVector{T}}, Nothing} = nothing,
        save_burnt::Bool = false,
        rng::AbstractRNG = default_rng(),
        thin::Int = 1,
        memory::Bool = false,
        memory_size::Int = n_its + n_burnin,
        γ₁::Union{Nothing, T} = nothing,
        γ₂::T = 1.0,
        p_γ₂::T = 0.1,
        β::ContinuousUnivariateDistribution = Uniform(-1.0e-4, 1.0e-4),
        chain_type = DifferentialEvolutionOutput,
        kwargs...
    ) where {T <: Real}

    #build sampler scheme
    if γ₁ != γ₂
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ₂, β = β, n_dims = dimension(model_wrapper.logdensity)),
            setup_de_update(γ = γ₁, β = β, n_dims = dimension(model_wrapper.logdensity));
            w = [p_γ₂, 1 - p_γ₂]
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ₁, β = β, n_dims = dimension(model_wrapper.logdensity))
        )
    end

    if save_burnt
        its = n_its + n_burnin
    else
        its = n_its
    end

    return sample(
        rng,
        model_wrapper,
        sampler_scheme,
        its;
        num_warmup = n_burnin,
        initial_position = initial_state,
        thinning = thin,
        discard_initial = save_burnt ? 0 : n_burnin,
        memory = memory,
        memory_size = memory_size,
        chain_type = chain_type,
        kwargs...
    )
end

"""
    deMCzs(model_wrapper, check_every; kwargs...)

Run the Differential Evolution Markov Chain with snooker update and historic sampling (DE-MCzs) sampler.

This adaptive sampler runs until convergence (measured by R̂ < `maximum_R̂`) or a maximum
number of epochs. It combines DE updates with optional snooker moves and uses memory-based
sampling to efficiently handle high-dimensional problems with fewer chains.

Proposed by ter Braak and Vrugt (2008), see doi.org/10.1007/s11222-008-9104-9.

# Arguments
- `model_wrapper`: LogDensityModel containing the target log-density function
- `check_every`: Number of iterations per chain per convergence check

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs before convergence checking. Defaults to 5.
- `epoch_limit`: Maximum number of total epochs. Defaults to 20.
- `maximum_R̂`: Convergence threshold for Gelman-Rubin diagnostic. Defaults to 1.2.
- `n_chains`: Number of chains. Defaults to `max(dimension(ld) * 2, 3)`.
- `N₀`: Size of initial population for memory-based sampling. Defaults to `n_chains * 2`.
- `initial_state`: Initial population state. Defaults to random initialization.
- `memory`: Use memory-based sampling. Defaults to `true`.
- `save_burnt`: Save warm-up samples in output. Defaults to `true`.
- `parallel`: Run chains in parallel using threading. Defaults to `false`.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `memory_size`: Maximum number of positions retained per chain in memory during initialization. Defaults to `check_every * 10`.
- `memory_refill`: When memory is full, overwrite from the start (cyclic) if `true`. Defaults to `true`.
- `memory_thin_interval`: If > 0, only every `memory_thin_interval`-th accepted state is stored in memory. Defaults to `0` (forwarded via keyword arguments).
- `silent`: Suppress informational initialization logs when `true`. Defaults to `false` (forwarded via keyword arguments).
- `γ`: Scaling factor for DE updates. Defaults to `2.38 / sqrt(2 * dim)`.
- `γₛ`: Scaling factor for snooker updates. Defaults to `2.38 / sqrt(2)`.
- `p_snooker`: Probability of snooker moves. Defaults to 0.1.
- `β`: Noise distribution for DE updates. Defaults to `Uniform(-1e-4, 1e-4)`.
- `thin`: Thinning interval for saved samples. Defaults to 1.
- `n_hot_chains`: Number of hot chains for parallel tempering. Defaults to 0 (no parallel tempering).
- `max_temp_pt`: Maximum temperature for parallel tempering. Defaults to 2*sqrt(dimension).
- `max_temp_sa`: Maximum temperature for simulated annealing. Defaults to `max_temp_pt`.
- `α`: Temperature ladder spacing parameter. Defaults to 1.0.
- `annealing`: Whether to use simulated annealing. Defaults to `false`.
- `annealing_steps`: Number of annealing steps. Defaults to 0 or the number of warmup-steps (when using AbstractMCMC.sample).
- `temperature_ladder`: Pre-defined temperature ladder. Defaults to automatic creation based on other parameters.
- `chain_type`: Type of chain to return (e.g., `Any`, `DifferentialEvolutionOutput`, `MCMCChains.Chains`). Defaults to `DifferentialEvolutionOutput`.
- `save_final_state`: Whether to return the final state along with samples, if true the output will be (samples::chain_type, final_state). Defaults to `false`.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` and the internal initialization step (e.g., `memory_thin_interval`, `silent`). See [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments).

# Returns
- Named tuple containing samples, sampler scheme, and optionally burn-in samples

# Example
```@example deMCzs
using DEMetropolis, Random, Distributions

# Define a simple log-density function
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Run adaptive differential evolution MCMC with convergence checking
result = deMCzs(model_wrapper, 1000; n_chains = 3, maximum_R̂ = 1.1)
```

See also [`deMC`](@ref), [`DREAMz`](@ref), [`r̂_stopping_criteria`](@ref).
"""
function deMCzs(
        model_wrapper::LogDensityModel, check_every::Int;
        warmup_epochs::Int = 5,
        epoch_limit::Int = 20,
        maximum_R̂::T = 1.2,
        initial_state::Union{AbstractVector{<:AbstractVector{T}}, Nothing} = nothing,
        save_burnt::Bool = false,
        memory_size::Int = check_every * 10,
        memory_refill::Bool = true,
        rng::AbstractRNG = default_rng(),
        γ::Union{Nothing, T} = nothing,
        γₛ::Union{Nothing, T} = nothing,
        p_snooker::Union{Nothing, T} = 0.1,
        β::Distributions.Uniform{T} = Distributions.Uniform(-1.0e-4, 1.0e-4),
        chain_type = DifferentialEvolutionOutput,
        thin::Int = 1,
        kwargs...
    ) where {T <: Real}

    #build sampler scheme
    if p_snooker == 0
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ, β = β, n_dims = dimension(model_wrapper.logdensity))
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ, β = β, n_dims = dimension(model_wrapper.logdensity)),
            setup_snooker_update(γ = γₛ),
            w = [1 - p_snooker, p_snooker]
        )
    end

    n_burnin = check_every * warmup_epochs

    return sample(
        rng,
        model_wrapper,
        sampler_scheme,
        r̂_stopping_criteria;
        maximum_R̂ = maximum_R̂,
        check_every = check_every,
        memory_size = memory_size,
        memory_refill = memory_refill,
        minimum_iterations = save_burnt ? n_burnin + 1 : 0,
        maximum_iterations = save_burnt ? (check_every * epoch_limit) + n_burnin :
            (check_every * epoch_limit),
        num_warmup = n_burnin,
        initial_position = initial_state,
        thinning = thin,
        discard_initial = save_burnt ? 0 : n_burnin,
        chain_type = chain_type,
        kwargs...
    )
end

"""
    DREAMz(model_wrapper, check_every; kwargs...)

Run the Differential Evolution Adaptive Metropolis (DREAMz) sampler.

This advanced adaptive sampler runs until convergence and uses subspace sampling with
adaptive crossover probabilities. It can switch between scaling factors and includes
outlier chain detection/replacement. The algorithm adapts during warm-up and can use
memory-based sampling for efficiency.

Based on Vrugt et al. (2009), see doi.org/10.1515/IJNSNS.2009.10.3.273.

# Arguments
- `model_wrapper`: LogDensityModel containing the target log-density function
- `check_every`: Number of iterations per chain per convergence check

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs for adaptation. Defaults to 5.
- `epoch_limit`: Maximum number of total epochs. Defaults to 20.
- `maximum_R̂`: Convergence threshold for Gelman-Rubin diagnostic. Defaults to 1.2.
- `n_chains`: Number of chains. Defaults to `max(dimension(ld) * 2, 3)`.
- `N₀`: Size of initial population. Defaults to `n_chains * 2`.
- `initial_state`: Initial population state. Defaults to random initialization.
- `memory`: Use memory-based sampling (`true`) or memoryless DREAM (`false`). Defaults to `true`.
- `save_burnt`: Save warm-up samples in output. Defaults to `true`.
- `parallel`: Run chains in parallel using threading. Defaults to `false`.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `memory_size`: Maximum number of positions retained per chain in memory during initialization. Defaults to `check_every * 10`.
- `memory_refill`: When memory is full, overwrite from the start (cyclic) if `true`. Defaults to `true`.
- `memory_thin_interval`: If > 0, only every `memory_thin_interval`-th accepted state is stored in memory. Defaults to `0` (forwarded via keyword arguments).
- `silent`: Suppress informational initialization logs when `true`. Defaults to `false` (forwarded via keyword arguments).
- `γ₁`: Primary scaling factor for subspace updates. Defaults to adaptive.
- `γ₂`: Secondary scaling factor. Defaults to 1.0.
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.2.
- `n_cr`: Number of crossover probabilities for adaptation. Defaults to 3.
- `cr₁`: Crossover probability for `γ₁`. Defaults to adaptive.
- `cr₂`: Crossover probability for `γ₂`. Defaults to adaptive.
- `ϵ`: Additive noise distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `e`: Multiplicative noise distribution. Defaults to `Normal(0.0, 1e-2)`.
- `δ`: Number of difference vectors distribution. Defaults to `DiscreteUniform(1, 3)`.
- `thin`: Thinning interval for saved samples. Defaults to 1.
- `n_hot_chains`: Number of hot chains for parallel tempering. Defaults to 0 (no parallel tempering).
- `max_temp_pt`: Maximum temperature for parallel tempering. Defaults to 2*sqrt(dimension).
- `max_temp_sa`: Maximum temperature for simulated annealing. Defaults to `max_temp_pt`.
- `α`: Temperature ladder spacing parameter. Defaults to 1.0.
- `annealing`: Whether to use simulated annealing. Defaults to `false`.
- `annealing_steps`: Number of annealing steps. Defaults to 0 or the number of warmup-steps (when using AbstractMCMC.sample).
- `chain_type`: Type of chain to return (e.g., `Any`, `DifferentialEvolutionOutput`, `MCMCChains.Chains`). Defaults to `DifferentialEvolutionOutput`.
- `save_final_state`: Whether to return the final state along with samples, if true the output will be (samples::chain_type, final_state). Defaults to `false`.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` and the internal initialization step (e.g., `memory_thin_interval`, `silent`). See [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments).

# Returns
- Named tuple containing samples, sampler scheme, and optionally burn-in samples

# Example
```@example DREAMz
using DEMetropolis, Random, Distributions

# Define a simple log-density function
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Run DREAM with subspace sampling
result = DREAMz(model_wrapper, 1000; n_chains = 10, memory = false)
```

See also [`deMC`](@ref), [`deMCzs`](@ref), [`setup_subspace_sampling`](@ref).
"""
function DREAMz(
        model_wrapper::LogDensityModel, check_every::Int;
        warmup_epochs::Int = 5,
        epoch_limit::Int = 20,
        maximum_R̂::T = 1.2,
        memory_size::Int = check_every * 10,
        memory_refill::Bool = true,
        initial_state::Union{AbstractVector{<:AbstractVector{T}}, Nothing} = nothing,
        save_burnt::Bool = false,
        rng::AbstractRNG = default_rng(),
        γ₁::Union{Nothing, T} = nothing,
        γ₂::Union{Nothing, T} = 1.0,
        p_γ₂::Union{Nothing, T} = 0.2,
        n_cr::Int = 3,
        cr₁::Union{Nothing, T} = nothing,
        cr₂::Union{Nothing, T} = nothing,
        ϵ::Distributions.Uniform{T} = Distributions.Uniform(-1.0e-4, 1.0e-4),
        e::Distributions.Normal{T} = Distributions.Normal(0.0, 1.0e-2),
        δ::Distributions.DiscreteUniform = Distributions.DiscreteUniform(1, 3),
        chain_type = DifferentialEvolutionOutput,
        thin::Int = 1,
        kwargs...
    ) where {T <: Real}

    #build sampler scheme
    if p_γ₂ == 0
        sampler_scheme = setup_sampler_scheme(
            setup_subspace_sampling(γ = γ₁, n_cr = n_cr, cr = cr₁, δ = δ, ϵ = ϵ, e = e)
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_subspace_sampling(γ = γ₁, n_cr = n_cr, cr = cr₁, δ = δ, ϵ = ϵ, e = e),
            setup_subspace_sampling(γ = γ₂, n_cr = n_cr, cr = cr₂, δ = δ, ϵ = ϵ, e = e),
            w = [1 - p_γ₂, p_γ₂]
        )
    end

    n_burnin = check_every * warmup_epochs

    return sample(
        rng,
        model_wrapper,
        sampler_scheme,
        r̂_stopping_criteria;
        maximum_R̂ = maximum_R̂,
        check_every = check_every,
        memory_size = memory_size,
        memory_refill = memory_refill,
        minimum_iterations = save_burnt ? n_burnin + 1 : 0,
        maximum_iterations = save_burnt ? (check_every * epoch_limit) + n_burnin :
            (check_every * epoch_limit),
        num_warmup = n_burnin,
        initial_position = initial_state,
        thinning = thin,
        discard_initial = save_burnt ? 0 : n_burnin,
        chain_type = chain_type,
        kwargs...
    )
end
