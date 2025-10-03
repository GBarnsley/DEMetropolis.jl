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
- `save_burnt`: Save burn-in samples in output. Defaults to `false`.
- `parallel`: Run chains in parallel using threading. Defaults to `false`.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `thin`: Thinning interval for saved samples. Defaults to 1.
- `γ₁`: Primary scaling factor. Defaults to `2.38 / sqrt(2 * dim)`.
- `γ₂`: Secondary scaling factor for mode switching. Defaults to 1.0.
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.1.
- `β`: Noise distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` (see [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments)).

# Returns
- Named tuple containing samples and optionally burn-in samples

# Example
```jldoctest
julia> result = deMC(model_wrapper, 1000; n_chains = 10, parallel = true)
```

See also [`deMCzs`](@ref), [`DREAMz`](@ref), [`setup_de_update`](@ref).
"""
function deMC(
        model_wrapper::LogDensityModel, n_its::Int;
        n_burnin::Int = n_its * 5,
        n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
        N₀::Int = n_chains,
        initial_state::Union{AbstractVector{<:AbstractVector{T}}, Nothing} = nothing,
        memory::Bool = false,
        save_burnt::Bool = false,
        parallel::Bool = false,
        rng::AbstractRNG = default_rng(),
        thin::Int = 1,
        γ₁::Union{Nothing, T} = nothing,
        γ₂::T = 1.0,
        p_γ₂::T = 0.1,
        β::ContinuousUnivariateDistribution = Uniform(-1e-4, 1e-4),
        kwargs...
) where {T<:Real}

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

    return process_outputs(sample(
        rng,
        model_wrapper,
        sampler_scheme,
        n_its;
        num_warmup = n_burnin,
        n_chains = n_chains,
        N₀ = N₀,
        initial_position = initial_state,
        thinning = thin,
        memory = memory,
        parallel = parallel,
        discard_initial = save_burnt ? 0 : n_burnin,
        kwargs...
    ); save_burnt = save_burnt, n_burnin = n_burnin)
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
- `γ`: Scaling factor for DE updates. Defaults to `2.38 / sqrt(2 * dim)`.
- `γₛ`: Scaling factor for snooker updates. Defaults to `2.38 / sqrt(2)`.
- `p_snooker`: Probability of snooker moves. Defaults to 0.1.
- `β`: Noise distribution for DE updates. Defaults to `Uniform(-1e-4, 1e-4)`.
- `thin`: Thinning interval for saved samples. Defaults to 10.
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` (see [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments)).

# Returns
- Named tuple containing samples, sampler scheme, and optionally burn-in samples

# Example
```jldoctest
julia> result = deMCzs(model_wrapper, 1000; n_chains = 3, maximum_R̂ = 1.1)
```

See also [`deMC`](@ref), [`DREAMz`](@ref), [`r̂_stopping_criteria`](@ref).
"""
function deMCzs(
        model_wrapper::LogDensityModel, check_every::Int;
        warmup_epochs::Int = 5,
        epoch_limit::Int = 20,
        maximum_R̂::T = 1.2,
        n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
        N₀::Int = n_chains * 2,
        initial_state::Union{AbstractVector{<:AbstractVector{T}}, Nothing} = nothing,
        memory::Bool = true,
        save_burnt::Bool = true,
        parallel::Bool = false,
        rng::AbstractRNG = default_rng(),
        γ::Union{Nothing, T} = nothing,
        γₛ::Union{Nothing, T} = nothing,
        p_snooker::Union{Nothing, T} = 0.1,
        β::Distributions.Uniform{T} = Distributions.Uniform(-1e-4, 1e-4),
        thin::Int = 10,
        kwargs...
) where {T<:Real}

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

    return process_outputs(sample(
        rng,
        model_wrapper,
        sampler_scheme,
        r̂_stopping_criteria;
        maximum_R̂ = maximum_R̂,
        check_every = check_every,
        minimum_iterations = save_burnt ? n_burnin + 1 : 0,
        maximum_iterations = save_burnt ? (check_every * epoch_limit) + n_burnin : (check_every * epoch_limit),
        num_warmup = n_burnin,
        n_chains = n_chains,
        N₀ = N₀,
        parallel = parallel,
        initial_position = initial_state,
        thinning = thin,
        memory = memory,
        discard_initial = save_burnt ? 0 : n_burnin,
        kwargs...
    ); save_burnt = save_burnt, n_burnin = n_burnin)
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
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` (see [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments)).

# Returns
- Named tuple containing samples, sampler scheme, and optionally burn-in samples

# Example
```jldoctest
julia> result = DREAMz(model_wrapper, 1000; n_chains = 10, memory = false)
```

See also [`deMC`](@ref), [`deMCzs`](@ref), [`setup_subspace_sampling`](@ref).
"""
function DREAMz(
        model_wrapper::LogDensityModel, check_every::Int;
        warmup_epochs::Int = 5,
        epoch_limit::Int = 20,
        maximum_R̂::T = 1.2,
        n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
        N₀::Int = n_chains * 2,
        initial_state::Union{AbstractVector{<:AbstractVector{T}}, Nothing} = nothing,
        memory::Bool = true,
        save_burnt::Bool = true,
        parallel::Bool = false,
        rng::AbstractRNG = default_rng(),
        γ₁::Union{Nothing, T} = nothing,
        γ₂::Union{Nothing, T} = 1.0,
        p_γ₂::Union{Nothing, T} = 0.2,
        n_cr::Int = 3,
        cr₁::Union{Nothing, T} = nothing,
        cr₂::Union{Nothing, T} = nothing,
        ϵ::Distributions.Uniform{T} = Distributions.Uniform(-1e-4, 1e-4),
        e::Distributions.Normal{T} = Distributions.Normal(0.0, 1e-2),
        δ::Distributions.DiscreteUniform = Distributions.DiscreteUniform(1, 3),
        thin::Int = 1,
        kwargs...
) where {T<:Real}

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

    return process_outputs(sample(
        rng,
        model_wrapper,
        sampler_scheme,
        r̂_stopping_criteria;
        maximum_R̂ = maximum_R̂,
        check_every = check_every,
        minimum_iterations = save_burnt ? n_burnin + 1 : 0,
        maximum_iterations = save_burnt ? (check_every * epoch_limit) + n_burnin : (check_every * epoch_limit),
        num_warmup = n_burnin,
        n_chains = n_chains,
        N₀ = N₀,
        initial_position = initial_state,
        parallel = parallel,
        thinning = thin,
        memory = memory,
        discard_initial = save_burnt ? 0 : n_burnin,
        kwargs...
    ); save_burnt = save_burnt, n_burnin = n_burnin)
end
