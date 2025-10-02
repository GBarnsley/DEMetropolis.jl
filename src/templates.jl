function build_initial_state(rng::AbstractRNG, ld::TransformedLogDensity,
        initial_state::Union{Nothing, Array{<:Real, 2}},
        n_chains::Int, N₀::Int, memory::Bool)
    if memory
        N₀ = max(N₀, n_chains + 3)
    else
        N₀ = n_chains
    end
    if isnothing(initial_state)
        return randn(rng, N₀, dimension(ld))
    else
        current_N = size(initial_state, 1)
        current_pars = size(initial_state, 2)
        if current_pars != dimension(ld)
            error("Number of parameters in initial state must be equal to the number of parameters in the log density")
        end
        if current_N == N₀
            return initial_state
        elseif current_N < N₀
            @warn "Initial state is smaller than the number of chains. Expanding initial state."
            return cat(
                randn(rng, eltype(initial_state), N₀ - current_N, current_pars),
                initial_state,
                dims = 1
            )
        else
            @warn "Initial state is larger than the number of chains. Shrinking initial state."
            #shrink initial state
            return initial_state[1:N₀, :]
        end
    end
end

"""
Run the Differential Evolution Markov Chain (DE-MC) sampler proposed by ter Braak (2006)

This sampler uses the `de_update` step. It can optionally switch between two `γ` values (`γ₁` and `γ₂`) with probability `p_γ₂`.
This is so that the sampler can occasionally move between modes, by having `γ₂ = 1` while γ₁ remains the optimal value based on the dimension of the problem.

This algorithm varies slightly from the original. Updates within a population occur on the previous position of that population.
i.e. if chain 1 has been updated (a₁ → a₂) and chain 2 picks chain 1 to update from, then the value of chain 1 used by chain 2 is the pre-update version of chain 1 (a₁).
This change allows the algorithm to be easily parallelised.

See doi.org/10.1007/s11222-006-8769-1 for more information on sampler.

# Arguments
- `ld`: The log-density function to sample from, intended to be a LogDensityProblem.
- `n_its`: The number of sampling iterations per chain.

# Keyword Arguments
- `n_burnin`: Number of burn-in iterations. Defaults to `n_its * 5`.
- `n_chains`: Number of chains. Defaults to `dimension(ld) * 2`.
- `initial_state`: Initial states for the chains. Defaults to `randn(rng, n_chains, dimension(ld))`.
- `N₀`: Size of the initial population (must be >= n_chains + 3), only used if using memory-based sampling. Defaults to `n_chains * 2`.
- `memory`: Use memory-based sampling (`true`) or memoryless (`false`). Defaults to `false`.
- `save_burnt`: Save burn-in samples. Defaults to `false`.
- `parallel`: Run chains in parallel. Defaults to `false`.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `diagnostic_checks`: Diagnostic checks to run during burn-in. Defaults to `nothing`.
- `check_epochs`: Splits `n_burnin` into `check_epochs + 1` epochs and applies the diagnostic checks at the end of each epoch, other than the final epoch. Defaults to 1.
- `thin`: Thinning interval. Defaults to 1.
- `γ₁`: Primary scaling factor for DE update. Defaults to `2.38 / sqrt(2 * dim)`.
- `γ₂`: Secondary scaling factor for DE update. Defaults to 1.0.
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.1.
- `β`: Noise distribution for DE update. Defaults to `Uniform(-1e-4, 1e-4)`.

# Returns
- A named tuple containing the samples, sampler scheme, and potentially burn-in samples.

# Example
```jldoctest
julia> deMC(ld, 1000; n_chains = 10)
```
See also [`composite_sampler`](@ref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function deMC(
        ld::TransformedLogDensity, n_its::Int;
        n_burnin::Int = n_its * 5,
        n_chains::Int = dimension(ld) * 2,
        N₀::Int = n_chains,
        initial_state::Union{Array{T, 2}, Nothing} = nothing,
        memory::Bool = false,
        save_burnt::Bool = false,
        parallel::Bool = false,
        rng::AbstractRNG = default_rng(),
        diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing,
        check_epochs::Int = 1,
        thin::Int = 1,
        γ₁::Union{Nothing, T} = nothing,
        γ₂::T = 1.0,
        p_γ₂::T = 0.1,
        β::Distributions.Uniform{T} = Distributions.Uniform(-1e-4, 1e-4)
) where {T <: Real}
    if n_chains < dimension(ld) && !memory
        @warn "Number of chains should be greater than or equal to the number of parameters"
    end

    #setup initial state
    initial_state = build_initial_state(rng, ld, initial_state, n_chains, N₀, memory)

    #build sampler scheme
    if γ₁ != γ₂
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ₂, β = β),
            setup_de_update(ld, γ = γ₁, β = β),
            w = [p_γ₂, 1 - p_γ₂]
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ₁, β = β)
        )
    end

    composite_sampler(
        ld, n_its, n_chains, memory, initial_state, sampler_scheme;
        save_burnt = save_burnt, rng = rng, n_burnin = n_burnin, parallel = parallel,
        diagnostic_checks = diagnostic_checks, check_epochs = check_epochs, thin = thin
    )
end

"""
Run the Differential Evolution Markov Chain with snooker update and historic sampling (DE-MCzs) sampler proposed by ter Braak and Vrugt (2008).

This sampler runs until a `stopping_criteria` (default: Rhat of the last 50% of the chains is <1.2) is met.
The sampler can occasionally propose snooker updates which can sample areas away from the current chains.
Combined with the adaptive memory-based sampling this sampler can efficiently sample from a problem where n_chains < the dimension of the problem.

See: doi.org/10.1007/s11222-008-9104-9 for more information

# Arguments
- `ld`: The log-density function to sample from, intended to be a LogDensityProblem.
- `epoch_size`: The number of saved iterations per chain per epoch.

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs. Defaults to 5.
- `epoch_limit`: Maximum number of sampling epochs. Defaults to 20.
- `n_chains`: Number of chains. Defaults to `dimension(ld) * 2`.
- `N₀`: Size of the initial population (must be >= n_chains + 3). Defaults to `n_chains * 2`.
- `initial_state`: Initial population state. Defaults to `randn(rng, N₀, dimension(ld))`.
- `memory`: Use memory based sampling? Defaults to `true`.
- `save_burnt`: Save warm-up samples. Defaults to `true`.
- `parallel`: Run chains in parallel with multithreading. Defaults to `false`.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `diagnostic_checks`: Diagnostic checks during warm-up. Defaults to `nothing`.
- `stopping_criteria`: Criterion to stop sampling. Defaults to `R̂_stopping_criteria()`.
- `γ`: Scaling factor for the DE update. Defaults to `2.38 / sqrt(2 * dim)`.
- `γₛ`: Scaling factor for the Snooker update. Defaults to `2.38 / sqrt(2)`.
- `p_snooker`: Probability of using the Snooker update. Defaults to 0.1.
- `β`: Noise distribution for DE update. Defaults to `Uniform(-1e-4, 1e-4)`.
- `thin`: Thinning interval. Defaults to 10.

# Returns
- A named tuple containing the samples, sampler scheme, and potentially burn-in samples.

# Example
```jldoctest
julia> deMCzs(ld, 1000; n_chains = 3)
```
See also [`composite_sampler`](@ref), [`deMC`](@ref), [`DREAMz`](@ref).
"""
function deMCzs(
        ld::TransformedLogDensity, epoch_size::Int;
        warmup_epochs::Int = 5,
        epoch_limit::Int = 20,
        n_chains::Int = dimension(ld) * 2,
        N₀::Int = n_chains * 2,
        initial_state::Union{Array{<:Real, 2}, Nothing} = nothing,
        memory::Bool = true,
        save_burnt::Bool = true,
        parallel::Bool = false,
        rng::AbstractRNG = default_rng(),
        diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing,
        stopping_criteria::stopping_criteria_struct = R̂_stopping_criteria(),
        γ::Union{Nothing, T} = nothing,
        γₛ::Union{Nothing, T} = nothing,
        p_snooker::Union{Nothing, T} = 0.1,
        β::Distributions.Uniform{T} = Distributions.Uniform(-1e-4, 1e-4),
        thin::Int = 10
) where {T <: Real}
    if n_chains < dimension(ld)
        @warn "Number of chains should be greater than or equal to the number of parameters"
    end

    #setup initial state
    initial_state = build_initial_state(rng, ld, initial_state, n_chains, N₀, memory)

    #build sampler scheme
    if p_snooker == 0
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ, β = β)
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ, β = β),
            setup_snooker_update(γ = γₛ),
            w = [1 - p_snooker, p_snooker]
        )
    end

    composite_sampler(
        ld, epoch_size, n_chains, memory, initial_state, sampler_scheme, stopping_criteria;
        save_burnt = save_burnt, rng = rng, warmup_epochs = warmup_epochs,
        parallel = parallel, epoch_limit = epoch_limit, diagnostic_checks = diagnostic_checks,
        thin = thin
    )
end

"""
Run the Differential Evolution Adaptive Metropolis (DREAMz) sampler

This sampler runs until a `stopping_criteria` (default: Rhat of the last 50% of the chains is <1.2) is met.
The sampler uses `subspace_sampling`, where the cross-over probability is adapted during the burn-in period.
It can optionally switch between two `γ` values, so that γ₁ can be the optimal value (based on sampled parameters) and γ₂ can be some fixed value (i.e. 1) so that the sampler can switch modes.
This sampler also checks for outlier chains (where the mean log-density falls outside the IQR) and replaces then with the position of the chain with the highest log-density.
This step breaks detailed balance its not performed in the last epoch of the warm-up period.

By default this is a memory-based sampler (DREAMz). Setting `memory = false` makes this the DREAM sampler.

See doi.org/10.1515/IJNSNS.2009.10.3.273 for more info.

# Arguments
- `ld`: The log-density function to sample from, intended to be a LogDensityProblem.
- `epoch_size`: The number of saved iterations per chain per epoch.

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs. Defaults to 5. Crossover probabilities are adapted in this period.
- `epoch_limit`: Maximum number of sampling epochs. Defaults to 20.
- `n_chains`: Number of chains. Defaults to `dimension(ld) * 2`.
- `N₀`: Size of the initial population (must be >= n_chains). Defaults to `n_chains`. Only the first `n_chains` will be used if `memory = false`.
- `initial_state`: Initial population state. Defaults to `randn(rng, N₀, dimension(ld))`.
- `memory`: Use memory-based sampling (`true`) or memoryless (`false`). Defaults to `true`.
- `save_burnt`: Save warm-up samples. Defaults to `true`.
- `parallel`: Run chains in parallel. Defaults to `false`.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `diagnostic_checks`: Diagnostic checks during warm-up. Defaults to `[ld_check()]`.
- `stopping_criteria`: Criterion to stop sampling. Defaults to `R̂_stopping_criteria()`.
- `γ₁`: Primary scaling factor for subspace update. Defaults to `nothing` (uses `2.38 / sqrt(2 * δ * d)`). Can also be a `Real` value.
- `γ₂`: Secondary scaling factor for subspace update. Defaults to 1.0. Can also be a `Real` value
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.2.
- `n_cr`: Number of crossover probabilities to adapt if `cr₁`/`cr₂` are `nothing`. Defaults to 3.
- `cr₁`: Crossover probability distribution/value for `γ₁`. Defaults to `nothing` (adaptive). Can also be a `Real` value (<1) or a `Distributions.UnivariateDistribution`, in either case it is not adapted.
- `cr₂`: Crossover probability distribution/value for `γ₂`. See above.
- `ϵ`: Additive noise distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `e`: Multiplicative noise distribution. Defaults to `Normal(0.0, 1e-2)`.
- `δ`: Number of difference vectors distribution. Defaults to `DiscreteUniform(1, 3)`.
- `thin`: Thinning interval. Defaults to 1.

# Returns
- A named tuple containing the samples, sampler scheme, and potentially burn-in samples.

# Example
```jldoctest
julia> DREAMz(ld, 1000; n_chains = 10)
```
See also [`composite_sampler`](@ref), [`deMC`](@ref), [`deMCzs`](@ref).
"""
function DREAMz(
        ld::TransformedLogDensity, epoch_size::Int;
        warmup_epochs::Int = 5,
        epoch_limit::Int = 20,
        n_chains::Int = dimension(ld) * 2,
        N₀::Int = n_chains,
        initial_state::Union{Array{<:Real, 2}, Nothing} = nothing,
        memory::Bool = true,
        save_burnt::Bool = true,
        parallel::Bool = false,
        rng::AbstractRNG = default_rng(),
        diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing,
        stopping_criteria::stopping_criteria_struct = R̂_stopping_criteria(),
        γ₁::Union{Nothing, T} = nothing,
        γ₂::Union{Nothing, T} = 1.0,
        p_γ₂::Union{Nothing, T} = 0.2,
        n_cr::Int = 3,
        cr₁::Union{Nothing, T} = nothing,
        cr₂::Union{Nothing, T} = nothing,
        ϵ::Distributions.Uniform{T} = Distributions.Uniform(-1e-4, 1e-4),
        e::Distributions.Normal{T} = Distributions.Normal(0.0, 1e-2),
        δ::Distributions.DiscreteUniform = Distributions.DiscreteUniform(1, 3),
        thin::Int = 1
) where {T <: Real}
    if n_chains < dimension(ld)
        @warn "Number of chains should be greater than or equal to the number of parameters"
    end

    #setup initial state
    initial_state = build_initial_state(rng, ld, initial_state, n_chains, N₀, memory)

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

    composite_sampler(
        ld, epoch_size, n_chains, memory, initial_state, sampler_scheme, stopping_criteria;
        save_burnt = save_burnt, rng = rng, warmup_epochs = warmup_epochs,
        parallel = parallel, epoch_limit = epoch_limit, diagnostic_checks = diagnostic_checks,
        thin = thin
    )
end
