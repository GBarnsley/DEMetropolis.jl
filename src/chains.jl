struct DifferentialEvolutionAdaptiveStatic{T} <: AbstractDifferentialEvolutionAdaptiveState{T} end

struct DifferentialEvolutionState{T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}, A<:AbstractDifferentialEvolutionAdaptiveState{T}} <: AbstractDifferentialEvolutionState{T, V, VV, A}
    "current position"
    x::VV
    "log density at current position"
    ld::V
    "struct for holding the status of the adaptive scheme"
    adaptive_state::A
end

DifferentialEvolutionState(x::VV, ld::V) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}} = DifferentialEvolutionState{T, V, VV, DifferentialEvolutionAdaptiveStatic{T}}(x, ld, DifferentialEvolutionAdaptiveStatic{T}())

struct DifferentialEvolutionStateMemory{T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}, A<:AbstractDifferentialEvolutionAdaptiveState{T}} <: AbstractDifferentialEvolutionState{T, V, VV, A}
    "current position"
    x::VV
    "log density at current position"
    ld::V
    "memory of past positions"
    mem_x::VV
    "struct for holding the status of the adaptive scheme"
    adaptive_state::A
end

DifferentialEvolutionStateMemory(x::VV, ld::V, mem_x::VV) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}} = DifferentialEvolutionStateMemory{T, V, VV, DifferentialEvolutionAdaptiveStatic{T}}(x, ld, mem_x, DifferentialEvolutionAdaptiveStatic{T}())

struct DifferentialEvolutionSample{V<:AbstractVector{<:Real}, VV<:AbstractVector{V}}
    "current position"
    x::VV
    "log density at current position"
    ld::V
end

function pick_chains(rng::AbstractRNG, state::DifferentialEvolutionState, current_chain::Int, n_chains::Int)
    #sample up to the current position
    indices = StatsBase.sample(rng, 1:(length(state.x) - 1), n_chains, replace = false)
    indices[indices .>= current_chain] .+= 1
    return state.x[indices]
end

function pick_chains(rng::AbstractRNG, state::DifferentialEvolutionStateMemory,  current_chain::Int, n_chains::Int)
    #sample up to the current position
    return StatsBase.sample(rng, state.mem_x, n_chains, replace = false)
end

function update_state(
    state::DifferentialEvolutionState{T, V, VV, A};
    adaptive_state::AbstractDifferentialEvolutionAdaptiveState{T} = state.adaptive_state,
    xₚ::VV = state.x,
    ldₚ::V = state.ld,
    kwargs...
    ) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}, A<:AbstractDifferentialEvolutionAdaptiveState{T}}
    return DifferentialEvolutionState(xₚ, ldₚ, adaptive_state)
end

function update_state(
    state::DifferentialEvolutionStateMemory{T, V, VV, A};
    adaptive_state::AbstractDifferentialEvolutionAdaptiveState{T} = state.adaptive_state,
    x::VV = state.x,
    ld::V = state.ld,
    update_memory::Bool = false,
    kwargs...
    ) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}, A<:AbstractDifferentialEvolutionAdaptiveState{T}}
    if update_memory
        return DifferentialEvolutionStateMemory(x, ld, cat(state.mem_x, x; dims=1), adaptive_state)
    else
        return DifferentialEvolutionStateMemory(x, ld, state.mem_x, adaptive_state)
    end
end

function update_chain!(model, rng, xₚ, ldₚ, x, ld, offset, i)
    if isinf(offset) & (sign(offset) == -1.0)
        xₚ[i] = x[i]
        ldₚ[i] = ld[i]
        return false
    else
        ldₚ[i] = logdensity(model, xₚ[i])
        if log(rand(rng)) > (ldₚ[i] - ld[i] + offset)
            xₚ[i] = x[i]
            ldₚ[i] = ld[i]
            return false
        else
            return true
        end
    end
end

# non-adaptive step
"""
    step(rng, model_wrapper, sampler, state; parallel=false, update_memory=true, kwargs...)

Perform a single MCMC step using differential evolution sampling.

This is the core sampling function that proposes new states for all chains and accepts
or rejects them according to the Metropolis criterion. For adaptive samplers, the
function automatically fixes adaptive parameters before sampling.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Differential evolution sampler (any AbstractDifferentialEvolutionSampler)
- `state`: Current state of all chains

# Keyword Arguments
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`. Advisable for slow models.
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers). Defaults to `true`. Useful is memory has grown too large.
- `kwargs...`: Additional keyword arguments passed to update functions (see https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments)

# Returns
- `sample`: DifferentialEvolutionSample containing new positions and log-densities
- `new_state`: Updated state for the next iteration

# Example
```jldoctest
julia> sample, new_state = step(rng, model, sampler, state; parallel=true)
```

See also [`step_warmup`](@ref), [`AbstractMCMC.sample`](@ref).
"""
function step(
    rng::AbstractRNG,
    model_wrapper::LogDensityModel,
    sampler::AbstractDifferentialEvolutionSampler,
    state::AbstractDifferentialEvolutionState{T, V, VV, DifferentialEvolutionAdaptiveStatic{T}};
    parallel::Bool = false,
    update_memory::Bool = true,
    kwargs...
) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}}
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current states
    (; x, ld, adaptive_state) = state

    # loop through chains running the update
    xₚ = similar(x)
    ldₚ = similar(ld)
    if parallel
        rngs = [Random.seed!(copy(rng),  rand(rng, UInt)) for i in eachindex(x)]
        @inbounds Threads.@threads for i in eachindex(x)
            prop = proposal(rngs[i], sampler, state, i)
            xₚ[i] = prop.xₚ
            update_chain!(model, rngs[i], xₚ, ldₚ, x, ld, prop.offset, i)
        end
    else
        @inbounds for i in eachindex(x)
            chain_rng = Random.seed!(copy(rng),  rand(rng, UInt)) #so its identical to parallel
            prop = proposal(chain_rng, sampler, state, i)
            xₚ[i] = prop.xₚ
            update_chain!(model, chain_rng, xₚ, ldₚ, x, ld, prop.offset, i)
        end
    end

    return DifferentialEvolutionSample(xₚ, ldₚ), update_state(state; adaptive_state = adaptive_state, x = xₚ, ld = ldₚ, update_memory = update_memory)
end

#previously adapted step
"""
    fix_sampler(sampler::AbstractDifferentialEvolutionSampler, adaptive_state::AbstractDifferentialEvolutionAdaptiveState)

Fix adaptive parameters of a sampler to their current adapted values.

For non-adaptive samplers, returns the sampler unchanged. For adaptive samplers,
returns a new sampler with the adaptive parameters fixed to their current values
in the `adaptive_state`.

# Arguments
- `sampler`: The differential evolution sampler to fix
- `adaptive_state`: The adaptive state containing current parameter values

# Returns
- A sampler with fixed (non-adaptive) parameters

# Example
```jldoctest
julia> fixed_sampler = fix_sampler(adaptive_sampler, state.adaptive_state)
```

See also [`fix_sampler_state`](@ref).
"""
fix_sampler(sampler::AbstractDifferentialEvolutionSampler, adaptive_state::AbstractDifferentialEvolutionAdaptiveState) = sampler

"""
    fix_sampler_state(sampler::AbstractDifferentialEvolutionSampler, state::DifferentialEvolutionState)

Fix adaptive sampler parameters and return a corresponding non-adaptive state.

Takes an adaptive sampler and state, fixes the sampler's adaptive parameters to their
current values, and returns both the fixed sampler and a simplified state without
adaptive components.

# Arguments
- `sampler`: The differential evolution sampler (potentially adaptive)
- `state`: The current sampler state (DifferentialEvolutionState or DifferentialEvolutionStateMemory)

# Returns
- `fixed_sampler`: Sampler with adaptive parameters fixed to current values
- `fixed_state`: State without adaptive components

# Example
```jldoctest
julia> fixed_sampler, fixed_state = fix_sampler_state(sampler, state)
```

See also [`fix_sampler`](@ref).
"""
function fix_sampler_state(sampler::AbstractDifferentialEvolutionSampler, state::DifferentialEvolutionState)
    return fix_sampler(sampler, state.adaptive_state), DifferentialEvolutionState(state.x, state.ld)
end

function fix_sampler_state(sampler::AbstractDifferentialEvolutionSampler, state::DifferentialEvolutionStateMemory)
    return fix_sampler(sampler, state.adaptive_state), DifferentialEvolutionStateMemory(state.x, state.ld, state.mem_x)
end

function step(
    rng::AbstractRNG,
    model_wrapper::LogDensityModel,
    sampler::AbstractDifferentialEvolutionSampler,
    state::AbstractDifferentialEvolutionState{T, V, VV, A};
    update_memory::Bool = true,
    kwargs...
) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}, A<:AbstractDifferentialEvolutionAdaptiveState{T}}

    fixed_sampler, fixed_state = fix_sampler_state(sampler, state)
    sample, new_state = step(rng, model_wrapper, fixed_sampler, fixed_state; kwargs...)

    return sample, update_state(state; x = new_state.x, ld = new_state.ld, update_memory = update_memory)
end

function initialize_adaptive_state(sampler::AbstractDifferentialEvolutionSampler, model_wrapper::LogDensityModel, n_chains::Int)
    return DifferentialEvolutionAdaptiveStatic{Float64}()
end

"""
    step(rng, model_wrapper, sampler; n_chains, memory=true, N₀, adapt=true, initial_position=nothing, parallel=false, kwargs...)

Initialize differential evolution sampling by setting up chains and computing initial state.

This function serves as the entry point for differential evolution MCMC sampling. It handles
chain initialization, memory setup for memory-based samplers, adaptive state initialization,
and returns the initial sample and state that can be used with `AbstractMCMC.sample`.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Differential evolution sampler to use

# Keyword Arguments
- `n_chains`: Number of parallel chains. Defaults to `max(2 * dimension, 3)` for adequate mixing.
- `memory`: Whether to use memory-based sampling that stores past positions. Memory-based
  samplers can be more efficient for high-dimensional problems. Defaults to `true`.
- `N₀`: Initial memory size for memory-based samplers. Should be ≥ `n_chains`. Defaults to `2 * n_chains`.
- `adapt`: Whether to enable adaptive behavior during warm-up (if the sampler supports it).
  Defaults to `true`.
- `initial_position`: Starting positions for chains. Can be `nothing` (random initialization),
  or a vector of parameter vectors. If the provided vector is smaller than `n_chains`, it will
  be expanded; if larger and `memory=true`, excess positions become initial memory. Defaults to `nothing`.
- `parallel`: Whether to evaluate initial log-densities in parallel. Useful for expensive models.
  Defaults to `false`.
- `kwargs...`: Additional keyword arguments (unused in this method)

# Returns
- `sample`: DifferentialEvolutionSample containing initial positions and log-densities
- `state`: Initial state (DifferentialEvolutionState or DifferentialEvolutionStateMemory)
  ready for sampling

# Examples
```jldoctest
# Basic initialization with default settings
julia> sample, state = step(rng, model_wrapper, sampler)

# Custom number of chains with memory disabled
julia> sample, state = step(rng, model_wrapper, sampler; n_chains=10, memory=false)

# With custom initial positions
julia> init_pos = [randn(5) for _ in 1:8]
julia> sample, state = step(rng, model_wrapper, sampler; initial_position=init_pos)
```

# Notes
- For non-memory samplers, `n_chains` should typically be ≥ dimension for good mixing
- Memory-based samplers can work effectively with fewer chains than the problem dimension
- The function handles dimension mismatches and provides informative warnings
- Initial log-densities are computed automatically for all starting positions

See also [`AbstractMCMC.sample`](@ref), [`deMC`](@ref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function step(
    rng::AbstractRNG,
    model_wrapper::LogDensityModel,
    sampler::AbstractDifferentialEvolutionSampler;
    n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
    memory::Bool = true,
    N₀::Int = 2 * n_chains,
    adapt::Bool = true,
    initial_position::Union{Nothing, AbstractVector{<:AbstractVector{<:Real}}} = nothing,
    parallel::Bool = false,
    kwargs...
)
    model = model_wrapper.logdensity

    if adapt
        adaptive_state = initialize_adaptive_state(sampler, model_wrapper, n_chains)
    else
        adaptive_state = DifferentialEvolutionAdaptiveStatic{Float64}()
    end

    extra_memory = nothing

    if isnothing(initial_position)
        x = [randn(rng, dimension(model)) for _ in 1:n_chains]
    else
        current_N = size(initial_position, 1)
        current_pars = size(initial_position[1], 1)
        if current_pars != dimension(model)
            error("Number of parameters in initial position must be equal to the number of parameters in the log density")
        end
        if current_N == n_chains
            x = initial_position
        elseif current_N < n_chains
            @warn "Initial position is smaller than the requested (or required) n_chains. Expanding initial position."
            x = cat(
                [randn(rng, eltype(initial_position[1]), current_pars) for _ in 1:(n_chains - current_N)],
                initial_position,
                dims = 1
            )
        elseif memory
            @warn "Initial position is larger than requested number of chains. Shrinking initial position appending the rest to initial memory."
            #shrink initial position
            x = initial_position[1:n_chains]
            extra_memory = initial_position[(n_chains + 1):end]
        else
            #assume n_chains is wrong
            n_chains = current_N
            x = initial_position
        end
    end

    if length(x) < dimension(model) && !memory
        @warn "Number of chains should be greater than or equal to the number of parameters"
    end

    if parallel
        ld = Vector{eltype(x[1])}(undef, length(x))
        Threads.@threads for i in eachindex(x)
            ld[i] = logdensity(model, x[i])
        end
    else
        ld = [logdensity(model, xi) for xi in x]
    end

    if memory
        mem_x = copy(x)
        if !isnothing(extra_memory)
            append!(mem_x, extra_memory)
        end
        if n_chains < N₀
            for _ in 1:(N₀ - length(mem_x))
                push!(mem_x, randn(rng, eltype(x[1]), dimension(model)))
            end
        elseif n_chains > N₀
            @warn "Initial memory size greater than N₀, truncating memory."
            mem_x = mem_x[end-N₀+1:end]
        end
        state = DifferentialEvolutionStateMemory(x, ld, mem_x, adaptive_state)
    else
        state = DifferentialEvolutionState(x, ld, adaptive_state)
    end

    return DifferentialEvolutionSample(x, ld), state
end
