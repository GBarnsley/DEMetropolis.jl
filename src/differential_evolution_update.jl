struct DifferentialEvolutionSampler <: AbstractDifferentialEvolutionSampler
    γ_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
    β_spl::Sampleable{Univariate, Continuous}
end


"""
Set up a Differential Evolution (DE) update step.

See doi.org/10.1007/s11222-006-8769-1 for more information.

# Keyword Arguments
- `γ`: The scaling factor for the difference vector. Can be a `<:Real`, a `UnivariateDistribution`, or `nothing`. If `nothing`, it is set based on `n_dims`. Defaults to nothing.
- `β`: Distribution for the small noise term added to the proposal. Defaults to `Uniform(-1e-4, 1e-4)` must a univariate distribution.
- `n_dims`: If > 0 and `γ` is `nothing`, sets `γ` to the theoretically optimal `2.38 / sqrt(2 * n_dims)`. If ≤ 0, sets `γ` to `Uniform(0.8, 1.2)`. Defaults to 0.

# Example
```jldoctest
julia> setup_de_update(γ = 1.0; β = Normal(0.0, 0.01))
```
See also [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).
"""
function setup_de_update(;
        γ::Union{Nothing, UnivariateDistribution, Real} = nothing,
        β::ContinuousUnivariateDistribution = Uniform(-1e-4, 1e-4),
        n_dims::Int = 0
)
    if isnothing(γ)
        if n_dims > 0
            γ = Dirac(2.38 / sqrt(2 * n_dims))
        else
            γ = Uniform(0.8, 1.2)
        end
    elseif isa(γ, Real)
        γ = Dirac(γ)
    end

    return DifferentialEvolutionSampler(sampler(γ), sampler(β))
end

struct DifferentialEvolutionState{T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}} <: AbstractDifferentialEvolutionState
    "current position"
    x::VV
    "log density at current position"
    ld::V
end

struct DifferentialEvolutionStateMemory{T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}} <: AbstractDifferentialEvolutionState
    "current position"
    x::VV
    "log density at current position"
    ld::V
    "memory of past positions"
    mem_x::VV
end

struct DifferentialEvolutionSample{V<:AbstractVector{<:Real}, VV<:AbstractVector{V}}
    "current position"
    x::VV
end

function pick_chains(rng::AbstractRNG, state::DifferentialEvolutionState, current_chain::Int, n_chains::Int)
    #sample up to the current position
    indices = sample(rng, 1:(length(state.x) - 1), n_chains, replace = false)
    indices[indices .>= current_chain] .+= 1
    return state.x[indices]
end

function pick_chains(rng::AbstractRNG, state::DifferentialEvolutionStateMemory,  current_chain::Int, n_chains::Int)
    #sample up to the current position
    return sample(rng, state.mem_x, n_chains, replace = false)
end

function step(
    rng::AbstractRNG,
    model_wrapper::LogDensityModel,
    sampler::AbstractDifferentialEvolutionSampler,
    state::DifferentialEvolutionState;
    kwargs...
)
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current states
    (; x, ld) = state
    # loop through chains running the update
    xₚ = similar(x)
    ldₚ = similar(ld)

    @inbounds for i in eachindex(x)
        xₚ[i] = proposal(rng, sampler, state, i)
        ldₚ[i] = logdensity(model, xₚ[i])

        if log(rand(rng)) > (ldₚ[i] - ld[i])
            xₚ[i] = x[i]
            ldₚ[i] = ld[i]
        end
    end

    return DifferentialEvolutionSample(xₚ), DifferentialEvolutionState(xₚ, ldₚ)
end

function step(
    rng::AbstractRNG,
    model_wrapper::LogDensityModel,
    sampler::AbstractDifferentialEvolutionSampler,
    state::DifferentialEvolutionStateMemory;
    kwargs...
)
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current states
    (; x, ld, mem_x) = state
    # loop through chains running the update
    xₚ = similar(x)
    ldₚ = similar(ld)

    @inbounds for i in eachindex(x)
        xₚ[i] = proposal(rng, sampler, state, i)
        ldₚ[i] = logdensity(model, xₚ[i])

        if log(rand(rng)) > (ldₚ[i] - ld[i])
            xₚ[i] = x[i]
            ldₚ[i] = ld[i]
        else
            # Add the new position to the memory, probably should just do this regardless
            push!(mem_x, xₚ[i])
        end
    end

    return DifferentialEvolutionSample(xₚ), DifferentialEvolutionStateMemory(xₚ, ldₚ, mem_x)
end

function step(
    rng::AbstractRNG,
    model_wrapper::LogDensityModel,
    sampler::AbstractDifferentialEvolutionSampler;
    n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
    memory::Bool = true,
    N₀::Int = 2 * n_chains,
    kwargs...
)
    model = model_wrapper.logdensity

    x = [randn(rng, dimension(model)) for _ in 1:n_chains]
    ld = [logdensity(model, xi) for xi in x]

    if memory
        mem_x = copy(x)
        if length(mem_x) < N₀
            for _ in 1:(N₀ - length(mem_x))
                push!(mem_x, randn(rng, dimension(model)))
            end
        elseif length(mem_x) > N₀
            @warn "Initial memory size greater than initial memory size, truncating memory."
            mem_x = mem_x[end-N₀+1:end]
        end
        state = DifferentialEvolutionStateMemory(x, ld, mem_x)
    else
        state = DifferentialEvolutionState(x, ld)
    end

    return DifferentialEvolutionSample(x), state
end

function proposal(rng::AbstractRNG, sampler::DifferentialEvolutionSampler, state::AbstractDifferentialEvolutionState, current_state::Int)
    # Propose a new position.
    x₁, x₂ = pick_chains(rng, state, current_state, 2)
    return state.x[current_state] .+ (rand(rng, sampler.γ_spl) .* (x₁ - x₂)) .+ rand(rng, sampler.β_spl, length(state.x[current_state]))
end
