struct DifferentialEvolutionSampler <: AbstractDifferentialEvolutionSampler
    γ_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
    β_spl::Sampleable{Univariate, Continuous}
end

"""
Set up a Differential Evolution (DE) update step for MCMC sampling.

Creates a sampler that proposes new states by adding scaled difference vectors between
randomly selected chains plus small noise. This is the core update mechanism from the
original DE-MC algorithm by ter Braak (2006).

See doi.org/10.1007/s11222-006-8769-1 for more information.

# Keyword Arguments
- `γ`: Scaling factor for the difference vector. Can be a `Real` (fixed value), a
  `UnivariateDistribution` (random scaling), or `nothing` (automatic based on `n_dims`).
  Defaults to `nothing`.
- `β`: Distribution for small noise added to proposals. Must be a univariate continuous
  distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `n_dims`: Problem dimension used for automatic `γ` selection. If > 0 and `γ` is `nothing`,
  sets `γ` to the theoretically optimal `2.38 / sqrt(2 * n_dims)`. If ≤ 0, uses
  `Uniform(0.8, 1.2)`. Defaults to 0.

# Returns
- A `DifferentialEvolutionSampler` that can be used with [`setup_sampler_scheme`](@ref) or [`step`](@ref) or [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments).

# Example
```@example de_update
using DEMetropolis, Distributions

# Setup differential evolution update with custom parameters
de_update = setup_de_update(γ = 1.0, β = Normal(0.0, 0.01))
```

See also [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref), [`setup_sampler_scheme`](@ref).
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

function proposal(rng::AbstractRNG, sampler::DifferentialEvolutionSampler,
        state::AbstractDifferentialEvolutionState, current_state::Int)
    # Propose a new position.
    x₁, x₂ = pick_chains(rng, state, current_state, 2)
    if x₁ == x₂
        return (xₚ = x₁, offset = -Inf)
    else
        return (
            xₚ = state.x[current_state] .+ (rand(rng, sampler.γ_spl) .* (x₁ - x₂)) .+
                 rand(rng, sampler.β_spl, length(state.x[current_state])),
            offset = zero(eltype(x₁))
        )
    end
end
