"""
    r̂_stopping_criteria(rng, model, sampler, samples, state, iteration; kwargs...)

Stopping criterion based on the Gelman-Rubin diagnostic (R̂).

Sampling continues until the R̂ value for all parameters falls below `maximum_R̂`,
indicating convergence across chains. This function is designed to be used as the
`N_or_isdone` argument in `AbstractMCMC.sample` for adaptive stopping.

The diagnostic is computed on the last half of the collected samples to focus on
the stationary portion of the chains.

# Arguments
- `rng`: Random number generator (unused but required by AbstractMCMC interface)
- `model`: The model being sampled (unused but required by interface)
- `sampler`: The differential evolution sampler (unused but required by interface)
- `samples`: Vector of collected samples from all chains
- `state`: Current sampler state (unused but required by interface)
- `iteration`: Current iteration number

# Keyword Arguments
- `check_every`: Frequency (in iterations) for checking R̂ values. Defaults to 1000.
- `maximum_R̂`: Maximum acceptable R̂ value for convergence. Defaults to 1.2.
- `maximum_iterations`: Maximum number of iterations before forced stopping. Defaults to 100000.
- `minimum_iterations`: Minimum iterations before convergence checking begins. Defaults to 0.

# Returns
- `true` if sampling should stop (converged or maximum iterations reached)
- `false` if sampling should continue

# Example
```@example convergence
using DEMetropolis, AbstractMCMC, Random, Distributions

# Create a simple model
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Setup sampler 
sampler = deMCzs()

# Use with adaptive stopping criterion
rng = Random.default_rng()
chains = sample(rng, model_wrapper, sampler, r̂_stopping_criteria; 
               n_chains=4, check_every=500, maximum_R̂=1.1)
```

See also [`MCMCDiagnosticTools.rhat`](@extref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function r̂_stopping_criteria(
        rng::AbstractRNG,
        model::AbstractModel,
        sampler::AbstractDifferentialEvolutionSampler,
        samples::Vector{DifferentialEvolutionSample{V, VV}},
        state::AbstractDifferentialEvolutionState{T, A, L, V, VV},
        iteration::Int;
        check_every::Int = 1000,
        maximum_R̂::T = 1.2,
        maximum_iterations::Int = 100000,
        minimum_iterations::Int = 0,
        kwargs...
) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T}}
    if iteration % check_every != 0 || iteration < minimum_iterations
        return false
    elseif iteration >= maximum_iterations
        println("Maximum iterations reached: ", maximum_iterations)
        return true
    else
        #check the last half of the sampling iterations
        rhat_ = rhat(samples_to_array(samples[(iteration ÷ 2 + 1):end]))
        println("Rhat: ", round.(rhat_, sigdigits = 3))
        if all(rhat_ .< maximum_R̂)
            return true
        else
            return false
        end
    end
end
