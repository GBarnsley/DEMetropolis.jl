#code for sample() outputs into something useful, see MCMCChains

function samples_to_array(samples::Vector{DifferentialEvolutionSample{V, VV}}) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}}
    n_draws = length(samples)
    n_chains = length(samples[1].x)
    n_params = length(samples[1].x[1])

    # Pre-allocate the 3D array
    result = Array{T, 3}(undef, n_draws, n_chains, n_params)

    @inbounds for (i, sample) in enumerate(samples)
        for (j, chain) in enumerate(sample.x)
            for (k, param) in enumerate(chain)
                result[i, j, k] = param
            end
        end
    end

    return result
end

function ld_to_array(samples::Vector{DifferentialEvolutionSample{V, VV}}) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}}
    n_draws = length(samples)
    n_chains = length(samples[1].ld)

    # Pre-allocate the 3D array
    result = Array{T, 2}(undef, n_draws, n_chains)

    @inbounds for (i, sample) in enumerate(samples)
        for (j, ld) in enumerate(sample.ld)
            result[i, j] = ld
        end
    end

    return result
end

"""
    process_outputs(samples; save_burnt=false, n_burnin=0)

Process raw differential evolution sampling output into a structured format.

This internal function converts the vector of DifferentialEvolutionSample objects
returned by AbstractMCMC.sample into organized arrays suitable for analysis.
It handles separation of burn-in and post-burn-in samples when requested.

# Arguments
- `samples`: Vector of DifferentialEvolutionSample objects from sampling

# Keyword Arguments
- `save_burnt`: Whether to include burn-in samples in the output. Defaults to `false`.
- `n_burnin`: Number of burn-in iterations to separate. Defaults to 0.

# Returns
- Named tuple containing:
  - `samples`: 3D array (iterations, chains, parameters) of post-burn-in samples
  - `ld`: 2D array (iterations, chains) of post-burn-in log-densities
  - `burnt_samples`: 3D array of burn-in samples (only if `save_burnt=true`)
  - `burnt_ld`: 2D array of burn-in log-densities (only if `save_burnt=true`)

# Example
```julia
# Internal usage in template functions
result = process_outputs(raw_samples; save_burnt=true, n_burnin=5000)
```

See also [`samples_to_array`](@ref), [`ld_to_array`](@ref).
"""
function process_outputs(
    samples::Vector{DifferentialEvolutionSample{V, VV}};
    save_burnt::Bool = false,
    n_burnin::Int = 0
    ) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}}
    ld = ld_to_array(samples)
    x = samples_to_array(samples)
    if save_burnt
        return (
            samples = x[(n_burnin + 1):end, :, :],
            ld = ld[(n_burnin + 1):end, :],
            burnt_samples = x[1:n_burnin, :, :],
            burnt_ld = ld[1:n_burnin, :]
        )
    else
        return (
            samples = x,
            ld = ld
        )
    end
end
