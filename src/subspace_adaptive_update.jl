struct DifferentialEvolutionAdaptiveSubspace{T <: Real} <:
       AbstractDifferentialEvolutionAdaptiveState{T}
    "attempts for each crossover probability"
    L::Vector{Int}
    "squared normalised jumping distance for each crossover probability for each crossover probability"
    Δ::Vector{T}
    "distribution for crossover probabilities"
    cr_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
    "running count for variance calculation"
    var_count::Int
    "running mean for each dimension"
    var_mean::Vector{T}
    "running M2 for variance calculation (Welford's algorithm)"
    var_m2::Vector{T}
end

# Helper function to update running variance using Welford's algorithm
function calculate_running_variance(adaptive_state::DifferentialEvolutionAdaptiveSubspace,
        new_values::Vector{V}) where {T <: Real, V <: Vector{T}}
    new_count = copy(adaptive_state.var_count)
    new_mean = copy(adaptive_state.var_mean)
    new_m2 = copy(adaptive_state.var_m2)
    @inbounds for new_value in new_values
        new_count += 1
        delta = new_value .- new_mean
        new_mean .+= delta ./ new_count
        delta2 = new_value .- new_mean
        new_m2 .+= delta .* delta2
    end
    return new_count, new_mean, new_m2
end

# Helper function to get current variance
function get_current_variance(adaptive_state::DifferentialEvolutionAdaptiveSubspace{T}) where {T <:
                                                                                               Real}
    if adaptive_state.var_count < 2
        return ones(T, length(adaptive_state.var_m2))  # Use 1.0 as default when insufficient data
    else
        return adaptive_state.var_m2 ./ adaptive_state.var_count
    end
end

#update the sampler with the adapted cr

function fix_sampler(sampler::DifferentialEvolutionSubspaceSampler,
        adaptive_state::DifferentialEvolutionAdaptiveSubspace)
    DifferentialEvolutionSubspaceSampler(
        adaptive_state.cr_spl,
        sampler.n_cr,
        sampler.δ_spl,
        sampler.ϵ_spl,
        sampler.e_spl
    )
end

function fix_sampler(sampler::DifferentialEvolutionSubspaceSamplerFixedGamma,
        adaptive_state::DifferentialEvolutionAdaptiveSubspace)
    DifferentialEvolutionSubspaceSamplerFixedGamma(
        adaptive_state.cr_spl,
        sampler.n_cr,
        sampler.δ_spl,
        sampler.ϵ_spl,
        sampler.e_spl,
        sampler.γ
    )
end

"""
    step_warmup(rng, model_wrapper, sampler, state; parallel=false, kwargs...)

Perform a single MCMC step during the warm-up (adaptive) phase.

During warm-up, this function performs the same sampling as [`step`](@ref) but also
updates adaptive parameters. For subspace samplers, it adapts crossover probabilities
based on the effectiveness of different parameter subsets.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Adaptive differential evolution sampler
- `state`: Current state including adaptive parameters

# Keyword Arguments
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers).
  Defaults to `true`. Useful if memory has grown too large.
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`.
- `kwargs...`: Additional keyword arguments passed to update functions

# Returns
- `sample`: DifferentialEvolutionSample containing new positions and log-densities
- `new_state`: Updated state with adapted parameters for the next iteration

# Example
```@example step_warmup
using DEMetropolis, Random, Distributions

# Setup for warmup step example
rng = Random.default_rng()
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)
sampler = DREAMz()

# Initialize state (this would typically be done by AbstractMCMC.sample)
# sample, new_state = step_warmup(rng, model_wrapper, sampler, state; parallel=false)
```

See also [`step`](@ref), [`fix_sampler`](@ref).
"""
function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDifferentialEvolutionSubspaceSampler,
        state::AbstractDifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveSubspace{T}};
        update_memory::Bool = true,
        parallel::Bool = false,
        kwargs...
) where {T <: Real}

    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current state
    x = state.x
    ld = state.ld
    adaptive_state = state.adaptive_state

    variance = get_current_variance(adaptive_state)

    L = copy(adaptive_state.L)
    Δ = copy(adaptive_state.Δ)

    # loop through chains running the update
    xₚ = similar(x)
    ldₚ = similar(ld)
    if parallel
        # thread safe updating
        Δ_update = zeros(T, length(x))
        cr_update = Vector{Int}(undef, length(x))
        rngs = [Random.seed!(copy(rng), rand(rng, UInt)) for i in eachindex(x)]

        @inbounds Threads.@threads for i in eachindex(x)
            prop = proposal(rngs[i], fix_sampler(sampler, adaptive_state), state, i)
            xₚ[i] = prop.xₚ
            accepted = update_chain!(model, rngs[i], xₚ, ldₚ, x, ld, prop.offset, i,
                get_temperature(state.temperature_ladder, i))
            cr_update[i] = prop.cr
            if accepted
                Δ_update[i] += sum(
                    (x[i] .- xₚ[i]) .* (x[i] .- xₚ[i]) ./ variance
                )
            end
        end
        for i in eachindex(x)
            L[cr_update[i]] += 1
            Δ[cr_update[i]] += Δ_update[i]
        end
    else
        @inbounds for i in eachindex(x)
            chain_rng = Random.seed!(copy(rng), rand(rng, UInt)) #so its identical to parallel
            prop = proposal(chain_rng, fix_sampler(sampler, adaptive_state), state, i)
            xₚ[i] = prop.xₚ
            accepted = update_chain!(model, chain_rng, xₚ, ldₚ, x, ld, prop.offset,
                i, get_temperature(state.temperature_ladder, i))

            L[prop.cr] += 1
            if accepted
                Δ[prop.cr] += sum(
                    (x[i] .- xₚ[i]) .* (x[i] .- xₚ[i]) ./ variance
                )
            end
        end
    end

    #update variance
    var_count, var_mean, var_m2 = calculate_running_variance(adaptive_state, xₚ)
    if (sum(L .== 0) == 0) && (sum(Δ .== 0) == 0)
        p = sum(L) .* (Δ ./ L) ./ sum(Δ)
        # correct (need to check this is right)
        p ./= sum(p)
        cr_spl = Distributions.sampler(Categorical(p))
    else
        cr_spl = adaptive_state.cr_spl
    end

    return create_sample(xₚ, ldₚ, state),
    update_state(
        state;
        adaptive_state = DifferentialEvolutionAdaptiveSubspace{T}(
            L, Δ, cr_spl, var_count, var_mean, var_m2),
        update_memory = update_memory,
        x = xₚ, ld = ldₚ
    )
end

function initialize_adaptive_state(sampler::AbstractDifferentialEvolutionSubspaceSampler,
        model_wrapper::LogDensityModel, n_chains::Int)
    n_cr = sampler.n_cr
    T = Float64
    d = dimension(model_wrapper.logdensity)
    if n_cr == 0
        @warn "sampler already has a fixed crossover probability, cannot adapt."
        return DifferentialEvolutionAdaptiveStatic{T}()
    elseif n_cr == 1
        @warn "Only one crossover probability, cannot adapt."
        return DifferentialEvolutionAdaptiveStatic{T}()
    else
        L = zeros(Int, n_cr)
        Δ = zeros(T, n_cr)
        cr_spl = Distributions.sampler(Categorical(repeat([1 / n_cr], n_cr)))
        # Initialize running variance tracking
        var_count = 0
        var_mean = zeros(T, d)
        var_m2 = zeros(T, d)
        return DifferentialEvolutionAdaptiveSubspace{T}(
            L, Δ, cr_spl, var_count, var_mean, var_m2)
    end
end
