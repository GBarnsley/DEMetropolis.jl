#code for sample() outputs into something useful, see MCMCChains

function samples_to_array(samples::Vector{DifferentialEvolutionSample{
        V, VV}}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
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

function ld_to_array(samples::Vector{DifferentialEvolutionSample{
        V, VV}}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
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

function process_outputs(
        samples::Vector{DifferentialEvolutionSample{
        V, VV}}
) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return DifferentialEvolutionOutput{T}(
        samples_to_array(samples),
        ld_to_array(samples)
    )
end

#code for post-processing outputs from sample()
function bundle_samples(
        samples::Vector{DifferentialEvolutionSample{V, VV}},
        model_wrapper::LogDensityModel,
        sampler::AbstractDifferentialEvolutionSampler,
        state::AbstractDifferentialEvolutionState,
        ::Type{T};
        save_final_state::Bool = false,
        kwargs...
) where {T, T2 <: Real, V <: AbstractVector{T2}, VV <: AbstractVector{V}}
    samples_ = convert(T, samples)
    if save_final_state
        return (
            samples_,
            state
        )
    else
        return samples_
    end
end

function AbstractMCMC.chainsstack(chns::Vector{DifferentialEvolutionOutput{T}}) where {T <:
                                                                                       Real}
    DifferentialEvolutionOutput{T}(
        cat([c.samples for c in chns]...; dims = 2),
        cat((c.ld for c in chns)...; dims = 2)
    )
end

function AbstractMCMC.chainsstack(
        chns::Vector{Tuple{DifferentialEvolutionOutput{T},
        E}}
) where {T <: Real, E <: AbstractDifferentialEvolutionState}
    (
        AbstractMCMC.chainsstack([c[1] for c in chns]),
        [c[2] for c in chns]
    )
end

function AbstractMCMC.chainsstack(
        chns::Vector{Tuple{
        C, E}}
) where {C <: Chains, E <: AbstractDifferentialEvolutionState}
    (
        AbstractMCMC.chainsstack([c[1] for c in chns]),
        [c[2] for c in chns]
    )
end

function convert(
        ::Type{T},
        samples::Vector{DifferentialEvolutionSample{V, VV}}
) where {T, T2 <: Real, V <: AbstractVector{T2}, VV <: AbstractVector{V}}
    return samples
end

function convert(
        ::Type{DifferentialEvolutionOutput},
        samples::Vector{DifferentialEvolutionSample{V, VV}}
) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return process_outputs(samples)
end

function convert(
        ::Type{Chains},
        samples::Vector{DifferentialEvolutionSample{V, VV}}
) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    output = process_outputs(samples)

    array_out = cat(
        permutedims(output.samples, (1, 3, 2)),
        insertdims(output.ld, dims = 2), dims = 2
    )

    chns = Chains(array_out)
    chns = replacenames(chns, "param_$(size(output.samples, 3) + 1)" => "ld")

    return chns
end
