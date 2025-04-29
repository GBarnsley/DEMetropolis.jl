function halve(x::Integer)
    return Int(ceil(x / 2))
end

function setup_rngs(rng, n_chains)
    [Random.MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains]
end

function population_to_samples(chains::chains_struct, its)
    samples = Array{eltype(chains.X)}(undef, length(its), chains.n_chains, size(chains.X, 2));
    for i in eachindex(its), j in 1:chains.n_chains;
        samples[i, j, :] = chains.X[j + ((its[i] - 1) * chains.n_chains) + chains.N₀, :]
    end
    return samples
end

function ld_to_samples(chains::chains_struct, its)
    lds = Array{eltype(chains.X)}(undef, length(its), chains.n_chains);
    for i in eachindex(its), j in 1:chains.n_chains;
        lds[i, j] = chains.ld[j + ((its[i] - 1) * chains.n_chains) + chains.N₀]
    end
    return lds
end

function format_output(chains::chains_struct, sampler_scheme, sample_indices)
    return (
        sampler_scheme = sampler_scheme,
        samples = population_to_samples(chains, sample_indices),
        ld = ld_to_samples(chains, sample_indices)
    )
end

function format_output(chains::chains_struct, sampler_scheme, sample_indices, burnt_indices)
    return (
        sampler_scheme = sampler_scheme,
        samples = population_to_samples(chains, sample_indices),
        ld = ld_to_samples(chains, sample_indices),
        burnt_samples = population_to_samples(chains, burnt_indices),
        burnt_ld = ld_to_samples(chains, burnt_indices)
    )
end

function get_sampling_indices(min_it, max_it)
    n_its = halve(max_it - min_it);
    (max_it - n_its + 1):max_it;
end

function partition_integer(I::Int, n::Int)
    base = I ÷ n  # Base size of each group
    remainder = I % n  # Remaining units to distribute

    # Create n groups: first 'remainder' groups get (base + 1), the rest get 'base'
    return vcat(fill(base + 1, remainder), fill(base, n - remainder))
end