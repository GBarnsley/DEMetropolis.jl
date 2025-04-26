function setup_rngs(rng, n_chains)
    [Random.MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains]
end

function population_to_samples(chains::chains_struct, its, n_chains, N₀)
    samples = Array{eltype(chains.X)}(undef, length(its), n_chains, size(chains.X, 2));
    for i in eachindex(its), j in 1:n_chains;
        samples[i, j, :] = chains.X[j + ((its[i] - 1) * n_chains) + N₀, :]
    end
    return samples
end

function ld_to_samples(chains::chains_struct, its, n_chains, N₀)
    lds = Array{eltype(chains.X)}(undef, length(its), n_chains);
    for i in eachindex(its), j in 1:n_chains;
        lds[i, j] = chains.ld[j + ((its[i] - 1) * n_chains) + N₀]
    end
    return lds
end

function format_output(chains::chains_struct, n_chains, N₀, sample_indices)
    return (
        samples = population_to_samples(chains, sample_indices, n_chains, N₀),
        ld = ld_to_samples(chains, sample_indices, n_chains, N₀)
    )
end

function format_output(chains::chains_struct, n_chains, N₀, sample_indices, burnt_indices)
    return (
        samples = population_to_samples(chains, sample_indices, n_chains, N₀),
        ld = ld_to_samples(chains, sample_indices, n_chains, N₀),
        burnt_samples = population_to_samples(chains, burnt_indices, n_chains, N₀),
        burnt_ld = ld_to_samples(chains, burnt_indices, n_chains, N₀)
    )
end