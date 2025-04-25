function setup_rngs(rng, n_chains)
    [Random.MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains]
end

function preallocate_proposals(n_chains, initial_state)
    xₚs = Vector{Vector{eltype(initial_state)}}(undef, n_chains);
    for i in 1:n_chains
        xₚs[i] = Vector{eltype(initial_state)}(undef, size(initial_state, 2));
    end
    xₚs
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