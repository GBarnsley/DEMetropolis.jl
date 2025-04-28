abstract type chains_struct end

function setup_population(ld, initial_state, total_iterations, N₀, n_pars, n_chains, memory, parallel)
    X = Array{eltype(initial_state)}(undef, total_iterations + N₀, n_pars);
    X_ld = Array{eltype(initial_state)}(undef, total_iterations + N₀);
    X[1:N₀, :] .= initial_state;
    if parallel
        Threads.@threads for i in 1:N₀
            X_ld[i] = LogDensityProblems.logdensity(ld, initial_state[i, :]);
        end
    else
        for i in 1:N₀
            X_ld[i] = LogDensityProblems.logdensity(ld, initial_state[i, :]);
        end
    end
    current_position = collect((N₀ - n_chains + 1):(N₀));
    if memory
        return chains_memory(
            X, X_ld, current_position, n_chains, N₀, true
        )
    else
        other_chains = map(c -> setdiff(1:n_chains, c), 1:n_chains);
        return chains_memoryless(
            X, X_ld, other_chains, current_position, n_chains, N₀, true
        )
    end
end

mutable struct chains_memoryless{X_type <: Real} <: chains_struct
    X::Array{X_type, 2}
    ld::Array{X_type, 1}
    other_chains::Vector{Vector{Int}}
    current_position::Array{Int, 1}
    n_chains::Int
    N₀::Int
    warmup::Bool
end

function get_value(chains::chains_struct, chain)
    return chains.X[chains.current_position[chain], :]
end

function update_value!(chains::chains_struct, rng, chain, xₚ, ldₚ)
    old_position = chains.current_position[chain];
    new_position = old_position + chains.n_chains;
    if log(rand(rng)) < ldₚ - chains.ld[old_position]
        chains.X[new_position, :] .= xₚ
        chains.ld[new_position] = ldₚ
        return true
    else
        chains.X[new_position, :] .= chains.X[old_position, :]
        chains.ld[new_position] = chains.ld[old_position]
        return false
    end
end

function update_value!(chains::chains_struct, rng, chain, xₚ, ldₚ, offset)
    old_position = chains.current_position[chain];
    new_position = old_position + chains.n_chains;
    if log(rand(rng)) < ldₚ - chains.ld[old_position] + offset
        chains.X[new_position, :] .= xₚ
        chains.ld[new_position] = ldₚ
        return true
    else
        chains.X[new_position, :] .= chains.X[old_position, :]
        chains.ld[new_position] = chains.ld[old_position]
        return false
    end
end

function update_position!(chains::chains_struct)
    # Update the current position of the chains
    chains.current_position .+= chains.n_chains
end

function sample_chains(chains::chains_memoryless, rng, chain, n_samples)
    # Sample from the chains
    chains.current_position[StatsBase.sample(rng, chains.other_chains[chain], n_samples, replace = false)]
end

mutable struct chains_memory{X_type <: Real} <: chains_struct
    X::Array{X_type, 2}
    ld::Array{X_type, 1}
    current_position::Array{Int, 1}
    n_chains::Int
    N₀::Int
    warmup::Bool
end

function sample_chains(chains::chains_memory, rng, chain, n_samples)
    indices = StatsBase.sample(rng, 1:(chains.current_position[end] - 1), n_samples, replace = false)
    indices[indices .>= chains.current_position[chain]] .+= 1 #so we don't sample the current chain
    indices
end

function resize_chains!(chains::chains_struct, new_size)
    #not great have to re allocate the whole array
    new_X = Array{eltype(chains.X)}(undef, new_size, size(chains.X, 2));
    new_X[axes(chains.X, 1), :] .= chains.X;
    new_ld = Array{eltype(chains.ld)}(undef, new_size);
    new_ld[axes(chains.ld, 1)] .= chains.ld;

    chains.X = new_X;
    chains.ld = new_ld;
end