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
            X, X_ld, current_position, n_chains
        )
    else
        other_chains = map(c -> setdiff(1:n_chains, c), 1:n_chains);
        return chains_memoryless(
            X, X_ld, other_chains, current_position, n_chains
        )
    end
end

struct chains_memoryless{X_type <: Real} <: chains_struct
    X::Array{X_type, 2}
    ld::Array{X_type, 1}
    other_chains::Vector{Vector{Int}}
    current_position::Array{Int, 1}
    n_chains::Int
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
    else
        chains.X[new_position, :] .= chains.X[old_position, :]
        chains.ld[new_position] = chains.ld[old_position]
    end
end

function update_value!(chains::chains_struct, rng, chain, xₚ, ldₚ, offset)
    old_position = chains.current_position[chain];
    new_position = old_position + chains.n_chains;
    if log(rand(rng)) < ldₚ - chains.ld[old_position] + offset
        chains.X[new_position, :] .= xₚ
        chains.ld[new_position] = ldₚ
    else
        chains.X[new_position, :] .= chains.X[old_position, :]
        chains.ld[new_position] = chains.ld[old_position]
    end
end

function update_position!(chains::chains_struct)
    # Update the current position of the chains
    chains.current_position .+= n_chains
end

function sample_chains(chains::chains_memoryless, rng, chain, n_samples)
    # Sample from the chains
    chains.current_position[StatsBase.sample(rng, chains.other_chains[chain], n_samples, replace = false)]
end

struct chains_memory{X_type <: Real} <: chains_struct
    X::Array{X_type, 2}
    ld::Array{X_type, 1}
    current_position::Array{Int, 1}
    n_chains::Int
end

function sample_chains(chains::chains_memory, rng, chain, n_samples)
    # Sample from the previous chains
    StatsBase.sample(rng, setdiff(1:(chains.current_position[end]), chains.current_position[chain]), n_samples, replace = false)
end