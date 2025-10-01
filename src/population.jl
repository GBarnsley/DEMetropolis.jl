abstract type chains_struct{T<:Real} end

function always_store(samples::Int)
    return true
end

function is_multiple_of(a::Int)
    func(x::Int) = x % a == 0
    return func
end

function setup_population(
        ld::TransformedLogDensity, initial_state::Array{T, 2}, total_iterations::Int, N₀::Int, n_pars::Int, n_chains::Int, memory::Bool, parallel::Bool, thin::Int) where T <: Real
    X = Array{T}(undef, total_iterations + N₀, n_pars)
    X_ld = Array{T}(undef, total_iterations + N₀)
    X[1:N₀, :] .= initial_state
    if parallel
        Threads.@threads for i in 1:N₀
            X_ld[i] = logdensity(ld, initial_state[i, :])
        end
    else
        for i in 1:N₀
            X_ld[i] = logdensity(ld, initial_state[i, :])
        end
    end
    current_position = collect((N₀ - n_chains + 1):(N₀))
    samples = 1

    if thin == 1
        store_sample = always_store
    else
        store_sample = is_multiple_of(thin)
    end

    if memory
        return chains_memory(
            X, X_ld, current_position, n_chains, N₀, true, samples, store_sample
        )
    else
        other_chains = map(c -> setdiff(1:n_chains, c), 1:n_chains)
        return chains_memoryless(
            X, X_ld, other_chains, current_position,
            n_chains, N₀, true, samples, store_sample
        )
    end
end

mutable struct chains_memoryless{T} <: chains_struct{T}
    X::Array{T, 2}
    ld::Array{T, 1}
    other_chains::Vector{Vector{Int}}
    current_position::Array{Int, 1}
    n_chains::Int
    N₀::Int
    warmup::Bool
    samples::Int
    store_sample::Function
end

function get_value(chains::chains_struct, chain::Int)
    return chains.X[chains.current_position[chain], :]
end

function update_value!(chains::chains_struct{T}, rng::AbstractRNG, chain::Int, xₚ::Array{T, 1}, ldₚ::T) where T <: Real
    old_position = chains.current_position[chain]
    new_position = old_position + chains.n_chains
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

function update_value!(chains::chains_struct{T}, rng::AbstractRNG, chain::Int, xₚ::Array{T, 1}, ldₚ::T, offset::T) where T <: Real
    old_position = chains.current_position[chain]
    new_position = old_position + chains.n_chains
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
    if chains.store_sample(chains.samples)
        # Update the current position of the chains
        chains.current_position .+= chains.n_chains
    else
        # set the current position to the updated value
        chains.X[chains.current_position, :] .= chains.X[
            chains.current_position .+ chains.n_chains, :]
        chains.ld[chains.current_position] .= chains.ld[chains.current_position .+ chains.n_chains]
    end
    chains.samples += 1
end

function sample_chains(chains::chains_memoryless, rng::AbstractRNG, chain::Int, n_samples::Int)
    # Sample from the chains
    chains.current_position[sample(
        rng, chains.other_chains[chain], n_samples, replace = false)]
end

mutable struct chains_memory{T} <: chains_struct{T}
    X::Array{T, 2}
    ld::Array{T, 1}
    current_position::Array{Int, 1}
    n_chains::Int
    N₀::Int
    warmup::Bool
    samples::Int
    store_sample::Function
end

function sample_chains(chains::chains_memory, rng::AbstractRNG, chain::Int, n_samples::Int)
    #sample up to the current position
    indices = sample(rng, 1:(chains.current_position[1] - 1), n_samples, replace = false)
    indices
end

function resize_chains!(chains::chains_struct{T}, new_size::Int) where T <: Real
    #not great have to re allocate the whole array
    new_X = Array{T}(undef, new_size, size(chains.X, 2))
    new_X[axes(chains.X, 1), :] .= chains.X
    new_ld = Array{T}(undef, new_size)
    new_ld[axes(chains.ld, 1)] .= chains.ld

    chains.X = new_X
    chains.ld = new_ld
end
