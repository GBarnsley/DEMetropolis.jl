#uses preallocated for anything less than 4 chains (which the most default methods ask for)
function fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int},
        current_chain::Int
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    if n_chains ≤ length(indices)
        return _fast_sample_chains!(
            rng,
            x,
            max_length,
            n_chains,
            indices,
            ordered_indices,
            current_chain
        )
    else
        @warn "Picking $n_chains chains but only $(length(indices)) preallocated, consider setting `n_preallocated_indices = $n_chains`." maxlog = 1
        return _fast_sample_chains!(
            rng,
            x,
            max_length,
            n_chains,
            Vector{Int}(undef, n_chains),
            Vector{Int}(undef, n_chains),
            current_chain
        )
    end
end

function _fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int},
        current_chain::Int
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    ordered_indices[1] = current_chain
    for i in 1:(n_chains - 1)
        idx = rand(rng, 1:(max_length - i))
        for j in 1:i
            if ordered_indices[j] ≤ idx
                idx += 1
            end
        end
        ordered_indices[i + 1] = idx
        sort!(view(ordered_indices, 1:(i + 1)))
        indices[i] = idx
    end
    idx = rand(rng, 1:(max_length - n_chains))
    for i in 1:n_chains
        if ordered_indices[i] ≤ idx
            idx += 1
        end
    end
    indices[n_chains] = idx
    return view(x, view(indices, 1:n_chains))
end

#uses preallocated for anything less than 4 chains (which the most default methods ask for)
function fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int}
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    if n_chains ≤ length(indices)
        return _fast_sample_chains!(
            rng,
            x,
            max_length,
            n_chains,
            indices,
            ordered_indices
        )
    else
        @warn "Picking $n_chains chains but only $(length(indices)) preallocated, consider setting `n_preallocated_indices = $n_chains`." maxlog = 1
        return _fast_sample_chains!(
            rng,
            x,
            max_length,
            n_chains,
            Vector{Int}(undef, n_chains),
            Vector{Int}(undef, n_chains - 1)
        )
    end
end

function _fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int}
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    indices[1] = rand(rng, 1:max_length)
    ordered_indices[1] = indices[1]
    for i in 2:(n_chains - 1)
        idx = rand(rng, 1:(max_length - i + 1))
        for j in 1:(i - 1)
            if ordered_indices[j] ≤ idx
                idx += 1
            end
        end
        ordered_indices[i] = idx
        sort!(view(ordered_indices, 1:i))
        indices[i] = idx
    end
    idx = rand(rng, 1:(max_length - n_chains + 1))
    for i in 1:(n_chains - 1)
        if ordered_indices[i] ≤ idx
            idx += 1
        end
    end
    indices[n_chains] = idx
    return view(x, view(indices, 1:n_chains))
end
