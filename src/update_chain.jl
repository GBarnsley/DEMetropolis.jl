function update_chains!(ld::TransformedLogDensity, rngs::Vector{<:AbstractRNG},
        chains::chains_struct, sampler_scheme::sampler_scheme_struct)
    #update the chains
    for chain in 1:(chains.n_chains)
        update!(get_update(sampler_scheme, rngs[chain]), chains, ld, rngs[chain], chain)
    end
    update_position!(chains)
    adapt_samplers!(sampler_scheme, chains)
end

function update_chains_parallel!(ld::TransformedLogDensity, rngs::Vector{<:AbstractRNG},
        chains::chains_struct, sampler_scheme::sampler_scheme_struct)
    #update the chains
    Threads.@threads for chain in 1:(chains.n_chains)
        update!(get_update(sampler_scheme, rngs[chain]), chains, ld, rngs[chain], chain)
    end
    update_position!(chains)
    adapt_samplers!(sampler_scheme, chains)
end

function get_update_chains_func(parallel::Bool)
    if parallel
        return update_chains_parallel!
    else
        return update_chains!
    end
end

function epoch!(iterations::UnitRange{Int}, rngs::Vector{<:AbstractRNG},
        chains::chains_struct, ld::TransformedLogDensity,
        sampler_scheme::sampler_scheme_struct, update_chains_func!::Function, desc::String)
    p = Progress(length(iterations); dt = 1.0, desc = desc)
    for i in iterations
        update_chains_func!(ld, rngs, chains, sampler_scheme)
        next!(p)
    end
    finish!(p)
    nothing
end
