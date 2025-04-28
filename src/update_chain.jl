function update_chains!(ld, rngs, chains, sampler_scheme)
    #update the chains
    for chain in 1:(chains.n_chains)
        update!(get_update(sampler_scheme, rngs[chain]), chains, ld, rngs[chain], chain);
    end
    update_position!(chains);
end

function update_chains_parallel!(ld, rngs, chains, sampler_scheme)
    #update the chains
    Threads.@threads for chain in 1:(chains.n_chains)
        update!(get_update(sampler_scheme, rngs[chain]), chains, ld, rngs[chain], chain);
    end
    update_position!(chains);
end

function get_update_chains_func(parallel)
    if parallel
        return update_chains_parallel!
    else
        return update_chains!
    end
end

function epoch!(iterations, rngs, chains, ld, sampler_scheme, update_chains_func!, desc)
    p = ProgressMeter.Progress(length(iterations); dt = 1.0, desc = desc)
    for i in iterations
        update_chains_func!(ld, rngs, chains, sampler_scheme);
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
    nothing
end