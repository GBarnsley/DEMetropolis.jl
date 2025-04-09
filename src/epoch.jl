
function partition_integer(I::Int, n::Int)
    base = I ÷ n  # Base size of each group
    remainder = I % n  # Remaining units to distribute

    # Create n groups: first 'remainder' groups get (base + 1), the rest get 'base'
    return vcat(fill(base + 1, remainder), fill(base, n - remainder))
end

function partition_its_over_epochs(I::Int, n::Int, current_it, memory)
    values = partition_integer(I, n)
    if memory
        its = cumsum(vcat(current_it, values))
        [its[i]:(its[i + 1] - 1) for i in 1:length(its) - 1]
    else
        map(v -> 1:v, values)
    end
end

function evolution_epoch!(X, X_ld, its, update_chains_func, chains, update_chain_func, tuning_pars, rngs, ld, desc)
    p = ProgressMeter.Progress(length(its); dt = 1.0, desc = desc)
    for it in its
        update_chains_func(X, X_ld, it, chains, update_chain_func, tuning_pars, rngs, ld);
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
end
