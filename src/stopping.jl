abstract type stopping_criteria_struct end

struct R̂_stopping_criteria <: stopping_criteria_struct
    maximum_R̂::Float64
end

function R̂_stopping_criteria()
    return R̂_stopping_criteria(1.2)
end

function stop_sampling(stopping_criteria::R̂_stopping_criteria, chains::chains_struct, sample_from, last_iteration)
    #check the last half of the sampling iterations
    rhat = MCMCDiagnosticTools.rhat(
        population_to_samples(chains, get_sampling_indices(sample_from, last_iteration))
    )
    println("Rhat: ", round.(rhat, sigdigits = 3))
    if all(rhat .< stopping_criteria.maximum_R̂)
        return true
    else
        return false
    end
end

