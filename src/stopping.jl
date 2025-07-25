abstract type stopping_criteria_struct end


"""
Create a stopping criterion based on the rank Gelman-Rubin diagnostic (R̂). Sampling stops when the R̂ value for all parameters is below `maximum_R̂`.

# Arguments
- `maximum_R̂`: The maximum acceptable R̂ value. Defaults to 1.2.

See also [`MCMCDiagnosticTools.rhat`](@extref).
"""
struct R̂_stopping_criteria <: stopping_criteria_struct
    maximum_R̂::Float64
end

function R̂_stopping_criteria()
    return R̂_stopping_criteria(1.2)
end

function stop_sampling(stopping_criteria::R̂_stopping_criteria, chains::chains_struct, sample_from, last_iteration)
    #check the last half of the sampling iterations
    rhat_ = rhat(
        population_to_samples(chains, get_sampling_indices(sample_from, last_iteration))
    )
    println("Rhat: ", round.(rhat_, sigdigits = 3))
    if all(rhat_ .< stopping_criteria.maximum_R̂)
        return true
    else
        return false
    end
end

