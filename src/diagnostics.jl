abstract type diagnostic_check_struct end

"""
Create a diagnostic check that identifies outlier chains based on their mean log-density (of the last 50% of the chain) during burn-in/warm-up.
Outlier chains (those with mean log-density below Q1 - 2*IQR) are reset to the state of the chain with the highest mean log-density.

See: Vrugt 2009 doi.org/10.1515/IJNSNS.2009.10.3.273

See also [`acceptance_check`](@ref).
"""
struct ld_check <: diagnostic_check_struct
end

function run_diagnostic_check!(chains::chains_struct, diagnostic_check::ld_check, rngs::Vector{R}, current_iteration::Int) where R <: AbstractRNG

    #calculate IQR of log densities of the last 50%
    ld_means = mean(
        ld_to_samples(chains, get_sampling_indices(1, current_iteration)), dims = 1)[1, :]
    q₁ = quantile(ld_means, 0.25)

    outliers = findall(ld_means .< q₁ - 2 * (quantile(ld_means, 0.75) - q₁))

    if length(outliers) > 0
        @warn string(length(outliers)) * " outlier chains detected, setting to best chain"
        #remove outliers
        best_chain = argmax(ld_means)

        chains.ld[chains.current_position[outliers], :] .= chains.ld[
            chains.current_position[[best_chain]], :]
        chains.X[chains.current_position[outliers], :] .= chains.X[
            chains.current_position[[best_chain]], :]
    end
end

"""
Create a diagnostic check that identifies poorly mixing chains based on their acceptance rate (of the last 50% of the chain) during burn-in/warm-up.
Chains with acceptance rates below `min_acceptance` and significantly lower than others (based on log-acceptance rate IQR) are reset to the state of the chain closest to the `target_acceptance` rate.

# Arguments
- `min_acceptance`: The minimum acceptable acceptance rate. Defaults to 0.1.
- `target_acceptance`: The target acceptance rate used to select the best chain for resetting outliers. Defaults to 0.24.

See also [`ld_check`](@ref).
"""
struct acceptance_check <: diagnostic_check_struct
    min_acceptance::Float64
    target_acceptance::Float64
end

acceptance_check() = acceptance_check(0.1, 0.24)

function run_diagnostic_check!(
        chains::chains_struct, diagnostic_check::acceptance_check, rngs::Vector{R}, current_iteration::Int) where R <: AbstractRNG

    #calculate average acceptance
    X = population_to_samples(chains, get_sampling_indices(1, current_iteration))
    p_acceptance = sum(
        sum((X[2:end, :, :] .- X[1:(end - 1), :, :]) .!= 0, dims = (3))[:, :, 1] .> 0,
        dims = 1)[1, :] ./
                   (current_iteration - 1)

    #IQR on a log level to be more sensitive
    lp_acceptance = log.(p_acceptance)

    q₁ = quantile(lp_acceptance, 0.25)
    outliers = findall((lp_acceptance .< (q₁ - 2 * (quantile(lp_acceptance, 0.75) - q₁))) .&
                       (p_acceptance .< diagnostic_check.min_acceptance))

    if length(outliers) > 0
        @warn string(length(outliers)) *
              " poorly mixing chain chains detected, setting to best chain"
        #remove outliers
        best_chain = [argmin(abs.(p_acceptance .- diagnostic_check.target_acceptance))]
        chains.ld[chains.current_position[outliers], :] .= chains.ld[
            chains.current_position[best_chain], :]
        chains.X[chains.current_position[outliers], :] .= chains.X[
            chains.current_position[best_chain], :]
    end
end
