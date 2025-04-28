abstract type diagnostic_check_struct end

struct ld_check <: diagnostic_check_struct
end

function run_diagnostic_check!(chains, diagnostic_check::ld_check, rngs, current_iteration)

    #calculate IQR of log densities of the last 50%
    ld_means = StatsBase.mean(ld_to_samples(chains, get_sampling_indices(1, current_iteration)), dims = 1)[1, :]
    q₁ = StatsBase.quantile(ld_means, 0.25);
    
    outliers = findall(ld_means .< q₁ - 2 * (StatsBase.quantile(ld_means, 0.75) - q₁));

    if length(outliers) > 0
        @warn string(length(outliers)) * " outlier chains detected, setting to best chain"
        #remove outliers
        best_chain = argmax(ld_means);
        
        chains.ld[chains.current_position[outliers], :] .= chains.ld[chains.current_position[[best_chain]], :];
        chains.X[chains.current_position[outliers], :] .= chains.X[chains.current_position[[best_chain]], :];
    end
end

struct acceptance_check <: diagnostic_check_struct
    min_acceptance::Float64
    target_acceptance::Float64
end

acceptance_check() = acceptance_check(0.1, 0.24)

function run_diagnostic_check!(chains, diagnostic_check::acceptance_check, rngs, current_iteration)

    #calculate average acceptance
    X = population_to_samples(chains, get_sampling_indices(1, current_iteration))
    p_acceptance = sum(sum((X[2:end, :, :] .- X[1:(end - 1), :, :]) .!= 0, dims = (3))[:, :, 1] .> 0, dims = 1)[1, :] ./ 
    (current_iteration - 1)

    #IQR on a log level to be more sensitive
    lp_acceptance = log.(p_acceptance);

    q₁ = StatsBase.quantile(lp_acceptance, 0.25);
    outliers = findall((lp_acceptance .< (q₁ - 2 * (StatsBase.quantile(lp_acceptance, 0.75) - q₁))) .& (p_acceptance .< diagnostic_check.min_acceptance));

    if length(outliers) > 0
        @warn string(length(outliers)) * " poorly mixing chain chains detected, setting to best chain"
        #remove outliers
        best_chain = [argmin(abs.(p_acceptance .- diagnostic_check.target_acceptance))];
        chains.ld[chains.current_position[outliers], :] .= chains.ld[chains.current_position[best_chain], :];
        chains.X[chains.current_position[outliers], :] .= chains.X[chains.current_position[best_chain], :];
    end
end