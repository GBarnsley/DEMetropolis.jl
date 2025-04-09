function halve(i::Int)
    Int(ceil(i * 0.5))
end

#update these to replace with random chains
function outlier_chains(X_ld, current_its)
    #check chain via IQR
    ld_means = StatsBase.mean(X_ld[halve(current_its):current_its, :], dims = 1)[1, :];
    q₁ = StatsBase.quantile(ld_means, 0.25);
    (
        findall(ld_means .< q₁ - 2 * (StatsBase.quantile(ld_means, 0.75) - q₁)),
        argmax(ld_means)
    )
end

function replace_outlier_chains!(X, X_ld, its, rngs)
    outliers, best_chain = outlier_chains(X_ld, its[end] + 1);
    if length(outliers) > 0
        @warn string(length(outliers)) * " outlier chains detected, resampling from remaining chains"
        #remove outliers
        replacement_chains = map(rng -> rand(rng, setdiff(axes(X, 2), outliers)), rngs[outliers]);
        X_ld[its .+ 1, outliers] .= X_ld[its .+ 1, replacement_chains];
        X[its .+ 1, outliers, :] .= X[its .+ 1, replacement_chains, :];
        return true
    else
        return false
    end
end

function poorly_mixing_chains(X, current_its)
    #calculate average acceptance
    h_its = halve(current_its);
    p_acceptance = sum(sum((X[(h_its + 1):current_its, :, :] .- X[h_its:(current_its - 1), :, :]) .!= 0, dims = (3))[:, :, 1] .> 0, dims = 1)[1, :] ./ 
    (current_its - h_its - 1)

    (
        findall(p_acceptance .< 0.05),
        argmin(abs.(p_acceptance .- 0.24))
    )
end

function replace_poorly_mixing_chains!(X, X_ld, its, rngs)
    poorly_mixing, best_chain = poorly_mixing_chains(X_ld, its[end] + 1);
    if length(poorly_mixing) > size(X, 2) * 0.80
        @warn "Majority of chains poorly mixing, making no change for diversity"
        return false
    elseif length(poorly_mixing) > 0
        @warn string(length(poorly_mixing)) * " poorly mixed chains detected, resampling from remaining chains"
        replacement_chains = map(rng -> rand(rng, setdiff(axes(X, 2), poorly_mixing)), rngs[poorly_mixing]);
        #remove outliers
        X_ld[its .+ 1, poorly_mixing] .= X_ld[its .+ 1, replacement_chains];
        X[its .+ 1, poorly_mixing, :] .= X[its .+ 1, replacement_chains, :];
        return true
    else
        return false
    end
end

function chains_converged(X, max_it; min_viable = halve(max_it))
    rhat = MCMCDiagnosticTools.rhat(X[min_viable:max_it, :, :])
    println("Rhat: ", round.(rhat, sigdigits = 3))
    if all(rhat .< 1.2)
        return true
    else
        return false
    end
end