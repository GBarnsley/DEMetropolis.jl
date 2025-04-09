function snooker_update(x, x₁, x₂, xₐ, ld, γₛ)
    diff = x₁ .- x₂;
    e = LinearAlgebra.normalize(xₐ .- x);
    xₚ = x .+ γₛ .* LinearAlgebra.dot(diff, e) .* e;
    (
        xₚ,
        ld(xₚ) + (length(x) - 1) * (log(LinearAlgebra.norm(xₐ .- xₚ)) - log(LinearAlgebra.norm(xₐ .- x)))
    )
end

function de_update(x, x₁, x₂, ld, γ, β)
    xₚ = x .+ γ .* (x₁ .- x₂) .+ β
    (
        xₚ,
        ld(xₚ)
    )
end

function update_chain_inner!(X, X_ld, it, chain, i1, i2, is, γₛ, γ, tuning_pars, rng, ld)
    r1 = rand(rng, setdiff(chains, chain));
    r2 = rand(rng, setdiff(chains, [chain, r1]));
    if rand(rng) < tuning_pars.snooker_p
        xₚ, ld_xₚ = snooker_update(
            X[it, chain, :], X[i1, r1, :], X[i2, r2, :], X[is, rand(rng, setdiff(chains, chain)), :], ld, γₛ
        );
    else
        xₚ, ld_xₚ = de_update(
            X[it, chain, :], X[i1, r1, :], X[i2, r2, :], ld, γ, (rand(rng, size(X, 3)) .- 0.5) .* 2 .* tuning_pars.β
        );
    end
    if log(rand(rng)) < (ld_xₚ - X_ld[it, chain])
        X[it + 1, chain, :] .= xₚ;
        X_ld[it + 1, chain] = ld_xₚ;
    else 
        X[it + 1, chain, :] .= X[it, chain, :];
        X_ld[it + 1, chain] = X_ld[it, chain];
    end
end

function update_chain!(X, X_ld, it, chain, tuning_pars, rng, ld)
    update_chain_inner!(X, X_ld, it, chain, it, it, it, tuning_pars.γₛ, tuning_pars.γ, tuning_pars, rng, ld)
end

function update_chain_rγ!(X, X_ld, it, chain, tuning_pars, rng, ld)
    γₛ = tuning_pars.γₛ + ((rand(rng) .* 0.5) .+ 0.5);
    γ = tuning_pars.γ + ((rand(rng) .* 0.5) .+ 0.5);
    update_chain_inner!(X, X_ld, it, chain, it, it, it, γₛ, γ, tuning_pars, rng, ld)
end

function update_chain_memory!(X, X_ld, it, chain, tuning_pars, rng, ld)
    i1 = rand(rng, 1:it);
    i2 = rand(rng, 1:it);
    is = rand(rng, 1:it);
    update_chain_inner!(X, X_ld, it, chain, i1, i2, is, tuning_pars.γₛ, tuning_pars.γ, tuning_pars, rng, ld)
end

function update_chain_memory_rγ!(X, X_ld, it, chain, tuning_pars, rng, ld)
    i1 = rand(rng, 1:it);
    i2 = rand(rng, 1:it);
    is = rand(rng, 1:it);
    γₛ = tuning_pars.γₛ + ((rand(rng) .* 0.5) .+ 0.5);
    γ = tuning_pars.γ + ((rand(rng) .* 0.5) .+ 0.5);
    update_chain_inner!(X, X_ld, it, chain, i1, i2, is, γₛ, γ, tuning_pars, rng, ld)
end

function update_chains!(X, X_ld, it, chains, update_chain_func, tuning_pars, rngs, ld)
    for chain in chains
        update_chain_func(X, X_ld, it, chain, tuning_pars, rngs[chain], ld)
    end
end

function update_chains_threaded!(X, X_ld, it, chains, update_chain_func, tuning_pars, rngs, ld)
    Threads.@threads for chain in chains
        update_chain_func(X, X_ld, it, chain, tuning_pars, rngs[chain], ld)
    end
end