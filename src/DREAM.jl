function run_DREAM_defaults(; n_its = 1000, n_chains = nothing, n_burnin = 500, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, check_ld = true, kwargs...)
    fitting_parameters = (; n_burnin, parallel, check_ld);
    (; n_its, n_chains, rng, save_burnt, fitting_parameters, kwargs...)
end

function update_chain_DREAM!(X, X_ld, it, chain, chains, tuning_pars, rng, ld)
    r1 = rand(rng, setdiff(chains, chain), δ); #precalc these in tuning_pars
    r2 = map(chain1 -> rand(rng, setdiff(chains, [chain, chain1])), r1);
    z = X[it, chain, :] .+ (1.0 .+ e) .* γ(δ, d) .* (sum(X[it, r1, :] .- X[it, r2, :], dims = 1)) .+ ϵ;

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

function run_DREAM_inner(ld, initial_state; n_its, n_chains, rng, save_burnt, fitting_parameters)
    (;  n_burnin, parallel, check_ld) = fitting_parameters;
    
    dim = size(initial_state, 2);

#    tuning_pars = define_tuning_pars(fitting_parameters, dim);

    chains = 1:n_chains;
    
    X, X_ld = setup_X_X_ld(n_burnin, n_chains, dim, initial_state, ld);

    rngs = setup_rngs(rng, n_chains);

    #select update function
    update_chains_func, update_chain_func = select_update_funcs(fitting_parameters);
    
    update_chain_func = update_chain_DREAM!;

    current_it = 1;
    epoch = 1;
    max_it = current_it + check_every - 2;
    min_viable = map(c -> halve(max_it+1), 1:n_chains);
    not_converged = true;

    while not_converged
        evolution_epoch!(X, X_ld, current_it:max_it, update_chains_func, chains, update_chain_func, tuning_pars, rngs, ld, "Epoch " * string(epoch) * ":");
        #check if converaged
        if chains_converged(X, max_it; min_viable = min_viable)
            println("Chains converged, stopping sampling")
            not_converged = false;
        elseif epoch > epoch_limit
            println("Epoch limit reached, stopping sampling")
            not_converged = false;
        else
            #check for outliers
            if check_ld
                resampled_chains = replace_outlier_chains!(X, X_ld, current_it:max_it, rngs; check_every = check_every);
                if length(resampled_chains) > 0
                    #set the min viable for these chains to the current iteration
                    min_viable[resampled_chains] .= max_it;
                end
            end
            if check_acceptance
                resampled_chains = replace_poorly_mixing_chains!(X, X_ld, current_it:max_it, rngs; check_every = check_every);
                if length(resampled_chains) > 0
                    min_viable[resampled_chains] .= max_it;
                end
            end
            #make space
            current_it = max_it + 1;
            max_it = current_it + check_every - 1;
            X = cat(X, Array{eltype(X)}(undef, check_every, n_chains, dim), dims = 1);
            X_ld = cat(X_ld, Array{eltype(X)}(undef, check_every, n_chains), dims = 1);
            epoch = epoch + 1;
            min_viable = max.(min_viable, halve(max_it+1));
        end
    end

    #format samples
    output = (
        samples = thin_X(X, min_viable, max_it, n_its),
        ld = thin_X_ld(X_ld, min_viable, max_it, n_its)
    )
    if save_burnt
        output = (
            output...,
            burnt_samples = X,
            burnt_ld = X_ld
        )
    end
    return output
end

function run_DREAM(ld::Function, initial_state::Array{Float64, 2}; kwargs...)
    (; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_DREAM_defaults(;kwargs...)

    dim = size(initial_state, 2);

    if isnothing(n_chains)
        n_chains = size(initial_state, 1);
    end

    if n_chains <= dim
        @warn "n_chains ≤ the number of variables, sampler will perform poorly"
    end

    #check initial state matches
    n_initial_state_chains = size(initial_state, 1);
    if n_initial_state_chains == n_chains
        true_initial_state = copy(initial_state);
    elseif n_initial_state_chains < n_chains
        @warn "initial_state has fewer chains than n_chains, adding random chains"
        true_initial_state = cat(initial_state, randn(rng, n_chains - n_initial_state_chains, dim), dims = 1);
    else
        @warn "initial_state has more chains than n_chains, removing chains at random"
        true_initial_state = initial_state[Random.randperm(rng, n_initial_state_chains)[1:n_chains], :];
    end

    true_initial_state = reshape(true_initial_state, 1, n_chains, dim);
    if fitting_parameters.N₀ > 1
        #add random memory chains
        true_initial_state = cat(randn(rng, fitting_parameters.N₀ - 1, n_chains, dim), true_initial_state, dims = 1);
    end

    run_DREAM_inner(ld, true_initial_state; n_its = n_its, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end

function run_DREAM(ld::Function, dim::Int; kwargs...)
    
    (; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_DREAM_defaults(;kwargs...)

    if isnothing(n_chains)
        n_chains = dim * 2;
    end


    #setup population with random initial values
    initial_state = randn(rng, n_chains, dim);

    run_DREAM_inner(ld, initial_state; n_its = n_its, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end 

function run_DREAM(ld::TransformedLogDensities.TransformedLogDensity, initial_state::Array{Float64, 2}; kwargs...)

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_DREAM(_ld_func, initial_state;  kwargs...)
end

function run_DREAM(ld::TransformedLogDensities.TransformedLogDensity; kwargs...) 

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_DREAM(_ld_func, LogDensityProblems.dimension(ld);  kwargs...)
end