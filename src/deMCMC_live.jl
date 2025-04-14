function run_deMCMC_live_defaults(; n_its = 1000, check_every = 5000, n_chains = nothing, γ = nothing, γₛ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, deterministic_γ = true, snooker_p = 0.1, epoch_limit = 100, check_ld = true, check_acceptance = false, N₀ = 1, kwargs...)
    fitting_parameters = (; check_every, γ, γₛ, β, deterministic_γ, snooker_p, parallel, check_ld, check_acceptance, memory = true, epoch_limit, N₀);
    (; n_its, n_chains, rng, save_burnt, fitting_parameters, kwargs...)
end

function partition_integer_for_thin(I, n)
    base = I ÷ n  # Base size of each group
    remainder = I - base * n # Remaining units to distribute

    # Create an array with base size for all groups
    partition = fill(base, n)
    # Evenly distribute the remainder across the groups
    if remainder > 0
        partition[cumsum(partition_integer(n, remainder))] .+= 1
    end

    return partition
end

function thin_X(X, min_viable, max_it, n_its)
    thinned_X = Array{eltype(X)}(undef, n_its, size(X, 2), size(X, 3));
    for i in axes(X, 2)
        thinned_X[:, i, :] .= X[min_viable[i] .+ cumsum(partition_integer(max_it + 1 - min_viable[i], n_its)), i, :]
    end

    return thinned_X
end

function thin_X_ld(X_ld, min_viable, max_it, n_its)
    thinned_X_ld = Array{eltype(X_ld)}(undef, n_its, size(X_ld, 2));
    for i in axes(X_ld, 2)
        thinned_X_ld[:, i] .= X_ld[min_viable[i] .+ cumsum(partition_integer(max_it + 1 - min_viable[i], n_its)), i]
    end

    return thinned_X_ld
end

function run_deMCMC_live_inner(ld, initial_state; n_its, n_chains, rng, save_burnt, fitting_parameters)
    (; check_every, epoch_limit, check_ld, check_acceptance) = fitting_parameters;
    
    dim = size(initial_state, 3);

    tuning_pars = define_tuning_pars(fitting_parameters, dim);

    chains = 1:n_chains;
    N₀ = fitting_parameters.N₀;
    
    X, X_ld = setup_X_X_ld(check_every, n_chains, dim, initial_state, ld, N₀);

    rngs = setup_rngs(rng, n_chains);

    #select update function
    update_chains_func, update_chain_func = select_update_funcs(fitting_parameters);
    
    current_it = N₀;
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

function run_deMCMC_live(ld::Function, initial_state::Array{Float64, 2}; kwargs...)
    (; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_live_defaults(;kwargs...)

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

    run_deMCMC_live_inner(ld, true_initial_state; n_its = n_its, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end

function run_deMCMC_live(ld::Function, dim::Int; kwargs...)
    
    (; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_live_defaults(;kwargs...)

    if isnothing(n_chains)
        n_chains = dim * 2;
    end

    #setup population with random initial values
    initial_state = randn(rng, fitting_parameters.N₀, n_chains, dim);

    run_deMCMC_live_inner(ld, initial_state; n_its = n_its, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end 

function run_deMCMC_live(ld::TransformedLogDensities.TransformedLogDensity, initial_state::Array{Float64, 2}; kwargs...)

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC_live(_ld_func, initial_state;  kwargs...)
end

function run_deMCMC_live(ld::TransformedLogDensities.TransformedLogDensity; kwargs...) 

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC_live(_ld_func, LogDensityProblems.dimension(ld);  kwargs...)
end