function run_deMCMC_live_defaults(; n_its = 1000, check_every = 5000, n_chains = nothing, γ = nothing, γₛ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, deterministic_γ = true, snooker_p = 0.1, epoch_limit = 100, check_ld = true, check_acceptance = true, kwargs...)
    fitting_parameters = (; check_every, γ, γₛ, β, deterministic_γ, snooker_p, parallel, check_ld, check_acceptance, memory = true, epoch_limit);
    (; n_its, n_chains, rng, save_burnt, fitting_parameters, kwargs...)
end

function run_deMCMC_live_inner(ld, initial_state; n_its, n_chains, rng, save_burnt, fitting_parameters)
    (; check_every, epoch_limit) = fitting_parameters;
    
    dim = size(initial_state, 2);

    tuning_pars = define_tuning_pars(fitting_parameters, dim);

    chains = 1:n_chains;

    X, X_ld = setup_X_X_ld(check_every, n_chains, dim, initial_state, ld);

    rngs = setup_rngs(rng, n_chains);

    #select update function
    update_chains_func, update_chain_func = select_update_funcs(fitting_parameters);
    
    current_it = 1;
    epoch = 1;
    max_it = current_it + check_every - 2;
    min_viable = halve(max_it+1);
    not_converged = true;

    while not_converged && epoch <= epoch_limit
        evolution_epoch!(X, X_ld, current_it:max_it, update_chains_func, chains, update_chain_func, tuning_pars, rngs, ld, "Epoch " * string(epoch) * ":");
        #check if converaged
        if chains_converged(X, max_it; min_viable = min_viable)
            println("Chains converged, stopping sampling")
            not_converged = false;
        else
            #check for outliers
            if replace_outlier_chains!(X, X_ld, 1:max_it)
                #since we've replace outliers, before the current iteration we have identical chains
                #so we can only sample from this point onwards
                min_viable = max_it;
            end
            if replace_poorly_mixing_chains!(X, X_ld, 1:max_it)
                min_viable = max_it;
            end
            #make space
            current_it = max_it + 1;
            max_it = current_it + check_every - 1;
            X = cat(X, Array{eltype(X)}(undef, check_every, n_chains, dim), dims = 1);
            X_ld = cat(X_ld, Array{eltype(X)}(undef, check_every, n_chains), dims = 1);
            epoch = epoch + 1;
            min_viable = max(min_viable, halve(max_it+1));
        end
    end

    #format samples
    #thin the last half to the desired amount
    indices = min_viable .+ cumsum(partition_integer(max_it + 1 - min_viable, n_its));

    output = (
        samples = X[indices, :, :],
        ld = X_ld[indices, :]
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

    run_deMCMC_live_inner(ld, true_initial_state; n_its = n_its, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end

function run_deMCMC_live(ld::Function, dim::Int; kwargs...)
    
    (; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_live_defaults(;kwargs...)

    if isnothing(n_chains)
        n_chains = dim * 2;
    end

    #setup population with random initial values
    initial_state = randn(rng, n_chains, dim);

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