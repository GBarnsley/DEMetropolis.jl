function run_deMCMC_defaults(; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, γₛ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, deterministic_γ = true, snooker_p = 0.1, memory = false, check_chain_epochs = 1, check_ld = true, check_acceptance = true, N₀ = 1, kwargs...)
    fitting_parameters = (; γ, γₛ, β, parallel, deterministic_γ, snooker_p, memory, check_chain_epochs, check_ld, check_acceptance, N₀ = N₀);
    (; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters, kwargs...)
end

function define_tuning_pars(fitting_parameters, dim)
    if isnothing(fitting_parameters.γ)
        γ = 2.38/sqrt(2*dim);
    else 
        γ = fitting_parameters.γ
    end
    if isnothing(fitting_parameters.γₛ)
        γₛ = 2.38/sqrt(2);
    else 
        γₛ = fitting_parameters.γₛ
    end
    β = fitting_parameters.β
    snooker_p = fitting_parameters.snooker_p
    (; γ, γₛ, β, snooker_p)
end

function calculate_total_iterations(n_its, n_thin, n_burn)
    return n_its * n_thin + n_burn
end

function setup_X_X_ld(n_its, n_chains, dim, initial_state, ld; initial_ld = nothing)
    X = Array{eltype(initial_state)}(undef, n_its + 1, n_chains, dim);
    X_ld = Array{eltype(initial_state)}(undef, n_its + 1, n_chains);
    X[1, :, :] .= initial_state;
    if isnothing(initial_ld)
        X_ld[1, :] .= map(x -> ld(x), eachrow(initial_state));
    else
        X_ld[1, :] .= initial_ld;
    end
    return X, X_ld
end

function setup_X_X_ld(n_its, n_chains, dim, initial_state, ld, N₀)
    X = Array{eltype(initial_state)}(undef, n_its + N₀, n_chains, dim);
    X_ld = Array{eltype(initial_state)}(undef, n_its + N₀, n_chains);
    X[1:N₀, :, :] .= initial_state;
    for i in 1:N₀, j in 1:n_chains
        X_ld[i, j] = ld(X[i, j, :]);
    end
    return X, X_ld
end

function restart_X_X_ld!(X, X_ld, its, memory)
    if !memory
        #return to start
        X[1, :, :] .= X[its[end] + 1, :, :];
        X_ld[1, :] .= X_ld[its[end] + 1, :];
    end
end

function record_samples!(samples, sample_ld, X, X_ld, its, memory, previous_its, N₀)
    if memory
        #record samples
        samples[its .- N₀ .+ 1, :, :] .= X[its .+ 1, :, :];
        sample_ld[its .- N₀ .+ 1, :] .= X_ld[its .+ 1, :];
    else
        offset = sum(map(i -> i[end], previous_its));
        #record samples
        samples[its .+ offset, :, :] .= X[its .+ 1, :, :];
        sample_ld[its .+ offset, :] .= X_ld[its .+ 1, :];
    end

end

function select_update_funcs(fitting_parameters)
    if fitting_parameters.parallel
        update_chains_func = update_chains_threaded!
    else
        update_chains_func = update_chains!
    end

    if fitting_parameters.memory
        if fitting_parameters.deterministic_γ
            update_chain_func = update_chain_memory!
        else
            update_chain_func = update_chain_memory_rγ!
        end
    else
        if fitting_parameters.deterministic_γ
            update_chain_func = update_chain!
        else
            update_chain_func = update_chain_rγ!
        end
    end

    return (update_chains_func, update_chain_func)
end

function setup_rngs(rng, n_chains)
    [Random.MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains]
end

function run_deMCMC_inner(ld, initial_state; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters)

    memory = fitting_parameters.memory;

    if memory
        dim = size(initial_state, 3);
    else
        dim = size(initial_state, 2);
    end

    tuning_pars = define_tuning_pars(fitting_parameters, dim);

    chains = 1:n_chains;

    if memory
        N₀ = fitting_parameters.N₀
        total_iterations = calculate_total_iterations(n_its, n_thin, n_burn);
        X, X_ld = setup_X_X_ld(total_iterations, n_chains, dim, initial_state, ld, N₀)
    else
        N₀ = 1;
        sampling_iterations = n_its * n_thin;
    end

    rngs = setup_rngs(rng, n_chains);

    #select update function
    update_chains_func, update_chain_func = select_update_funcs(fitting_parameters);

    #burn in run
    if n_burn > 0
        if save_burnt
            burn_samples = Array{eltype(initial_state)}(undef, n_burn, n_chains, dim);
            burn_sample_ld = Array{eltype(initial_state)}(undef, n_burn, n_chains);
        end

        burn_epochs = fitting_parameters.check_chain_epochs;
        its_per_epoch = partition_its_over_epochs(n_burn, burn_epochs, N₀, memory);

        if !memory
            X, X_ld = setup_X_X_ld(its_per_epoch[1][end], n_chains, dim, initial_state, ld)
        end

        check_epochs = its_per_epoch[1:(burn_epochs - 1)]
        final_epoch = its_per_epoch[burn_epochs]
        for (epoch, its) in enumerate(check_epochs)
            evolution_epoch!(X, X_ld, its, update_chains_func, chains, update_chain_func, tuning_pars, rngs, ld, "Burn Epoch " * string(epoch) * ":");
            if save_burnt
                record_samples!(burn_samples, burn_sample_ld, X, X_ld, its, memory, its_per_epoch[1:(epoch - 1)], N₀);
            end
            if fitting_parameters.check_ld
                replace_outlier_chains!(X, X_ld, its, rngs);
            end
            if fitting_parameters.check_acceptance            
                replace_poorly_mixing_chains!(X, X_ld, its, rngs);
            end
            restart_X_X_ld!(X, X_ld, its, memory);
        end
        evolution_epoch!(X, X_ld, final_epoch, update_chains_func, chains, update_chain_func, tuning_pars, rngs, ld, "Burn Epoch Final:");
        if save_burnt
            record_samples!(burn_samples, burn_sample_ld, X, X_ld, final_epoch, memory, check_epochs, N₀);
        end
        if !memory
            setup_X_X_ld(sampling_iterations, n_chains, dim, X[final_epoch[end] + 1, :, :], ld; initial_ld = X_ld[final_epoch[end] + 1, :]);
        end
    end

    #sampling run
    if memory
        sampling_epoch = (final_epoch[end] + 1):total_iterations;
    else
        sampling_epoch = 1:sampling_iterations;
    end
    evolution_epoch!(X, X_ld, sampling_epoch, update_chains_func, chains, update_chain_func, tuning_pars, rngs, ld, "Sampling:");

    #thin samples
    sampling_indices = cumsum(partition_integer(length(sampling_epoch), n_its)) .+ sampling_epoch[1];
    
    samples = X[sampling_indices, :, :];
    sample_ld = X_ld[sampling_indices, :];

    #format output
    output = (
        samples = samples,
        ld = sample_ld
    )
    if n_burn > 0 && save_burnt
        output = (
            output...,
            burnt_samples = burn_samples,
            burnt_ld = burn_sample_ld
        )
    end
    return output
end

function run_deMCMC(ld::Function, initial_state::Array{Float64, 2}; kwargs...)
    (; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_defaults(;kwargs...)

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

    if memory
        true_initial_state = reshape(true_initial_state, 1, n_chains, dim);
        if fitting_parameters.N₀ > 1
            #add random memory chains
            true_initial_state = cat(randn(rng, fitting_parameters.N₀ - 1, n_chains, dim), true_initial_state, dims = 1);
        end
    end

    run_deMCMC_inner(ld, true_initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end

function run_deMCMC(ld::Function, dim::Int; kwargs...)
    
    (; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_defaults(; kwargs...)

    if isnothing(n_chains)
        n_chains = dim * 2;
    end

    #setup population with random initial values
    if fitting_parameters.memory
        initial_state = randn(rng, fitting_parameters.N₀, n_chains, dim);
    else
        initial_state = randn(rng, n_chains, dim);
    end
    
    run_deMCMC_inner(ld, initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end 

function run_deMCMC(ld::TransformedLogDensities.TransformedLogDensity, initial_state::Array{Float64, 2}; kwargs...)

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC(_ld_func, initial_state;  kwargs...)
end

function run_deMCMC(ld::TransformedLogDensities.TransformedLogDensity; kwargs...) 

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC(_ld_func, LogDensityProblems.dimension(ld);  kwargs...)
end
