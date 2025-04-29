abstract type sampler_scheme_struct end

struct sampler_scheme_single <: sampler_scheme_struct
    update::update_struct
end

struct sampler_scheme_multi <: sampler_scheme_struct
    update_weights::Vector{Float64} #should this be a non-parameteric type?
    updates::Vector{<:update_struct}
    sampler_scheme_multi(update_weights::Vector{Float64}, updates::Vector{<:update_struct}) = begin
        if length(update_weights) != length(updates)
            error("Number of update weights must be equal to the number of updates")
        end
        return new(update_weights, updates)
    end
end

function setup_sampler_scheme(updates...; w = nothing)
    if isnothing(w)
        w = ones(length(updates)) ./ length(updates)
    end
    if any(w .< 0)
        error("Update weights must be non-negative")
    end
    if length(w) > 1
        sampler_scheme_multi(w, collect(updates))
    else
        sampler_scheme_single(updates[1])
    end
end

function get_update(sampler_scheme::sampler_scheme_single, rng)
    sampler_scheme.update
end

function get_update(sampler_scheme::sampler_scheme_multi, rng)
    wsample(rng, sampler_scheme.updates, sampler_scheme.update_weights)
end

function adapt_samplers!(sampler_scheme::sampler_scheme_single, chains)
    adapt_update!(sampler_scheme.update, chains)
end

function adapt_samplers!(sampler_scheme::sampler_scheme_multi, chains)
    for update in sampler_scheme.updates
        adapt_update!(update, chains)
    end
end

function check_initial_state(n_chains, N₀, n_pars, ld, memory)
    if !memory
        if N₀ != n_chains
            error("Number of chains must be equal to the number of initial states")
        end
        if n_chains < 4
            error("Number of chains must be greater than or equal to 4")
        end
    else
        if N₀ < 3 + n_chains
            error("The initial population must be greater than or equal to 3 + the number of chains")
        end
    end

    if n_pars != dimension(ld)
        error("Number of parameters in initial state must be equal to the number of parameters in the log density")
    end

    nothing
end

function composite_sampler(
    ld, n_its, n_chains, memory, initial_state, sampler_scheme::sampler_scheme_struct;
    thin = 1, save_burnt = false, rng = default_rng(), n_burnin = n_its * 5, parallel = false,
    diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing, check_epochs = 1
)

    N₀, n_pars = size(initial_state);
    
    check_initial_state(n_chains, N₀, n_pars, ld, memory);

    #setup the chains
    total_iterations = (n_its + n_burnin) * n_chains;
    chains = setup_population(ld, initial_state, total_iterations, N₀, n_pars, n_chains, memory, parallel, thin);

    #setup rng streams
    rngs = setup_rngs(rng, n_chains);

    #mulithreading
    update_chains_func! = get_update_chains_func(parallel);

    #burnin
    if isnothing(diagnostic_checks) || check_epochs == 0
        epoch!(1:(n_burnin * thin), rngs, chains, ld, sampler_scheme, update_chains_func!, "Burnin: ")
    else
        burnin_epochs_iterations = partition_integer(n_burnin, check_epochs + 1);
        for epoch in check_epochs
            epoch!(1:(burnin_epochs_iterations[epoch] * thin), rngs, chains, ld, sampler_scheme, update_chains_func!, "Burnin $epoch: ")
            for diagnostic_check in diagnostic_checks
                run_diagnostic_check!(chains, diagnostic_check, rngs, sum(burnin_epochs_iterations[1:epoch])) 
            end
        end
        epoch!(1:(burnin_epochs_iterations[end] * thin), rngs, chains, ld, sampler_scheme, update_chains_func!, "Burnin final: ")
    end

    #samples
    chains.warmup = false;
    epoch!(1:(n_its * thin), rngs, chains, ld, sampler_scheme, update_chains_func!, "Sampling: ")

    #format outputs
    sample_indices = n_burnin .+ (1:n_its);
    if save_burnt
        burnt_indices = 1:n_burnin;
        return format_output(chains, sampler_scheme, sample_indices, burnt_indices)
    else
        return format_output(chains, sampler_scheme, sample_indices)
    end
end


function composite_sampler(
    ld, epoch_size, n_chains, memory, initial_state, sampler_scheme::sampler_scheme_struct, stopping_criteria::stopping_criteria_struct;
    thin = 1, save_burnt = false, rng = default_rng(), warmup_epochs = 5, parallel = false, epoch_limit = 20,
    diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing
)

    N₀, n_pars = size(initial_state);
    
    check_initial_state(n_chains, N₀, n_pars, ld, memory);

    #setup the chains, initial for the the minimum number of warm-ups and one epoch

    total_iterations = (warmup_epochs + 1) * epoch_size * n_chains;
    chains = setup_population(ld, initial_state, total_iterations, N₀, n_pars, n_chains, memory, parallel, thin);

    #setup rng streams
    rngs = setup_rngs(rng, n_chains);

    #mulithreading
    update_chains_func! = get_update_chains_func(parallel);

    for epoch in 1:warmup_epochs
        epoch!(1:(epoch_size * thin), rngs, chains, ld, sampler_scheme, update_chains_func!, "Warm-up Epoch $epoch: ")
        if !isnothing(diagnostic_checks) && epoch < warmup_epochs
            for diagnostic_check in diagnostic_checks
                run_diagnostic_check!(chains, diagnostic_check, rngs, epoch * epoch_size) 
            end
        end
    end

    not_done = true;
    epoch = 1;
    sample_from = warmup_epochs * epoch_size; #don't sample warmup
    chains.warmup = false; #disable adaptation
    while not_done
        epoch!(1:(epoch_size * thin), rngs, chains, ld, sampler_scheme, update_chains_func!, "Epoch $epoch: ")
        if epoch > epoch_limit
            println("Epoch limit reached")
            not_done = false
        elseif stop_sampling(stopping_criteria, chains, sample_from, (epoch + warmup_epochs) * epoch_size)
            println("Stopping criteria met")
            not_done = false
        else
            epoch += 1
            #add more space to the chains
            resize_chains!(chains, length(chains.ld) + (epoch_size * n_chains))
        end
    end

    #format outputs
    #sample from 
    max_its = (epoch + warmup_epochs) * epoch_size
    sample_indices = get_sampling_indices(sample_from, max_its)
    if save_burnt
        burnt_indices = 1:(max_its - sample_indices[1]);
        return format_output(chains, sampler_scheme, sample_indices, burnt_indices)
    else
        return format_output(chains, sampler_scheme, sample_indices)
    end
end