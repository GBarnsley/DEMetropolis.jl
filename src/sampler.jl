abstract type sampler_scheme_struct end

struct sampler_scheme_single <: sampler_scheme_struct
    update::update_struct
end

struct sampler_scheme_multi <: sampler_scheme_struct
    update_weights::Vector{Float64} #should this be a parameteric type?
    updates::Vector{<:update_struct}
    sampler_scheme_multi(update_weights::Vector{Float64}, updates::Vector{<:update_struct}) = begin
        if length(update_weights) != length(updates)
            error("Number of update weights must be equal to the number of updates")
        end
        return new(update_weights, updates)
    end
end

function get_update(sampler_scheme::sampler_scheme_single, rng)
    sampler_scheme.update
end

function get_update(sampler_scheme::sampler_scheme_multi, rng)
    StatsBase.wsample(rng, sampler_scheme.updates, sampler_scheme.update_weights)
end

function composite_sampler(
    ld, n_its, n_chains, memory, initial_state, sampler_scheme::sampler_scheme_struct;
    save_burnt = false, rng = Random.default_rng(), n_burnin = n_its * 5, parallel = false
)

    N₀, n_pars = size(initial_state);
    
    if !memory
        if N₀ != n_chains
            error("Number of chains must be equal to the number of initial states")
        end
    else
        if N₀ < n_chains
            error("Number of chains must be greater than or equal to the number of initial states")
        end
    end

    if n_pars != LogDensityProblems.dimension(ld)
        error("Number of parameters in initial state must be equal to the number of parameters in the log density")
    end

    if n_chains < 4
        error("Number of chains must be greater than or equal to 4")
    end

    #setup the chains
    total_iterations = (n_its + n_burnin) * n_chains;
    chains = setup_population(ld, initial_state, total_iterations, N₀, n_pars, n_chains, memory, parallel);

    #setup rng streams
    rngs = setup_rngs(rng, n_chains);

    #mulithreading
    update_chains_func! = get_update_chains_func(parallel);

    #burnin
    epoch!(1:n_burnin, rngs, chains, ld, sampler_scheme, update_chains_func!, "Burnin: ")

    #samples
    epoch!(1:n_its, rngs, chains, ld, sampler_scheme, update_chains_func!, "Sampling: ")

    #format outputs
    sample_indices = n_burnin .+ (1:n_its);
    if save_burnt
        burnt_indices = 1:n_burnin;
        return format_output(chains, n_chains, N₀, sample_indices, burnt_indices)
    else
        return format_output(chains, n_chains, N₀, sample_indices)
    end
end