abstract type sampler_scheme_struct end

struct sampler_scheme_single <: sampler_scheme_struct
    update::update_struct
end

struct sampler_scheme_multi <: sampler_scheme_struct
    update_weights::Vector{Float64} #should this be a non-parameteric type?
    updates::Vector{<:update_struct}
    function sampler_scheme_multi(
            update_weights::Vector{Float64}, updates::Vector{<:update_struct})
        if length(update_weights) != length(updates)
            error("Number of update weights must be equal to the number of updates")
        end
        if any(update_weights .< 0)
            error("Update weights must be non-negative")
        end
        return new(update_weights, updates)
    end
end

"""
Create a sampler scheme defining the update steps to be used in `composite_sampler`.

The update used in each iteration for each chain is randomly selected from the `updates` given here.

# Arguments
- `updates...`: One or more `update_struct` objects (e.g., created by `setup_de_update`, `setup_snooker_update`, `setup_subspace_sampling` or your own custom sampler).

# Keyword Arguments
- `w`: A vector of weights corresponding to each update step. If `nothing`, updates are chosen with equal probability. Weights must be non-negative.

# Examples
```jldoctest
# only snooker updates
julia> setup_sampler_scheme(setup_snooker_update())

# DE and Snooker
julia> setup_sampler_scheme(setup_snooker_update(), setup_de_update())

# With weights, snookers 10% of the time
julia> setup_sampler_scheme(setup_snooker_update(), setup_de_update(), w = [0.9, 0.1])
```
"""
function setup_sampler_scheme(updates::update_struct...; w::Vector{Float64} = ones(length(updates)))
    return sampler_scheme_multi(w, collect(updates))
end

function setup_sampler_scheme(update::update_struct)
    return sampler_scheme_single(update)
end

function get_update(sampler_scheme::sampler_scheme_single, rng::AbstractRNG)
    return sampler_scheme.update
end

function get_update(sampler_scheme::sampler_scheme_multi, rng::AbstractRNG)
    return wsample(rng, sampler_scheme.updates, sampler_scheme.update_weights)
end

function adapt_samplers!(sampler_scheme::sampler_scheme_single, chains::chains_struct)
    adapt_update!(sampler_scheme.update, chains)
end

function adapt_samplers!(sampler_scheme::sampler_scheme_multi, chains::chains_struct)
    for update in sampler_scheme.updates
        adapt_update!(update, chains)
    end
end

function check_initial_state(
        n_chains::Int, N₀::Int, n_pars::Int, ld::TransformedLogDensity, memory::Bool)
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

"""
Run a composite MCMC sampler for a fixed number of iterations.

For sampling with your own sampling scheme from `setup_sampling_scheme`

# Arguments
- `ld`: The log-density function to sample from, intended to be a LogDensityProblem.
- `n_its`: The desired number of samples per chain.
- `n_chains`: The number of chains to run.
- `memory`: Boolean indicating whether to *sample from historic values* instead sampling from current chains.
- `initial_state`: An array containing the initial states for the chains and the initial population if `memory=true`.
- `sampler_scheme`: A `sampler_scheme_struct` defining the update steps to use.

# Keyword Arguments
- `thin`: Thinning interval for storing samples. Defaults to 1. If using `memory` this also effects which samples are added to the memory.
- `save_burnt`: Boolean indicating whether to save burn-in samples. Defaults to `false`. Does not save thinned samples
- `rng`: Random number generator. Defaults to `default_rng()`.
- `n_burnin`: Number of burn-in iterations. Defaults to `n_its * 5`. Any adaptions occur over this period.
- `parallel`: Boolean indicating whether to run chains in parallel using multithreading. Defaults to `false`.
- `diagnostic_checks`: A vector of `diagnostic_check_struct` to run during burn-in. Defaults to `nothing`.
- `check_epochs`: Splits `n_burnin` into `check_epochs + 1` epochs and applies the diagnostic checks at the end of each epoch, other than the final epoch. Defaults to 1.

# Returns
- A named tuple containing the samples, sampler scheme, and potentially burn-in samples.

# Example
```jldoctest
# DE and Snooker sample scheme with memory
julia> composite_sampler(
    ld, 1000, 10, true, initial_state, setup_sampler_scheme(setup_snooker_update(), setup_de_update())
)
```

See also [`deMC`](@ref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function composite_sampler(
        ld::TransformedLogDensity, n_its::Int, n_chains::Int, memory::Bool,
        initial_state::Array{<:Real, 2}, sampler_scheme::sampler_scheme_struct;
        thin::Int = 1, save_burnt::Bool = false, rng::AbstractRNG = default_rng(),
        n_burnin::Int = n_its * 5, parallel::Bool = false,
        diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing, check_epochs::Int = 1
)
    N₀, n_pars = size(initial_state)

    check_initial_state(n_chains, N₀, n_pars, ld, memory)

    #setup the chains
    total_iterations = (n_its + n_burnin) * n_chains
    chains = setup_population(
        ld, initial_state, total_iterations, N₀, n_pars, n_chains, memory, parallel, thin)

    #setup rng streams
    rngs = setup_rngs(rng, n_chains)

    #mulithreading
    update_chains_func! = get_update_chains_func(parallel)

    #burnin
    if isnothing(diagnostic_checks) || check_epochs == 0
        epoch!(1:(n_burnin * thin), rngs, chains, ld,
            sampler_scheme, update_chains_func!, "Burnin: ")
    else
        burnin_epochs_iterations = partition_integer(n_burnin, check_epochs + 1)
        for epoch in check_epochs
            epoch!(1:(burnin_epochs_iterations[epoch] * thin), rngs, chains,
                ld, sampler_scheme, update_chains_func!, "Burnin $epoch: ")
            for diagnostic_check in diagnostic_checks
                run_diagnostic_check!(
                    chains, diagnostic_check, rngs, sum(burnin_epochs_iterations[1:epoch]))
            end
        end
        epoch!(1:(burnin_epochs_iterations[end] * thin), rngs, chains,
            ld, sampler_scheme, update_chains_func!, "Burnin final: ")
    end

    #samples
    chains.warmup = false
    epoch!(1:(n_its * thin), rngs, chains, ld,
        sampler_scheme, update_chains_func!, "Sampling: ")

    #format outputs
    sample_indices = n_burnin .+ (1:n_its)
    if save_burnt
        burnt_indices = 1:n_burnin
        return format_output(chains, sampler_scheme, sample_indices, burnt_indices)
    else
        return format_output(chains, sampler_scheme, sample_indices)
    end
end

"""
Run a composite MCMC sampler until a stopping criterion is met.

For sampling with your own sampling scheme from `setup_sampling_scheme`.

# Arguments
- `ld`: The log-density function to sample from, intended to be a LogDensityProblem.
- `epoch_size`: The number of saved iterations per chain per epoch.
- `n_chains`: The number of chains to run.
- `memory`: Boolean indicating whether to *sample from historic values* instead sampling from current chains.
- `initial_state`: An array containing the initial states for the chains and the initial population if `memory==true`.
- `sampler_scheme`: A `sampler_scheme_struct` defining the update steps to use.
- `stopping_criteria`: A `stopping_criteria_struct` defining when to stop sampling.

# Keyword Arguments
- `thin`: Thinning interval for storing samples. Defaults to 1. If using `memory` this also effects which samples are added to the memory.
- `save_burnt`: Boolean indicating whether to save burn-in samples. Defaults to `false`. Does not save thinned samples.
- `rng`: Random number generator. Defaults to `default_rng()`.
- `warmup_epochs`: Number of warm-up epochs before we begin checking the stopping criteria. Defaults to 5. Samples from these epochs won't be included in the final sample. Sampler adaptation only occurs in this period.
- `parallel`: Boolean indicating whether to run chains in parallel using multithreading. Defaults to `false`.
- `epoch_limit`: Maximum number of sampling epochs to run. Defaults to 20.
- `diagnostic_checks`: A vector of `diagnostic_check_struct` to run during warm-up. Defaults to `nothing`.

# Returns
- A named tuple containing the samples, sampler scheme, and potentially burn-in samples.

# Example
```jldoctest
# DE and Snooker sample scheme with memory until Rhat ≤ 1.05
julia> composite_sampler(
    ld, 1000, 10, true, initial_state, setup_sampler_scheme(setup_snooker_update(), setup_de_update()), R̂_stopping_criteria(1.05)
)
```

See also [`deMC`](@ref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function composite_sampler(
        ld::TransformedLogDensity, epoch_size::Int, n_chains::Int, memory::Bool, initial_state::Array{
            <:Real, 2},
        sampler_scheme::sampler_scheme_struct, stopping_criteria::stopping_criteria_struct;
        thin::Int = 1, save_burnt::Bool = false, rng::AbstractRNG = default_rng(),
        warmup_epochs::Int = 5, parallel::Bool = false, epoch_limit::Int = 20,
        diagnostic_checks::Union{Nothing, Vector{<:diagnostic_check_struct}} = nothing
)
    N₀, n_pars = size(initial_state)

    check_initial_state(n_chains, N₀, n_pars, ld, memory)

    #setup the chains, initial for the the minimum number of warm-ups and one epoch

    total_iterations = (warmup_epochs + 1) * epoch_size * n_chains
    chains = setup_population(
        ld, initial_state, total_iterations, N₀, n_pars, n_chains, memory, parallel, thin)

    #setup rng streams
    rngs = setup_rngs(rng, n_chains)

    #mulithreading
    update_chains_func! = get_update_chains_func(parallel)

    for epoch in 1:warmup_epochs
        epoch!(1:(epoch_size * thin), rngs, chains, ld, sampler_scheme,
            update_chains_func!, "Warm-up Epoch $epoch: ")
        if !isnothing(diagnostic_checks) && epoch < warmup_epochs
            for diagnostic_check in diagnostic_checks
                run_diagnostic_check!(chains, diagnostic_check, rngs, epoch * epoch_size)
            end
        end
    end

    not_done = true
    epoch = 1
    sample_from = warmup_epochs * epoch_size #don't sample warmup
    chains.warmup = false #disable adaptation
    while not_done
        epoch!(1:(epoch_size * thin), rngs, chains, ld,
            sampler_scheme, update_chains_func!, "Epoch $epoch: ")
        if epoch > epoch_limit
            println("Epoch limit reached")
            not_done = false
        elseif stop_sampling(
            stopping_criteria, chains, sample_from, (epoch + warmup_epochs) * epoch_size)
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
        burnt_indices = 1:(max_its - sample_indices[1])
        return format_output(chains, sampler_scheme, sample_indices, burnt_indices)
    else
        return format_output(chains, sampler_scheme, sample_indices)
    end
end
