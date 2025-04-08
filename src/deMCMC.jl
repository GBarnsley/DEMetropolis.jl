module deMCMC
export run_deMCMC
import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter, OhMyThreads, LinearAlgebra, StatsBase, Logging, MCMCDiagnosticTools

abstract type deMCMC_params_base end

struct deMCMC_params <: deMCMC_params_base
    βs::Array{Float64, 4}
    acceptances::Array{Float64, 3}
    chain_draws_1::Array{Int64, 3}
    chain_draws_2::Array{Int64, 3}
    snooker_draw::Array{Int64, 3}
end

struct deMCMC_params_rγ <: deMCMC_params_base
    γs::Array{Float64, 3}
    base_params::deMCMC_params
end

abstract type deMCMC_params_base_parallel end

struct deMCMC_params_parallel <: deMCMC_params_base_parallel
    βs::Array{Float64, 4}
    acceptances::Array{Float64, 3}
    chain_draws_1::Array{Int64, 3}
    chain_draws_2::Array{Int64, 3}
    snooker_draw::Array{Int64, 3}
end

struct deMCMC_params_parallel_rγ <: deMCMC_params_base_parallel
    γs::Array{Float64, 3}
    base_params::deMCMC_params_parallel
end

abstract type deMCMC_params_base_memory end

struct deMCMC_params_memory <: deMCMC_params_base_memory
    βs::Array{Float64, 3}
    acceptances::Array{Float64, 2}
    chain_draws_1::Array{Int64, 2}
    memory_draws_1::Array{Int64, 2}
    chain_draws_2::Array{Int64, 2}
    memory_draws_2::Array{Int64, 2}
    snooker_draw::Array{Int64, 2}
    snooker_draw_memory::Array{Int64, 2}
end

struct deMCMC_params_memory_rγ <: deMCMC_params_base_memory
    γs::Array{Float64, 2}
    base_params::deMCMC_params_memory
end

function generate_random_numbers(rng, iterations, chains; S = Float64)
    # could try vectors of vectors?
    rand(rng, S,  length(iterations), length(chains))
end

function generate_random_numbers(rng, iterations, iteration_generation, chains; S = Float64)
    # could try vectors of vectors?
    rand(rng, S,  length(iterations), length(iteration_generation), length(chains))
end

function generate_random_numbers(rng, iterations, iteration_generation, chains, params; S = Float64)
    # could try vectors of vectors?
    rand(rng, S,  length(iterations), length(iteration_generation), length(chains), length(params))
end

function select_element(object, iteration, chain)
    object[iteration, chain]
end

function select_element(object, iteration, generation, chain)
    object[iteration, generation, chain]
end

function select_element(object, iteration, generation, chain, params)
    object[iteration, generation, chain, params]
end

function setup_samples(iterations, chains, params)
    (
        Array{Float64, 3}(undef, length(iterations), length(chains), length(params)),
        Array{Float64, 2}(undef, length(iterations), length(chains))
    )
end

function update_sample!(samples, sample_ld, X, X_ld, it)
    samples[it, :, :] .= X;
    sample_ld[it, :] .= X_ld;
end

function deMCMC_params(iterations, iteration_generation, chains, params, rng, fitting_parameters)
    (; β, parallel, deterministic_γ, snooker_p) = fitting_parameters;

    #random β values
    βs = (generate_random_numbers(rng, iterations, iteration_generation, chains, params) .- 0.5) .* 2 .* β;

    #random acceptance values
    acceptances = log.(generate_random_numbers(rng, iterations, iteration_generation, chains));

    other_chains = map(x -> setdiff(chains, [x]), chains);

    #random chain draws
    chain_draws_1 = generate_random_numbers(rng, iterations, iteration_generation, chains, S = 1:(length(chains) - 1));
    for k in axes(chain_draws_1, 3)
        chain_draws_1[:, :, k] .= other_chains[k][chain_draws_1[:, :, k]]
    end

    chain_draws_2 = generate_random_numbers(rng, iterations, iteration_generation, chains, S = 1:(length(chains) - 2));
    for i in axes(chain_draws_2, 1), j in axes(chain_draws_2, 2), k in axes(chain_draws_2, 3)
        chain_draws_2[i, j, k] = setdiff(other_chains[k], chain_draws_1[i, j, k])[chain_draws_2[i, j, k]]
    end
    
    #random chance of a snooker update
    snooker_draw = zeros(Int64, size(chain_draws_2));
    for i in axes(snooker_draw, 1), j in axes(snooker_draw, 2), k in axes(snooker_draw, 3)
        if rand(rng) < snooker_p
            snooker_draw[i, j, k] = rand(rng, other_chains[k])
        end
    end

    if deterministic_γ
        if parallel
            return deMCMC_params_parallel(βs, acceptances, chain_draws_1, chain_draws_2, snooker_draw)
        else
            return deMCMC_params(βs, acceptances, chain_draws_1, chain_draws_2, snooker_draw)
        end
    else
        #random γ values
        γs = (generate_random_numbers(rng, iterations, iteration_generation, chains) .* 0.5) .+ 0.5;
        if parallel
            return deMCMC_params_parallel_rγ(γs, deMCMC_params_parallel(βs, acceptances, chain_draws_1, chain_draws_2, snooker_draw))
        else
            return deMCMC_params_rγ(γs, deMCMC_params(βs, acceptances, chain_draws_1, chain_draws_2, snooker_draw))
        end
    end
end


function deMCMC_params(iterations, chains, params, rng, fitting_parameters)
    (; β, deterministic_γ, snooker_p) = fitting_parameters;

    #random β values
    βs = (generate_random_numbers(rng, iterations, chains, params) .- 0.5) .* 2 .* β;

    #random acceptance values
    acceptances = log.(generate_random_numbers(rng, iterations, chains));

    other_chains = map(x -> setdiff(chains, [x]), chains);

    #random chain draws
    chain_draws_1 = generate_random_numbers(rng, iterations, chains, S = 1:(length(chains) - 1));
    for j in axes(chain_draws_1, 2)
        chain_draws_1[:, j] .= other_chains[j][chain_draws_1[:, j]]
    end

    chain_draws_2 = generate_random_numbers(rng, iterations, chains, S = 1:(length(chains) - 2));
    for i in axes(chain_draws_2, 1), j in axes(chain_draws_2, 2)
        chain_draws_2[i, j] = setdiff(other_chains[j], chain_draws_1[i, j])[chain_draws_2[i, j]]
    end
    
    #memory draws
    memory_draws_1 = similar(chain_draws_1);
    memory_draws_2 = similar(chain_draws_2);
    for i in axes(memory_draws_1, 1)
        past_iterations = 1:i;
        memory_draws_1[i, :] .= rand(rng, past_iterations, length(chains));
        memory_draws_2[i, :] .= rand(rng, past_iterations, length(chains));
    end

    #random chance of a snooker update
    snooker_draw = zeros(Int64, size(chain_draws_2));
    snooker_draw_memory = zeros(Int64, size(chain_draws_2));
    for i in axes(snooker_draw, 1), j in axes(snooker_draw, 2)
        if rand(rng) < snooker_p
            snooker_draw[i, j] = rand(rng, other_chains[j])
            snooker_draw_memory[i, j] = rand(rng, 1:i)
        end
    end

    if deterministic_γ
        return deMCMC_params_memory(
            βs, acceptances, chain_draws_1, memory_draws_1, chain_draws_2, memory_draws_2, snooker_draw, snooker_draw_memory
        )
    else
        #random γ values
        γs = (generate_random_numbers(rng, iterations, chains) .* 0.5) .+ 0.5;
        deMCMC_params_memory_rγ(
            γs, 
            deMCMC_params_memory(
                βs, acceptances, chain_draws_1, memory_draws_1, chain_draws_2, memory_draws_2, snooker_draw, snooker_draw_memory
            )
        )
    end
end

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

#alternative idea, split chains into 3 and then do each grouping at a time where group 1 samples group 2 and 3
function update_chain!(X, X_ld, de_params::deMCMC_params, ld, γ, γₛ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    snooker_r = select_element(de_params.snooker_draw, it, gen, chain);
    if snooker_r > 0
        xₚ, ld_xₚ = snooker_update(
            X[chain, :], X[r1, :], X[r2, :], X[snooker_r, :], ld, γₛ
        );
    else
        xₚ, ld_xₚ = de_update(
            X[chain, :], X[r1, :], X[r2, :], ld, γ, select_element(de_params.βs, it, gen, chain, :)
        );
    end
    if (ld_xₚ - X_ld[chain]) > select_element(de_params.acceptances, it, gen, chain)
        X[chain, :] .= xₚ;
        X_ld[chain] = ld_xₚ;
    end
end

function update_chain!(X, X_ld, de_params::deMCMC_params_rγ, ld, γ, γₛ, it, gen, chain)
    update_chain!(X, X_ld, de_params.base_params, ld, 
        select_element(de_params.γs, it, gen, chain), select_element(de_params.γs, it, gen, chain), 
    it, gen, chain);
end

function update_chain(X, X_ld, de_params::deMCMC_params_parallel, ld, γ, γₛ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    snooker_r = select_element(de_params.snooker_draw, it, gen, chain);
    if snooker_r > 0
        xₚ, ld_xₚ = snooker_update(
            X[chain, :], X[r1, :], X[r2, :], X[snooker_r, :], ld, γₛ
        );
    else
        xₚ, ld_xₚ = de_update(
            X[chain, :], X[r1, :], X[r2, :], ld, γ, select_element(de_params.βs, it, gen, chain, :)
        );
    end
    if (ld_xₚ - X_ld[chain]) > select_element(de_params.acceptances, it, gen, chain)
        return (xₚ', ld_xₚ)
    else
        return (X[chain, :]', X_ld[chain])
    end
end

function update_chain(X, X_ld, de_params::deMCMC_params_parallel_rγ, ld, γ, γₛ, it, gen, chain)
    update_chain(X, X_ld, de_params.base_params, ld, 
        select_element(de_params.γs, it, gen, chain), select_element(de_params.γs, it, gen, chain),
    it, gen, chain)
end

function update_chains!(X, X_ld, de_params::deMCMC_params_base, ld, γ, γₛ, it, iteration_generation, chains)
    for gen in iteration_generation, chain in chains
        update_chain!(X, X_ld, de_params, ld, γ, γₛ, it, gen, chain)
    end
end

function combine_chains(x1, x2)
    (
        cat(x1[1], x2[1], dims = 1),
        cat(x1[2], x2[2], dims = 1)
    )
end

function update_chains!(X, X_ld, de_params::deMCMC_params_base_parallel, ld, γ, γₛ, it, iteration_generation, chains)
    for gen in iteration_generation
        #slightly different algorithm where we update all chains in parallel based on the previous generation
        output = OhMyThreads.tmapreduce(
            x -> update_chain(X, X_ld, de_params, ld, γ, γₛ, it, gen, x),
            combine_chains,
            chains
        );
        X .= output[1];
        X_ld .= output[2];
    end
end

function update_chain!(X, X_ld, de_params::deMCMC_params_memory, ld, γ, γₛ, it, chain)
    r1 = select_element(de_params.chain_draws_1, it, chain);
    i1 = select_element(de_params.memory_draws_1, it, chain);
    r2 = select_element(de_params.chain_draws_2, it, chain);
    i2 = select_element(de_params.memory_draws_2, it, chain);
    snooker_r = select_element(de_params.snooker_draw, it, chain);
    if snooker_r > 0
        xₚ, ld_xₚ = snooker_update(
            X[it, chain, :], X[i1, r1, :], X[i2, r2, :], X[select_element(de_params.snooker_draw_memory, it, chain), snooker_r, :], ld, γₛ
        );
    else
        xₚ, ld_xₚ = de_update(
            X[it, chain, :], X[i1, r1, :], X[i2, r2, :], ld, γ, select_element(de_params.βs, it, chain, :)
        );
    end
    if (ld_xₚ - X_ld[it, chain]) > select_element(de_params.acceptances, it, chain)
        X[it + 1, chain, :] .= xₚ;
        X_ld[it + 1, chain] = ld_xₚ;
    else 
        X[it + 1, chain, :] .= X[it, chain, :];
        X_ld[it + 1, chain] = X_ld[it, chain];
    end
end

function update_chain!(X, X_ld, de_params::deMCMC_params_memory_rγ, ld, γ, γₛ, it, chain)
    update_chain!(X, X_ld, de_params.base_params, ld, 
        select_element(de_params.γs, it, chain), select_element(de_params.γs, it, chain), 
    it, chain);
end

function update_chains_threaded!(X, X_ld, de_params::deMCMC_params_base_memory, ld, γ, γₛ, it, chains)
    #slightly different algorithm where we update all chains in parallel based on the previous generation
    OhMyThreads.tmap(
        x -> update_chain!(X, X_ld, de_params, ld, γ, γₛ, it, x),
        chains
    );
end

function update_chains!(X, X_ld, de_params::deMCMC_params_base_memory, ld, γ, γₛ, it, chains)
    #slightly different algorithm where we update all chains in parallel based on the previous generation
    map(
        x -> update_chain!(X, X_ld, de_params, ld, γ, γₛ, it, x),
        chains
    );
end

function partition_integer(I::Int, n::Int)
    base = I ÷ n  # Base size of each group
    remainder = I % n  # Remaining units to distribute

    # Create n groups: first 'remainder' groups get (base + 1), the rest get 'base'
    return vcat(fill(base + 1, remainder), fill(base, n - remainder))
end

function partition_integer_indices(I::Int, n::Int)
    values = cumsum(partition_integer(I, n))
    map(v -> (v - values[1] + 1):v, values)
end

function generate_epochs(n_its, n_thin, n_chains, epoch_limit)
    total_its = n_its * n_chains * n_thin;
    n_epoch = Int(ceil(total_its / epoch_limit));
    its_per_epoch = partition_integer(n_its, n_epoch)
    epochs = 1:n_epoch;
    return epochs, its_per_epoch
end

function evolution_epoch!(X, X_ld, epoch, its_per_epoch, ld, iteration_generation, chains, params, rng, p, fitting_parameters, γ, γₛ)
    iterations = 1:(its_per_epoch[epoch]);
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, rng, fitting_parameters);
    for it in iterations
        update_chains!(X, X_ld, de_params, ld, γ, γₛ, it, iteration_generation, chains);
        ProgressMeter.next!(p)
    end
end

function evolution_epoch_sample!(X, X_ld, samples, sample_ld, epoch, its_per_epoch, ld, iteration_generation, chains, params, rng, p, fitting_parameters, γ, γₛ)
    iterations = 1:(its_per_epoch[epoch]);
    it_offset = sum(its_per_epoch[1:(epoch - 1)]);
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, rng, fitting_parameters);
    for it in iterations
        update_chains!(X, X_ld, de_params, ld, γ, γₛ, it, iteration_generation, chains);
        update_sample!(samples, sample_ld, X, X_ld, it + it_offset);
        ProgressMeter.next!(p)
    end
end

function evolution_epoch!(X, X_ld, de_params, ld, γ, γₛ, its, chains, update_func, desc)
    p = ProgressMeter.Progress(length(its); dt = 1.0, desc = desc)
    for it in its
        update_func(X, X_ld, de_params, ld, γ, γₛ, it, chains);
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
end

function outlier_chains(X_ld, current_its)
    #check chain via IQR
    ld_means = StatsBase.mean(X_ld[Int(ceil(current_its * 0.5)):current_its, :], dims = 1)[1, :];
    q₁ = StatsBase.quantile(ld_means, 0.25);
    (
        findall(ld_means .< q₁ - 2 * (StatsBase.quantile(ld_means, 0.75) - q₁)),
        argmax(ld_means)
    )
end

function replace_outlier_chains!(X, X_ld, its)
    outliers, best_chain = outlier_chains(X_ld, its[end] + 1);
    if length(outliers) > 0
        @warn string(length(outliers)) * " outlier chains detected, setting to current best chain"
        #remove outliers
        X_ld[its .+ 1, outliers] .= X_ld[its .+ 1, best_chain];
        X[its .+ 1, outliers, :] .= X[its .+ 1, best_chain:best_chain, :];
    end
end

function run_deMCMC_inner(ld, initial_state; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters)

    dim = size(initial_state, 2);

    if isnothing(fitting_parameters.γ) && fitting_parameters.deterministic_γ
        γ = 2.38/sqrt(2*dim);
    else 
        γ = fitting_parameters.γ
    end

    if isnothing(fitting_parameters.γₛ) && fitting_parameters.deterministic_γ
        γₛ = 2.38/sqrt(2);
    else 
        γₛ = fitting_parameters.γₛ
    end

    chains = 1:n_chains;
    params = 1:dim; 

    if fitting_parameters.memory
        n_total_iterations = n_its * n_thin + n_burn;
        # setup initial conditions
        X = Array{Float64}(undef, n_total_iterations + 1, n_chains, dim);
        X_ld = Array{Float64}(undef, n_total_iterations + 1, n_chains);
        X[1, :, :] .= initial_state;
        X_ld[1, :] .= map(x -> ld(x), eachrow(initial_state));

        # setup parameters/
        total_iterations = 1:n_total_iterations;
        de_params = deMCMC_params(total_iterations, chains, params, rng, fitting_parameters);

        if fitting_parameters.parallel
            update_func = update_chains_threaded!
        else
            update_func = update_chains!
        end

        #iterate burnin
        burn_epochs = fitting_parameters.check_chain_epochs;
        its_per_epoch = partition_integer_indices(n_burn, burn_epochs)
        check_epochs = its_per_epoch[1:(burn_epochs - 1)]
        final_epoch = its_per_epoch[burn_epochs]
        for (epoch, its) in enumerate(check_epochs)
            evolution_epoch!(X, X_ld, de_params, ld, γ, γₛ, its, chains, update_func, "Burn Epoch " * string(epoch) * ":");

            replace_outlier_chains!(X, X_ld, its);
        end

        evolution_epoch!(X, X_ld, de_params, ld, γ, γₛ, final_epoch, chains, update_func, "Burn Epoch Final:");

        sampling_epoch = (final_epoch[end] + 1):n_total_iterations;
        evolution_epoch!(X, X_ld, de_params, ld, γ, γₛ, sampling_epoch, chains, update_func, "Sampling:");

        #format samples
        if save_burnt
            burn_indices = (1:n_burn) .+ 1;
            burn_samples = X[burn_indices, :, :];
            burn_sample_ld = X_ld[burn_indices, :];
        end
        indices = ((n_burn + 1):n_thin:(n_total_iterations)) .+ 1;
        samples = X[indices, :, :];
        sample_ld = X_ld[indices, :];
    else
        # pre deMCMC setup
        X = copy(initial_state);
        X_ld = map(x -> ld(x), eachrow(X));

        epoch_limit = max(1e6, n_chains * dim * 1000); #define this better

        #burn in run
        if n_burn > 0
            burn_p = ProgressMeter.Progress(n_burn; dt = 1.0, desc = "Burn in")
            if save_burnt
                burn_samples, burn_sample_ld = setup_samples(1:n_burn, chains, params);
            end
            epochs, its_per_epoch = generate_epochs(n_burn, 1, n_chains, epoch_limit);
            for epoch in epochs
                if save_burnt
                    evolution_epoch_sample!(X, X_ld, burn_samples, burn_sample_ld, epoch, its_per_epoch, ld, 1:1, chains, params, rng, burn_p, fitting_parameters, γ, γₛ);
                else
                    evolution_epoch!(X, X_ld, epoch, its_per_epoch, ld, 1:1, chains, params, rng, burn_p, fitting_parameters, γ, γₛ);
                end
            end
            ProgressMeter.finish!(burn_p)
        end

        #sampling run
        iteration_generation = 1:n_thin;
        epochs, its_per_epoch = generate_epochs(n_its, n_thin, n_chains, epoch_limit);
        samples, sample_ld = setup_samples(1:n_its, chains, params);
        sampling_p = ProgressMeter.Progress(n_its; dt = 1.0, desc = "Sampling")
        for epoch in epochs
            evolution_epoch_sample!(X, X_ld, samples, sample_ld, epoch, its_per_epoch, ld, iteration_generation, chains, params, rng, sampling_p, fitting_parameters, γ, γₛ);
        end
        ProgressMeter.finish!(sampling_p)
    end

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


function run_deMCMC_defaults(; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, γₛ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, deterministic_γ = true, snooker_p = 0.1, memory = false, check_chain_epochs = 1, kwargs...)
    fitting_parameters = (; γ, γₛ, β, parallel, deterministic_γ, snooker_p, memory, check_chain_epochs);
    (; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters, kwargs...)
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

    run_deMCMC_inner(ld, true_initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, rng = rng, save_burnt = save_burnt, fitting_parameters = fitting_parameters)
end

function run_deMCMC(ld::Function, dim::Int; kwargs...)
    
    (; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_defaults(; kwargs...)

    if isnothing(n_chains)
        n_chains = dim * 2;
    end

    #setup population with random initial values
    initial_state = randn(rng, n_chains, dim);

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

function update_chain_live!(X, X_ld, it, chain, ld, γ, γₛ, r1, i1, r2, i2, snooker, snooker_i, βs, acceptance)
    if snooker > 0
        xₚ, ld_xₚ = snooker_update(
            X[it, chain, :], X[i1, r1, :], X[i2, r2, :], X[snooker_i, snooker, :], ld, γₛ
        );
    else
        xₚ, ld_xₚ = de_update(
            X[it, chain, :], X[i1, r1, :], X[i2, r2, :], ld, γ, βs
        );
    end
    if (ld_xₚ - X_ld[it, chain]) > acceptance
        X[it + 1, chain, :] .= xₚ;
        X_ld[it + 1, chain] = ld_xₚ;
    else 
        X[it + 1, chain, :] .= X[it, chain, :];
        X_ld[it + 1, chain] = X_ld[it, chain];
    end
end

function update_chains_live_parallel!(X, X_ld, it, chains, n_chains, γ, γₛ, β, deterministic_γ, snooker_p, rng)
    r1 = [rand(rng, setdiff(chains, chain)) for chain in chains];
    r2 = [rand(rng, setdiff(chains, [chain, r1])) for (chain, r1) in enumerate(r1)];
    i1 = rand(rng, 1:it, length(chains));
    i2 = rand(rng, 1:it, length(chains));
    snooker = zeros(Int64, size(r1));
    snooker_i = zeros(Int64, size(r1));
    for c in chains
        if rand(rng) < snooker_p
            snooker[c] = rand(rng, setdiff(chains, c));
            snooker_i[c] = rand(rng, 1:it);
        end
    end
    βs = rand(rng, n_chains, dim) .- 0.5 .* 2 .* β;
    acceptances = log.(rand(rng, n_chains));
    OhMyThreads.tmap(
        x -> update_chain_live!(X, X_ld, it, x, ld, γ, γₛ, r1[x], i1[x], r2[x], i2[x], snooker[x], snooker_i[x], βs[x, :], acceptances[x]),
        chains
    );
end

function update_chains_live!(X, X_ld, it, chains, n_chains, γ, γₛ, β, deterministic_γ, snooker_p, rng)
    for chain in chains
        r1 = rand(rng, setdiff(chains, chain))
        if rand(rng) < snooker_p
            snooker_r = rand(rng, setdiff(chains, chain));
            snooker_i = rand(rng, 1:it);
        else
            snooker_r = 0;
            snooker_i = 0;
        end
        update_chain_live!(
            X, X_ld, it, chain, ld, γ, γₛ, 
            r1,
            rand(rng, 1:it),
            rand(rng, setdiff(chains, [chain, r1])),
            rand(rng, 1:it),
            snooker_r, snooker_i,
            rand(rng, dim),
            log.(rand(rng))
        );
    end
end

function chains_converged(X, max_it)
    #check chain via IQR
    rhat = MCMCDiagnosticTools.rhat(X[Int(ceil(max_it * 0.5)):max_it, :, :])
    println("Rhat: ", rhat)
    if all(rhat .< 1.2)
        return true
    else
        return false
    end
end

function run_deMCMC_live_inner(ld, initial_state; n_its, n_chains, rng, save_burnt, fitting_parameters)
    (; check_every, γ, γₛ, β, deterministic_γ, snooker_p, parallel) = fitting_parameters

    dim = size(initial_state, 2);

    if isnothing(fitting_parameters.γ) && fitting_parameters.deterministic_γ
        γ = 2.38/sqrt(2*dim);
    else 
        γ = fitting_parameters.γ
    end

    if isnothing(fitting_parameters.γₛ) && fitting_parameters.deterministic_γ
        γₛ = 2.38/sqrt(2);
    else 
        γₛ = fitting_parameters.γₛ
    end

    chains = 1:n_chains;
    params = 1:dim; 

    X = Array{Float64}(undef, check_every, n_chains, dim);
    X_ld = Array{Float64}(undef, check_every, n_chains);
    X[1, :, :] .= initial_state;
    X_ld[1, :] .= map(x -> ld(x), eachrow(initial_state));

    if parallel
        update_func = update_chains_live_parallel!
    else
        update_func = update_chains_live!
    end

    current_it = 1;
    epoch = 1;
    max_it = current_it + check_every - 2;
    not_converged = true;

    while not_converged
        p = ProgressMeter.Progress(n_its; dt = 1.0, desc = "Epoch " * string(epoch) * ":")
        for it in current_it:max_it
            update_func(X, X_ld, it, chains, n_chains, γ, γₛ, β, deterministic_γ, snooker_p, rng);
            ProgressMeter.next!(p)
        end
        ProgressMeter.finish!(p)
        #check if converaged
        if chains_converged(X, max_it)
            println("Chains converged, stopping sampling")
            not_converged = false;
        else
            #check for outliers
            replace_outlier_chains!(X, X_ld, 1:max_it);
            #make space
            current_it = max_it + 1;
            max_it = current_it + check_every - 1;
            X = cat(X, Array{Float64}(undef, check_every, n_chains, dim), dims = 1);
            X_ld = cat(X_ld, Array{Float64}(undef, check_every, n_chains), dims = 1);
            epoch = epoch + 1;
        end
    end

    #format samples
    if save_burnt
        burn_samples = X;
        burn_sample_ld = X_ld;
    end
    #thin the last half to the desired amount
    min_viable = Int(round((max_it+1)/2));
    indices = min_viable .+ cumsum(partition_integer(max_it + 1 - min_viable, n_its));
    samples = X[indices, :, :];
    sample_ld = X_ld[indices, :];

    output = (
        samples = samples,
        ld = sample_ld
    )
    if save_burnt
        output = (
            output...,
            burnt_samples = burn_samples,
            burnt_ld = burn_sample_ld
        )
    end
    return output
end


function run_deMCMC_live_defaults(; n_its = 1000, check_every = 5000, n_chains = nothing, γ = nothing, γₛ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, deterministic_γ = true, snooker_p = 0.1, kwargs...)
    fitting_parameters = (; check_every, γ, γₛ, β, deterministic_γ, snooker_p, parallel);
    (; n_its, n_chains, rng, save_burnt, fitting_parameters, kwargs...)
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

#function ld(x)
#    # normal distribution
#    return sum(-0.5 .* ((x .- [1.0, -1.0]) .^ 2))
#end
#dim = 2
#(; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_deMCMC_live_defaults();
#n_chains = 50;
#initial_state = randn(rng, n_chains, dim);

end

#using MCMCDiagnosticTools, Plots
#function ld(x)
#    # normal distribution
#    return sum(-0.5 .* ((x .- [1.0, -1.0]) .^ 2))
#end
#dim = 2
#
##output = deMCMC.run_deMCMC(ld, dim; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = 100, deterministic_γ = false, memory = true, parallel = true);
#output = deMCMC.run_deMCMC_live(ld, dim; n_its = 1000, check_every = 5000, n_chains = 100, deterministic_γ = true, parallel = true);
#
#println(ess_rhat(output.samples))
#plot(
#    output.ld
#)
#plot(
#    output.samples[:, :, 1]
#)
#plot(
#    output.samples[:, :, 2]
#)