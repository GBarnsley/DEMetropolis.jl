module deMCMC
export run_deMCMC
import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter, OhMyThreads, LinearAlgebra

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

function generate_random_numbers(rng, iterations, iteration_generation, chains; S = Float64)
    # could try vectors of vectors?
    rand(rng, S,  length(iterations), length(iteration_generation), length(chains))
end

function generate_random_numbers(rng, iterations, iteration_generation, chains, params; S = Float64)
    # could try vectors of vectors?
    rand(rng, S,  length(iterations), length(iteration_generation), length(chains), length(params))
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

function snooker_update(X, chain, r1, r2, snooker_r, ld, γₛ)
    diff = X[r1, :] .- X[r2, :];
    e = LinearAlgebra.normalize(X[snooker_r, :] .- X[chain, :]);
    xₚ = X[chain, :] .+ γₛ .* LinearAlgebra.dot(diff, e) .* e;
    (
        xₚ,
        ld(xₚ) + (size(X, 2) - 1) * (log(LinearAlgebra.norm(X[snooker_r, :] .- xₚ)) - log(LinearAlgebra.norm(X[snooker_r, :] .- X[chain, :])))
    )
end

function de_update(X, chain, r1, r2, ld, γ, β)
    xₚ = X[chain, :] .+ γ .* (X[r1, :] .- X[r2, :]) .+ β
    (
        xₚ,
        ld(xₚ)
    )
end

#alternative idea, split chains into 3 and then do each grouping at a time where group 1 samples group 2 and 3
function update_chain!(X, X_ld, de_params::deMCMC_params, ld, γ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    snooker_r = select_element(de_params.snooker_draw, it, gen, chain);
    if snooker_r > 0
        xₚ, ld_xₚ = snooker_update(X, chain, r1, r2, snooker_r, ld, 1.7);
    else
        xₚ, ld_xₚ = de_update(X, chain, r1, r2, ld, γ, select_element(de_params.βs, it, gen, chain, :));
    end
    if (ld_xₚ - X_ld[chain]) > select_element(de_params.acceptances, it, gen, chain)
        X[chain, :] .= xₚ;
        X_ld[chain] = ld_xₚ;
    end
end

function update_chain!(X, X_ld, de_params::deMCMC_params_rγ, ld, γ, it, gen, chain)
    update_chain!(X, X_ld, de_params.base_params, ld, select_element(de_params.γs, it, gen, chain), it, gen, chain);
end

function update_chain(X, X_ld, de_params::deMCMC_params_parallel, ld, γ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    snooker_r = select_element(de_params.snooker_draw, it, gen, chain);
    if snooker_r > 0
        xₚ, ld_xₚ = snooker_update(X, chain, r1, r2, snooker_r, ld, 1.7);
    else
        xₚ, ld_xₚ = de_update(X, chain, r1, r2, ld, γ, select_element(de_params.βs, it, gen, chain, :));
    end
    if (ld_xₚ - X_ld[chain]) > select_element(de_params.acceptances, it, gen, chain)
        return (xₚ', ld_xₚ)
    else
        return (X[chain, :]', X_ld[chain])
    end
end

function update_chain(X, X_ld, de_params::deMCMC_params_parallel_rγ, ld, γ, it, gen, chain)
    update_chain(X, X_ld, de_params.base_params, ld, select_element(de_params.γs, it, gen, chain), it, gen, chain)
end

function update_chains!(X, X_ld, de_params::deMCMC_params_base, ld, γ, it, iteration_generation, chains)
    for gen in iteration_generation, chain in chains
        update_chain!(X, X_ld, de_params, ld, γ, it, gen, chain)
    end
end

function combine_chains(x1, x2)
    (
        cat(x1[1], x2[1], dims = 1),
        cat(x1[2], x2[2], dims = 1)
    )
end

function update_chains!(X, X_ld, de_params::deMCMC_params_base_parallel, ld, γ, it, iteration_generation, chains)
    for gen in iteration_generation
        #slightly different algorithm where we update all chains in parallel based on the previous generation
        output = OhMyThreads.tmapreduce(
            x -> update_chain(X, X_ld, de_params, ld, γ, it, gen, x),
            combine_chains,
            chains
        );
        X .= output[1];
        X_ld .= output[2];
    end
end

function partition_integer(I::Int, n::Int)
    base = I ÷ n  # Base size of each group
    remainder = I % n  # Remaining units to distribute

    # Create n groups: first 'remainder' groups get (base + 1), the rest get 'base'
    return vcat(fill(base + 1, remainder), fill(base, n - remainder))
end

function generate_epochs(n_its, n_thin, n_chains, epoch_limit)
    total_its = n_its * n_chains * n_thin;
    n_epoch = Int(ceil(total_its / epoch_limit));
    its_per_epoch = partition_integer(n_its, n_epoch)
    epochs = 1:n_epoch;
    return epochs, its_per_epoch
end

function evolution_epoch!(X, X_ld, epoch, its_per_epoch, ld, iteration_generation, chains, params, rng, p, fitting_parameters, γ)
    iterations = 1:(its_per_epoch[epoch]);
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, rng, fitting_parameters);
    for it in iterations
        update_chains!(X, X_ld, de_params, ld, γ, it, iteration_generation, chains);
        ProgressMeter.next!(p)
    end
end

function evolution_epoch_sample!(X, X_ld, samples, sample_ld, epoch, its_per_epoch, ld, iteration_generation, chains, params, rng, p, fitting_parameters, γ)
    iterations = 1:(its_per_epoch[epoch]);
    it_offset = sum(its_per_epoch[1:(epoch - 1)]);
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, rng, fitting_parameters);
    for it in iterations
        update_chains!(X, X_ld, de_params, ld, γ, it, iteration_generation, chains);
        update_sample!(samples, sample_ld, X, X_ld, it + it_offset);
        ProgressMeter.next!(p)
    end
end

function run_deMCMC_inner(ld, initial_state; n_its, n_burn, n_thin, n_chains, rng, save_burnt, fitting_parameters)

    dim = size(initial_state, 2);

    if isnothing(fitting_parameters.γ) && fitting_parameters.deterministic_γ
        γ = 2.38/sqrt(2*dim);
    else 
        γ = fitting_parameters.γ
    end

    # pre deMCMC setup
    chains = 1:n_chains;
    params = 1:dim;

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
                evolution_epoch_sample!(X, X_ld, burn_samples, burn_sample_ld, epoch, its_per_epoch, ld, 1:1, chains, params, rng, burn_p, fitting_parameters, γ);
            else
                evolution_epoch!(X, X_ld, epoch, its_per_epoch, ld, 1:1, chains, params, rng, burn_p, fitting_parameters, γ);
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
        evolution_epoch_sample!(X, X_ld, samples, sample_ld, epoch, its_per_epoch, ld, iteration_generation, chains, params, rng, sampling_p, fitting_parameters, γ);
    end
    ProgressMeter.finish!(sampling_p)

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


function run_deMCMC_defaults(; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, deterministic_γ = true, snooker_p = 0.1, kwargs...)
    fitting_parameters = (; γ, β, parallel, deterministic_γ, snooker_p);
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

function ld(x)
    # normal distribution
    return sum(-0.5 .* ((x .- [1.0, -1.0]) .^ 2))
end
#dim = 2
#
#n_its = 1000
#n_burn = 5000
#n_thin = 1
#n_chains = 100
#γ = 0.5
#β = 1e-4
#rng = Random.GLOBAL_RNG
#parallel = false
#save_burnt = true

end

#using MCMCDiagnosticTools, Plots
#function ld(x)
#    # normal distribution
#    return sum(-0.5 .* ((x .- [1.0, -1.0]) .^ 2))
#end
#dim = 2
#
#output = deMCMC.run_deMCMC(ld, dim; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = 100, deterministic_γ = false);
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