module deMCMC
export run_deMCMC
import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter, OhMyThreads

struct deMCMC_params
    βs::Array{Float64, 4}
    acceptances::Array{Float64, 3}
    chain_draws_1::Array{Int64, 3}
    chain_draws_2::Array{Int64, 3}
end

struct deMCMC_params_parallel
    βs::Array{Float64, 4}
    acceptances::Array{Float64, 3}
    chain_draws_1::Array{Int64, 3}
    chain_draws_2::Array{Int64, 3}
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

function deMCMC_params(iterations, iteration_generation, chains, params, β, rng, parallel)

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
    
    if parallel
        return deMCMC_params_parallel(βs, acceptances, chain_draws_1, chain_draws_2)
    else
        return deMCMC_params(βs, acceptances, chain_draws_1, chain_draws_2)
    end
end

function update_chain!(X, X_ld, de_params::deMCMC_params, ld, γ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    xₚ = X[chain, :] .+ γ .* (X[r1, :] .- X[r2, :]) .+ select_element(de_params.βs, it, gen, chain, :);
    ld_xₚ = ld(xₚ);
    if (ld_xₚ - X_ld[chain]) > select_element(de_params.acceptances, it, gen, chain)
        X[chain, :] .= xₚ;
        X_ld[chain] = ld_xₚ;
    end
end

function update_chain(X, X_ld, de_params::deMCMC_params_parallel, ld, γ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    xₚ = X[chain, :] .+ γ .* (X[r1, :] .- X[r2, :]) .+ select_element(de_params.βs, it, gen, chain, :);
    ld_xₚ = ld(xₚ);
    if (ld_xₚ - X_ld[chain]) > select_element(de_params.acceptances, it, gen, chain)
        return (xₚ', ld_xₚ)
    else
        return (X[chain, :]', X_ld[chain])
    end
end

function update_chains!(X, X_ld, de_params::deMCMC_params, ld, γ, it, iteration_generation, chains)
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

function update_chains!(X, X_ld, de_params::deMCMC_params_parallel, ld, γ, it, iteration_generation, chains)
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

function run_deMCMC_inner(ld, initial_state; n_its, n_burn, n_thin, n_chains, γ, β, rng, parallel, save_burnt)

    dim = size(initial_state, 2);

    if isnothing(γ)
        γ = 2.38/sqrt(2*dim);
    end

    # pre deMCMC setup
    chains = 1:n_chains;
    params = 1:dim;

    X = copy(initial_state);
    X_ld = map(x -> ld(x), eachrow(X));

    #burn in run
    if n_burn > 0
        burns = 1:n_burn;
        burn_de_params = deMCMC_params(burns, 1:1, chains, params, β, rng, parallel);
        burn_p = ProgressMeter.Progress(n_burn; dt = 1.0, desc = "Burn in")
        if save_burnt
            burn_samples, burn_sample_ld = setup_samples(burns, chains, params);
        end
        for it in burns
            update_chains!(X, X_ld, burn_de_params, ld, γ, it, 1, chains);
            if save_burnt
                update_sample!(burn_samples, burn_sample_ld, X, X_ld, it);
            end
            ProgressMeter.next!(burn_p)
        end
        ProgressMeter.finish!(burn_p)
    end

    #sampling run
    iterations = 1:n_its;
    iteration_generation = 1:n_thin;
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, β, rng, parallel);

    samples, sample_ld = setup_samples(iterations, chains, params);
    sampling_p = ProgressMeter.Progress(n_its; dt = 1.0, desc = "Sampling")
    for it in iterations
        update_chains!(X, X_ld, de_params, ld, γ, it, iteration_generation, chains);
        update_sample!(samples, sample_ld, X, X_ld, it);
        ProgressMeter.next!(sampling_p)
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


function run_deMCMC_defaults(; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG, parallel = false, save_burnt = false, kwargs...)
    (; n_its, n_burn, n_thin, n_chains, γ, β, rng, parallel, save_burnt, kwargs...)
end


function run_deMCMC(ld::Function, initial_state::Array{Float64, 2}; kwargs...)
    (; n_its, n_burn, n_thin, n_chains, γ, β, rng, parallel, save_burnt) = run_deMCMC_defaults(;kwargs...)

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

    run_deMCMC_inner(ld, true_initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng, parallel = parallel, save_burnt = save_burnt)
end

function run_deMCMC(ld::Function, dim::Int; kwargs...)
    
    (; n_its, n_burn, n_thin, n_chains, γ, β, rng, parallel, save_burnt) = run_deMCMC_defaults(; kwargs...)

    if isnothing(n_chains)
        n_chains = dim * 2;
    end

    #setup population with random initial values
    initial_state = randn(rng, n_chains, dim);

    run_deMCMC_inner(ld, initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng, parallel = parallel, save_burnt = save_burnt)
end 

function run_deMCMC(ld::TransformedLogDensities.TransformedLogDensity, initial_state::Array{Float64, 2}; kwargs...)
    
    (; n_its, n_burn, n_thin, n_chains, γ, β, rng, parallel, save_burnt) = run_deMCMC_defaults(; kwargs...)

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC(_ld_func, initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng, parallel = parallel, save_burnt = save_burnt)
end

function run_deMCMC(ld::TransformedLogDensities.TransformedLogDensity; kwargs...) 
    
    (; n_its, n_burn, n_thin, n_chains, γ, β, rng, parallel, save_burnt) = run_deMCMC_defaults(; kwargs...)

    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC(_ld_func, LogDensityProblems.dimension(ld); n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng, parallel = parallel, save_burnt = save_burnt)
end
end

#function ld(x)
#    # normal distribution
#    return sum(-0.5 .* ((x .- [1.0, -1.0]) .^ 2))
#end
#dim = 2
#
#output = deMCMC.run_deMCMC(ld, dim; n_its = 1000, n_burn = 5000, n_thin = 10, n_chains = 100);
#output = deMCMC.run_deMCMC(ld, dim; n_its = 1000, n_burn = 5000, n_thin = 10, n_chains = 100, parallel = true);
#
#@time output = deMCMC.run_deMCMC(ld, dim; n_its = 1000, n_burn = 5000, n_thin = 10, n_chains = 100);
#@time output = deMCMC.run_deMCMC(ld, dim; n_its = 1000, n_burn = 5000, n_thin = 10, n_chains = 100, parallel = true);
#
#
#@time output = deMCMC.run_deMCMC(ld, dim; n_its = 10, n_burn = 5000, n_thin = 100, n_chains = 1000);
#@time output = deMCMC.run_deMCMC(ld, dim; n_its = 10, n_burn = 5000, n_thin = 100, n_chains = 1000, parallel = true);
#
#
#
#plot(cat(
#    output.burnt_samples[:, 1, 1],
#    output.samples[:, 1, 1],
#    dims = 1
#))
#
#sum(output.samples, dims = (1, 2))./prod(size(output.samples)[1:2])
#
#plot(output.samples[:, :, 1])
#
#plot(output.samples[:, 1:3, 2])
#
#plot(
#    output.samples[:, :, 1][:],
#    output.samples[:, :, 2][:]
#)
#
#
#(((10000*20) + (1000*20)) * 0.002)/60/60