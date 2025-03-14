module deMCMC
export run_deMCMC
import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter

struct deMCMC_params
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

function deMCMC_params(iterations, iteration_generation, chains, params, β, rng)

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
    
    return deMCMC_params(βs, acceptances, chain_draws_1, chain_draws_2)
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

function run_deMCMC_inner(ld, initial_state; n_its, n_burn, n_thin, n_chains, γ, β, rng)

    dim = size(initial_state, 2);

    if isnothing(γ)
        γ = 2.38/sqrt(2*dim);
    end

    # pre deMCMC setup
    iterations = 1:n_its;
    iteration_generation = 1:n_thin;
    chains = 1:n_chains;
    params = 1:dim;
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, β, rng);

    X = copy(initial_state);
    X_ld = map(x -> ld(x), eachrow(X));

    #burn in run
    if n_burn > 0
        burns = 1:n_burn;
        burn_de_params = deMCMC_params(burns, 1:1, chains, params, β, rng);
        burn_samples, burn_sample_ld = setup_samples(burns, chains, params);
        burn_p = ProgressMeter.Progress(n_burn; dt = 1.0, desc = "Burn in")
        for it in burns
            for chain in chains
                update_chain!(X, X_ld, burn_de_params, ld, γ, it, 1, chain)
            end
            update_sample!(burn_samples, burn_sample_ld, X, X_ld, it);
            ProgressMeter.next!(burn_p)
        end
        ProgressMeter.finish!(burn_p)
    end

    #sampling run
    samples, sample_ld = setup_samples(iterations, chains, params);
    sampling_p = ProgressMeter.Progress(n_its; dt = 1.0, desc = "Sampling")
    for it in iterations
        for gen in iteration_generation, chain in chains
            update_chain!(X, X_ld, de_params, ld, γ, it, gen, chain)
        end
        update_sample!(samples, sample_ld, X, X_ld, it);
        ProgressMeter.next!(sampling_p)
    end
    ProgressMeter.finish!(sampling_p)

    #format output
    output = (
        samples = samples,
        ld = sample_ld
    )
    if n_burn > 0
        output = (
            output...,
            burnt_samples = burn_samples,
            burnt_ld = burn_sample_ld
        )
    end

    return output
end


function run_deMCMC(ld::Function, initial_state::Array{Float64, 2}; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG)

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

    run_deMCMC_inner(ld, true_initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng)
end

function run_deMCMC(ld::Function, dim::Int; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG)
    
    if isnothing(n_chains)
        n_chains = dim * 2;
    end

    #setup population with random initial values
    initial_state = randn(rng, n_chains, dim);

    run_deMCMC_inner(ld, initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng)
end 

function run_deMCMC(ld::TransformedLogDensities.TransformedLogDensity, initial_state::Array{Float64, 2}; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG)
    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC(_ld_func, initial_state; n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng)
end

function run_deMCMC(ld::TransformedLogDensities.TransformedLogDensity; n_its = 1000, n_burn = 5000, n_thin = 1, n_chains = nothing, γ = nothing, β = 1e-4, rng = Random.GLOBAL_RNG)
    function _ld_func(x)
        LogDensityProblems.logdensity(ld, x)
    end

    run_deMCMC(_ld_func, LogDensityProblems.dimension(ld); n_its = n_its, n_burn = n_burn, n_thin = n_thin, n_chains = n_chains, γ = γ, β = β, rng = rng)
end

end
