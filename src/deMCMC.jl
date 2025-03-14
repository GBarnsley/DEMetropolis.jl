module deMCMC
import Random
export run_deMCMC

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

function deMCMC_params(iterations, iteration_generation, chains, params, β, rng)

    #random β values
    βs = (generate_random_numbers(rng, iterations, iteration_generation, chains, params) .- 0.5) .* 2 .* β;

    #random acceptance values
    acceptances = log.(generate_random_numbers(rng, iterations, iteration_generation, chains));

    other_chains = map(x -> setdiff(chains, [x]), chains);

    #random chain draws
    chain_draws_1 = generate_random_numbers(rng, iterations, iteration_generation, chains, S = 1:(length(chains) - 1));
    for i in axes(chain_draws_1, 3)
        chain_draws_1[:, :, i] .= other_chains[i][chain_draws_1[:, :, i]]
    end
    
    chain_draws_2 = generate_random_numbers(rng, iterations, iteration_generation, chains, S = 1:(length(chains) - 2));
    for (i, j, k) in axes(chain_draws_2)
        chain_draws_2[i, j, k] = setdiff(other_chains[k], chain_draws_1[i, j, k])[chain_draws_2[i, j, k]]
    end
    
    return deMCMC_params(βs, acceptances, chain_draws_1, chain_draws_2)
end

function update_chain!(X, X_ll, de_params::deMCMC_params, ld, γ, it, gen, chain)
    r1 = select_element(de_params.chain_draws_1, it, gen, chain);
    r2 = select_element(de_params.chain_draws_2, it, gen, chain);
    xₚ = X[chain] .+ γ .* (X[r1] .- X[r2]) .+ select_element(de_params.βs, it, gen, chain, :);
    if (ld(xₚ) - ld(X[chain])) > select_element(de_params.acceptances, it, gen, chain)
        X[chain] .= xₚ;
        X_ll[chain] = ld(xₚ);
    end
end

function update_sample!(samples, sample_ll, X, X_ll, it)
    samples[it] = X;
    sample_ll[it] = X_ll;
end

function run_deMCMC(ld::Function, dim; n_its = 1000, n_thin = 1, n_chains = 10, γ = 0.1, β = 0.1, rng = Random.GLOBAL_RNG)
    n_its = 1000; n_thin = 5; n_chains = 10; γ = 0.1; β = 0.1; rng = Random.GLOBAL_RNG;
    # pre deMCMC setup
    iterations = 1:n_its;
    iteration_generation = 1:n_thin;
    chains = 1:n_chains;
    params = 1:dim;
    de_params = deMCMC_params(iterations, iteration_generation, chains, params, β, rng);

    # setup population with random initial values (improve this)
    X = map(x -> (rand(rng, dim) .- 0.5) .* 2, chains);
    X_ll = map(x -> ld(x), X);
    
    samples = Array{Array{Array{Float64, 1}, 1}, 1}(undef, n_its);
    sample_ll = Array{Array{Float64, 1}, 1}(undef, n_its);

    for it in iterations
        for gen in iteration_generation; chain in chains;
            update_chain!(X, X_ll, de_params, ld, γ, it, gen, chain)
        end
        update_sample!(samples, sample_ll, X, X_ll, it);
    end
end

end