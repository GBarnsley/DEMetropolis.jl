abstract type update_struct end

struct de_update{T <: Real} <: update_struct
    γ::Distributions.UnivariateDistribution #really we want this to be distribution with type T
    β::Distributions.ContinuousUnivariateDistribution
end

function setup_de_update(
    ld::TransformedLogDensities.TransformedLogDensity;
    γ::Union{Nothing, Distributions.UnivariateDistribution, Real} = nothing,
    β = Distributions.Uniform(-1e-4, 1e-4),
    deterministic_γ = true
)
    if isnothing(γ)
        if deterministic_γ
            γ = Distributions.Dirac(2.38/sqrt(2*LogDensityProblems.dimension(ld)))
        else
            γ = Distributions.Uniform(0.8, 1.2)
        end
    elseif isa(γ, Real)
        γ = Distributions.Dirac(γ)
    end

    return de_update{eltype(γ)}(γ, β)
end

function update!(update::de_update, chains::chains_struct, ld, rng, chain)
    x = get_value(chains, chain);
    sampled_chains = sample_chains(chains, rng, chain, 2);
    x₁ = chains.X[sampled_chains[1], :];
    x₂ = chains.X[sampled_chains[2], :];
    if x₁ == x₂
        #just don't update (save the ld eval)
        update_value!(chains, rng, chain, x, -Inf)
    else
        xₚ = x .+ rand(rng, update.γ) .* (x₁ .- x₂) .+ rand(rng, update.β, length(x));
        update_value!(chains, rng, chain, xₚ, LogDensityProblems.logdensity(ld, xₚ));
    end
end

struct snooker_update{T <: Real} <: update_struct
    γ::Distributions.UnivariateDistribution #really we want this to be distribution with type T
end

function setup_snooker_update(;
    γ::Union{Nothing, Distributions.UnivariateDistribution, Real} = nothing,
    deterministic_γ = true
)
    if isnothing(γ)
        if deterministic_γ
            γ =  Distributions.Dirac(2.38/sqrt(2))
        else
            γ = Distributions.Uniform(0.8, 1.2)
        end
    elseif isa(γ, Real)
        γ = Distributions.Dirac(γ)
    end

    return snooker_update{eltype(γ)}(γ)
end

function update!(update::snooker_update, chains::chains_struct, ld, rng, chain)
    x = get_value(chains, chain);
    sampled_chains = sample_chains(chains, rng, chain, 3);
    x₁ = chains.X[sampled_chains[1], :];
    x₂ = chains.X[sampled_chains[2], :];
    xₐ = chains.X[sampled_chains[3], :];
    if xₐ == x || x₁ == x₂
        #just don't update (save the ld eval)
        update_value!(chains, rng, chain, x, -Inf)
    else
        e = LinearAlgebra.normalize(xₐ .- x);
        xₚ = x .+ rand(rng, update.γ) .* LinearAlgebra.dot(x₁ .- x₂, e) .* e; #really this could be assigned to the next value and just replace if rejected
        update_value!(
            chains, rng, chain, xₚ, LogDensityProblems.logdensity(ld, xₚ),
            (length(x) - 1) * (log(LinearAlgebra.norm(xₐ .- xₚ)) - log(LinearAlgebra.norm(xₐ .- x)))
        )
    end
end

abstract type subspace_sampling_struct <: update_struct end

struct subspace_sampling{T <: Real} <: subspace_sampling_struct
    crossover_probability::T
    δ::Distributions.DiscreteUnivariateDistribution
    ϵ::Distributions.ContinuousUnivariateDistribution
    e::Distributions.ContinuousUnivariateDistribution
end

struct subspace_sampling_fixed_γ{T <: Real} <: subspace_sampling_struct
    crossover_probability::T
    γ::T
    δ::Distributions.DiscreteUnivariateDistribution
    ϵ::Distributions.ContinuousUnivariateDistribution
    e::Distributions.ContinuousUnivariateDistribution
end


function setup_subspace_sampling(;
    γ::Union{Nothing, Real} = nothing,
    crossover_probability = 0.5,
    δ::Union{Real, Distributions.DiscreteUnivariateDistribution} = Distributions.DiscreteUniform(1, 3),
    ϵ = Distributions.Uniform(-1e-4, 1e-4),
    e = Distributions.Normal(0.0, 1e-2),
)
    if isa(δ, Real)
        δ = Distributions.Dirac(δ)
    end

    if isnothing(γ)
        subspace_sampling(
            crossover_probability,
            δ,
            ϵ,
            e
        )
    else
        subspace_sampling_fixed_γ(
            crossover_probability,
            γ,
            δ,
            ϵ,
            e
        )
    end
end

function get_γ(rng, update::subspace_sampling, δ, d)
    2.38 / sqrt(2 * δ * d)
end

function get_γ(rng, update::subspace_sampling_fixed_γ, δ, d)
    update.γ
end

function update!(update::subspace_sampling_struct, chains::chains_struct, ld, rng, chain)
    x = get_value(chains, chain);

    #determine how many dimensions to update
    to_update = rand(rng, length(x)) .< update.crossover_probability;
    d = sum(to_update);
    δ = rand(rng, update.δ);

    #generate candidate
    z = x;
    for _ in 1:δ
        #pick to random chains find the difference and add to the candidate
        z[to_update] .+= diff(chains.X[sample_chains(chains, rng, chain, 2), to_update], dims = 1)[1, :];
    end

    #add the other parts of the equation
    z[to_update] .= x[to_update] .+ (
            (1 .+ rand(rng, update.e, d)) .* get_γ(rng, update, δ, d) .* z[to_update]
        ) .+
        rand(rng, update.ϵ, d);

    update_value!(
        chains, rng, chain, z, LogDensityProblems.logdensity(ld, z)
    )
end