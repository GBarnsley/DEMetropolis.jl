abstract type update_struct end

abstract type de_update <: update_struct end

struct de_update_deterministic{T <: Real} <: de_update
    γ::T
    β::Distributions.ContinuousUnivariateDistribution #really we want this to be distribution with type T
end

struct de_update_random{T <: Real} <: de_update
    γ::Distributions.ContinuousUnivariateDistribution #really we want this to be distribution with type T
    β::Distributions.ContinuousUnivariateDistribution
end

function setup_de_update(
    ld::TransformedLogDensities.TransformedLogDensity;
    γ::Union{Nothing, Distributions.ContinuousUnivariateDistribution, Real} = nothing,
    β = Distributions.Uniform(-1e-4, 1e-4),
    deterministic_γ = true
)
    if isnothing(γ)
        if deterministic_γ
            γ =  2.38/sqrt(2*LogDensityProblems.dimension(ld))
        else
            γ = Distributions.Uniform(0.8, 1.2)
        end
    end
    if isa(γ, Real)
        return de_update_deterministic{eltype(γ)}(γ, β)
    else
        return de_update_random{eltype(γ)}(γ, β)
    end
end

function get_γ(rng, update::de_update_deterministic)
    return update.γ
end

function get_γ(rng, update::de_update_random)
    return rand(rng, update.γ)
end

function update!(update::de_update, chains::chains_struct, ld, rng, chain, xₚ)
    x = get_value(chains, chain);
    sampled_chains = sample_chains(chains, rng, chain, 2);
    x₁ = chains.X[sampled_chains[1], :];
    x₂ = chains.X[sampled_chains[2], :];
    if x₁ == x₂
        #just don't update (save the ld eval)
        update_value!(chains, rng, chain, x, -Inf)
    else
        xₚ .= x .+ get_γ(rng, update) .* (x₁ .- x₂) .+ rand(rng, update.β, length(x));
        update_value!(chains, rng, chain, xₚ, LogDensityProblems.logdensity(ld, xₚ));
    end
end

abstract type snooker_update <: update_struct end

struct snooker_update_deterministic{T <: Real} <: snooker_update
    γ::T
end

struct snooker_update_random{T <: Real} <: snooker_update
    γ::Distributions.ContinuousUnivariateDistribution #really we want this to be distribution with type T
end

function get_γ(rng, update::snooker_update_deterministic)
    return update.γ
end

function get_γ(rng, update::snooker_update_random)
    return rand(rng, update.γ)
end

function setup_snooker_update(;
    γ::Union{Nothing, Distributions.ContinuousUnivariateDistribution, Real} = nothing,
    deterministic_γ = true
)
    if isnothing(γ)
        if deterministic_γ
            γ =  2.38/sqrt(2)
        else
            γ = Distributions.Uniform(0.8, 1.2)
        end
    end
    if isa(γ, Real)
        return snooker_update_deterministic{eltype(γ)}(γ)
    else
        return snooker_update_random{eltype(γ)}(γ)
    end
end

function update!(update::snooker_update, chains::chains_struct, ld, rng, chain, xₚ)
    x = get_value(chains, chain);
    sampled_chains = sample_chains(chains, rng, chain, 3);
    x₁ = chains.X[sampled_chains[1], :];
    x₂ = chains.X[sampled_chains[2], :];
    xₐ = chains.X[sampled_chains[3], :];
    if xₐ == x || x₁ == x₂
        #just don't update (save the ld eval)
        update_value!(chains, rng, chain, x, -Inf)
    else
        diff = x₁ .- x₂;
        e = LinearAlgebra.normalize(xₐ .- x);
        xₚ .= x .+ get_γ(rng, update) .* LinearAlgebra.dot(diff, e) .* e; #really this could be assigned to the next value and just replace if rejected
        update_value!(
            chains, rng, chain, xₚ, LogDensityProblems.logdensity(ld, xₚ),
            (length(x) - 1) * (log(LinearAlgebra.norm(xₐ .- xₚ)) - log(LinearAlgebra.norm(xₐ .- x)))
        )
    end
end

function DREAM_defaults(; kwargs...)
    (; kwargs...)
end

function DREAM_update(x, x₁, x₂, xₐ, ld, γₛ)
    #todo
end