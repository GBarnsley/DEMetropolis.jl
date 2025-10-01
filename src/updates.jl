abstract type update_struct end

function adapt_update!(update::update_struct, chains::chains_struct)
    nothing
end

struct de_update{T <: Real} <: update_struct
    γ::Distributions.UnivariateDistribution #really we want this to be distribution with type T
    β::Distributions.ContinuousUnivariateDistribution
end

"""
Set up a Differential Evolution (DE) update step.

See doi.org/10.1007/s11222-006-8769-1 for more information.

# Arguments
- `ld`: The log-density function (used to determine dimension if `γ` is not provided).

# Keyword Arguments
- `γ`: The scaling factor for the difference vector. Can be a `Real`, a `Distributions.UnivariateDistribution`, or `nothing`. If `nothing`, it defaults based on `deterministic_γ` and the problem dimension.
- `β`: Distribution for the small noise term added to the proposal. Defaults to `Uniform(-1e-4, 1e-4)`.
- `deterministic_γ`: If `true` and `γ` is `nothing`, sets `γ` to the theoretically optimal `2.38 / sqrt(2 * dim)`. If `false`, sets `γ` to `Uniform(0.8, 1.2)`. Defaults to `true`.

# Example
```jldoctest
julia> setup_de_update(ld; β = Normal(0.0, 0.01))
```
See also [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).
"""
function setup_de_update(
        ld::TransformedLogDensity;
        γ::Union{Nothing, Distributions.UnivariateDistribution, Real} = nothing,
        β::Distributions.ContinuousUnivariateDistribution = Distributions.Uniform(-1e-4, 1e-4),
        deterministic_γ::Bool = true
)
    if isnothing(γ)
        if deterministic_γ
            γ = Distributions.Dirac(2.38 / sqrt(2 * dimension(ld)))
        else
            γ = Distributions.Uniform(0.8, 1.2)
        end
    elseif isa(γ, Real)
        γ = Distributions.Dirac(γ)
    end

    return de_update{eltype(γ)}(γ, β)
end

function update!(update::de_update, chains::chains_struct, ld::TransformedLogDensity, rng::AbstractRNG, chain::Int)
    x = get_value(chains, chain)
    sampled_chains = sample_chains(chains, rng, chain, 2)
    x₁ = chains.X[sampled_chains[1], :]
    x₂ = chains.X[sampled_chains[2], :]
    if x₁ == x₂
        #just don't update (save the ld eval)
        update_value!(chains, rng, chain, x, -Inf)
    else
        xₚ = x .+ rand(rng, update.γ) .* (x₁ .- x₂) .+ rand(rng, update.β, length(x))
        update_value!(chains, rng, chain, xₚ, logdensity(ld, xₚ))
    end
end

struct snooker_update{T <: Real} <: update_struct
    γ::Distributions.UnivariateDistribution #really we want this to be distribution with type T
end

"""
Set up a Snooker update step.

See doi.org/10.1007/s11222-008-9104-9 for more information.

# Keyword Arguments
- `γ`: The scaling factor for the projection. Can be a `Real`, a `Distributions.UnivariateDistribution`, or `nothing`. If `nothing`, it defaults based on `deterministic_γ`.
- `deterministic_γ`: If `true` and `γ` is `nothing`, sets `γ` to the theoretically optimal `2.38 / sqrt(2)`. If `false`, sets `γ` to `Uniform(0.8, 1.2)`. Defaults to `true`.

# Example
```jldoctest
julia> setup_snooker_update(γ = Uniform(0.1, 2.0))
```
See also [`setup_de_update`](@ref), [`setup_subspace_sampling`](@ref).
"""
function setup_snooker_update(;
        γ::Union{Nothing, Distributions.UnivariateDistribution, Real} = nothing,
        deterministic_γ::Bool = true
)
    if isnothing(γ)
        if deterministic_γ
            γ = Distributions.Dirac(2.38 / sqrt(2))
        else
            γ = Distributions.Uniform(0.8, 1.2)
        end
    elseif isa(γ, Real)
        γ = Distributions.Dirac(γ)
    end

    return snooker_update{eltype(γ)}(γ)
end

function update!(update::snooker_update, chains::chains_struct, ld::TransformedLogDensity, rng::AbstractRNG, chain::Int)
    x = get_value(chains, chain)
    sampled_chains = sample_chains(chains, rng, chain, 3)
    x₁ = chains.X[sampled_chains[1], :]
    x₂ = chains.X[sampled_chains[2], :]
    xₐ = chains.X[sampled_chains[3], :]
    if xₐ == x || x₁ == x₂
        #just don't update (save the ld eval)
        update_value!(chains, rng, chain, x, -Inf)
    else
        e = normalize(xₐ .- x)
        xₚ = x .+ rand(rng, update.γ) .* dot(x₁ .- x₂, e) .* e #really this could be assigned to the next value and just replace if rejected
        update_value!(
            chains, rng, chain, xₚ, logdensity(ld, xₚ),
            (length(x) - 1) * (log(norm(xₐ .- xₚ)) - log(norm(xₐ .- x)))
        )
    end
end

abstract type subspace_sampling_struct <: update_struct end

abstract type subspace_sampling_adaptation_struct end

struct adaptive_subspace_sampling{T <: Real} <: subspace_sampling_adaptation_struct
    L::Vector{Int}
    Δ::Vector{T}
    crs::Vector{T}
end

struct static_subspace_sampling <: subspace_sampling_adaptation_struct
end

struct subspace_sampling <: subspace_sampling_struct
    cr::Distributions.UnivariateDistribution
    δ::Distributions.DiscreteUnivariateDistribution
    ϵ::Distributions.ContinuousUnivariateDistribution
    e::Distributions.ContinuousUnivariateDistribution
    adaptation::subspace_sampling_adaptation_struct
end

struct subspace_sampling_fixed_γ{T <: Real} <: subspace_sampling_struct
    cr::Distributions.UnivariateDistribution
    γ::T
    δ::Distributions.DiscreteUnivariateDistribution
    ϵ::Distributions.ContinuousUnivariateDistribution
    e::Distributions.ContinuousUnivariateDistribution
    adaptation::subspace_sampling_adaptation_struct
end

"""
Set up a Subspace Sampling (DREAM-like) update step.

See doi.org/10.1515/IJNSNS.2009.10.3.273 for more information.

# Keyword Arguments
- `γ`: Fixed scaling factor for the difference vector sum. If `nothing` (default), uses `2.38 / sqrt(2 * δ * d)` where `d` is the number of updated dimensions. If a `Real` is provided, uses that fixed value.
- `cr`: Crossover probability. Can be a `Real` (fixed probability), `nothing` (adaptive probability using `n_cr` values), or a `Distributions.UnivariateDistribution`. Defaults to `nothing`.
- `n_cr`: Number of crossover probabilities to adapt between if `cr` is `nothing`. Defaults to 3.
- `δ`: Number of difference vectors to add. Can be an `Integer` or a `Distributions.DiscreteUnivariateDistribution`. Defaults to `DiscreteUniform(1, 3)`.
- `ϵ`: Distribution for small noise added to the proposal in the selected subspace. Defaults to `Uniform(-1e-4, 1e-4)`.
- `e`: Distribution for multiplicative noise (`e + 1`) applied to the difference vector sum. Defaults to `Normal(0.0, 1e-2)`.

# Example
```jldoctest
julia> setup_subspace_sampling(cr = Beta(1, 2))
```
See also [`setup_de_update`](@ref), [`setup_snooker_update`](@ref).
"""
function setup_subspace_sampling(;
        γ::Union{Nothing, Real} = nothing,
        cr::Union{Real, Distributions.UnivariateDistribution, Nothing} = nothing,
        n_cr::Int = 3,
        δ::Union{Integer, Distributions.DiscreteUnivariateDistribution} = Distributions.DiscreteUniform(
            1, 3),
        ϵ::Distributions.ContinuousUnivariateDistribution = Distributions.Uniform(-1e-4, 1e-4),
        e::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.0, 1e-2)
)
    if isa(δ, Integer)
        δ = Distributions.Dirac(δ)
    end

    adaptation = static_subspace_sampling()

    if isa(cr, Real)
        cr = Distributions.Dirac(cr)
        n_cr = 1
    elseif isnothing(cr)
        cr = Distributions.DiscreteNonParametric(
            collect((1:n_cr) ./ n_cr), repeat([1 / n_cr], n_cr))
        adaptation = adaptive_subspace_sampling(
            zeros(Int, n_cr), zeros(eltype(cr), n_cr), Distributions.params(cr)[1])
    end

    if isnothing(γ)
        subspace_sampling(
            cr,
            δ,
            ϵ,
            e,
            adaptation
        )
    else
        subspace_sampling_fixed_γ(
            cr,
            γ,
            δ,
            ϵ,
            e,
            adaptation
        )
    end
end

function get_γ(rng::AbstractRNG, update::subspace_sampling, δ::Int, d::Int)
    2.38 / sqrt(2 * δ * d)
end

function get_γ(rng::AbstractRNG, update::subspace_sampling_fixed_γ{T}, δ::Int, d::Int) where {T <: Real}
    update.γ
end

function update!(update::subspace_sampling_struct, chains::chains_struct, ld::TransformedLogDensity, rng::AbstractRNG, chain::Int)
    x = get_value(chains, chain)

    #determine how many dimensions to update
    cr = rand(rng, update.cr)
    to_update = rand(rng, length(x)) .< cr
    d = sum(to_update)

    if d == 0
        #just pick one
        to_update = zeros(Bool, length(x))
        to_update[rand(rng, axes(to_update, 1))] = true
        d = 1
    end

    δ = rand(rng, update.δ)

    #generate candidate
    z = copy(x)
    for _ in 1:δ
        #pick to random chains find the difference and add to the candidate
        z[to_update] .+= diff(
            chains.X[sample_chains(chains, rng, chain, 2), to_update], dims = 1)[1, :]
    end

    #add the other parts of the equation
    z[to_update] .= x[to_update] .+ (
        (1 .+ rand(rng, update.e, d)) .* get_γ(rng, update, δ, d) .* z[to_update]
    ) .+
                    rand(rng, update.ϵ, d)

    accepted = update_value!(
        chains, rng, chain, z, logdensity(ld, z)
    )

    update_jumping_distance!(update.adaptation, x, z, chains, accepted, cr)
end

function update_jumping_distance!(
        adaptation::adaptive_subspace_sampling{T}, x::Vector{T}, z::Vector{T}, chains::chains_struct, accepted::Bool, cr) where {T <: Real}
    m = findfirst(cr .== adaptation.crs)
    adaptation.L[m] += 1 #update because this number of attempts
    if accepted
        adaptation.Δ[m] += sum((x .- z) .* (x .- z) ./
                               var(chains.X[chains.current_position, :], dims = 1))
    end
end

function update_jumping_distance!(
        adaptation::static_subspace_sampling, x::Vector{T}, z::Vector{T}, chains::chains_struct{T}, accepted::Bool, cr) where {T <: Real}
    nothing
end

function adapt_update!(update::subspace_sampling_struct, chains::chains_struct)
    if chains.warmup
        update_probability!(update, update.adaptation)
    end
end

function update_probability!(
        update::subspace_sampling_struct, adaptation::adaptive_subspace_sampling)
    #only begin if we've got some attempts at all of the values
    if sum(adaptation.L .== 0) == 0 && sum(adaptation.Δ .== 0) == 0
        Distributions.params(update.cr)[2] .= sum(adaptation.L) .*
                                              (adaptation.Δ ./ adaptation.L) ./
                                              sum(adaptation.Δ)
        #correct probability (this can' be right)
        Distributions.params(update.cr)[2] ./= sum(Distributions.params(update.cr)[2])
    end
end

function update_probability!(
        update::subspace_sampling_struct, adaptation::static_subspace_sampling)
    nothing
end
