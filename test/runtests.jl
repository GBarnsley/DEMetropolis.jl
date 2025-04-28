using deMCMC
using Test
using TransformVariables, TransformedLogDensities, Random, Distributions

@testset "composite sampler" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x)/2)
    end
    n_pars = 2
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal)
    rng = Random.MersenneTwister(1234);
    n_its = 10;
    n_chains = 4;
    n_burnin = 10;
    initial_state = randn(n_chains, n_pars);
    sampler_scheme = sampler_scheme_multi(
        [1.0, 1.0, 1.0, 1.0],
        [
            setup_de_update(ld, deterministic_γ = false),
            setup_de_update(ld, deterministic_γ = true),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true)
        ]
    );
    output_mem = composite_sampler(
        ld, n_its, n_chains, true, initial_state, sampler_scheme;
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )
    output = composite_sampler(
        ld, n_its, n_chains, false, initial_state, sampler_scheme;
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )
    output_nob = composite_sampler(
        ld, n_its, n_chains, false, initial_state, sampler_scheme;
        save_burnt = false, rng = rng, n_burnin = n_burnin, parallel = false
    )
    @test size(output.samples) == (n_its, n_chains, n_pars)
    @test size(output.ld) == (n_its, n_chains)
    @test size(output.burnt_samples) == (n_its, n_chains, n_pars)
    @test size(output.burnt_ld) == (n_its, n_chains)
    @test size(output_mem.samples) == (n_its, n_chains, n_pars)
    @test size(output_mem.ld) == (n_its, n_chains)
    @test size(output_mem.burnt_samples) == (n_its, n_chains, n_pars)
    @test size(output_mem.burnt_ld) == (n_its, n_chains)
    @test !(:burnt_samples in keys(output_nob))
    @test eltype(output.samples) == eltype(initial_state)
    @test eltype(output.ld) == eltype(initial_state)
end

@testset "composite sampler until converged" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x)/2)
    end
    n_pars = 2
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal)
    rng = Random.MersenneTwister(1234);
    epoch_size = 1000;
    n_chains = 4;
    warm_up_epochs = 10;
    initial_state = randn(n_chains, n_pars);
    sampler_scheme = sampler_scheme_multi(
        [1.0, 1.0, 1.0, 1.0],
        [
            setup_de_update(ld, deterministic_γ = false),
            setup_de_update(ld, deterministic_γ = true),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true)
        ]
    );
    output = composite_sampler(
        ld, n_its, n_chains, true, initial_state, sampler_scheme, deMCMC.R̂_stopping_criteria(1.5);
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )
    @test size(output.samples) == (n_its, n_chains, n_pars)
    @test size(output.ld) == (n_its, n_chains)
    @test size(output.burnt_samples) == (n_its, n_chains, n_pars)
    @test size(output.burnt_ld) == (n_its, n_chains)
    @test eltype(output.samples) == eltype(initial_state)
    @test eltype(output.ld) == eltype(initial_state)
end

@testset "regular deMCMC" begin
    function ld_normal(x)
        sum(-(x .* x)/2)
    end
    n_pars = 2
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal);
    double_dist = setup_de_update(
        ld;
        γ = Distributions.Normal(0.8, 1.2),
        β = Distributions.Uniform(-1e-4, 1e-4),
        deterministic_γ = false
    )
    @test isa(double_dist.γ, Distributions.Normal)
    @test isa(double_dist.β, Distributions.Uniform)
    @test double_dist == setup_de_update(
        ld;
        γ = Distributions.Normal(0.8, 1.2),
        β = Distributions.Uniform(-1e-4, 1e-4),
        deterministic_γ = true
    )

    single_dist = setup_de_update(
        ld;
        γ = 0.5,
        β = Distributions.Beta(1e-4, 1e-4),
        deterministic_γ = false
    )
    @test isa(single_dist.γ, Real)
    @test isa(single_dist.β, Distributions.Beta)
    @test single_dist == setup_de_update(
        ld;
        γ = 0.5,
        β = Distributions.Beta(1e-4, 1e-4),
        deterministic_γ = true
    )
    det = setup_de_update(
        ld;
        deterministic_γ = true
    )
    @test isa(det.γ, Real)
    @test isa(det.β, Distributions.Uniform)
    ran = setup_de_update(
        ld;
        deterministic_γ = false
    )
    @test isa(ran.γ, Distributions.Uniform)
    @test isa(ran.β, Distributions.Uniform)
end

@testset "snooker" begin
    dist = setup_snooker_update(
        γ = Distributions.Normal(0.8, 1.2),
        deterministic_γ = false
    )
    @test isa(dist.γ, Distributions.Normal)
    @test dist == setup_snooker_update(
        γ = Distributions.Normal(0.8, 1.2),
        deterministic_γ = false
    )

    rel = setup_snooker_update(
        γ = 10,
        deterministic_γ = false
    )
    @test isa(rel.γ, Real)
    @test rel == setup_snooker_update(
        γ = 10,
        deterministic_γ = true
    )
    det = setup_snooker_update(
        deterministic_γ = true
    )
    @test isa(det.γ, Real)
    ran = setup_snooker_update(
        deterministic_γ = false
    )
    @test isa(ran.γ, Distributions.Uniform)
end