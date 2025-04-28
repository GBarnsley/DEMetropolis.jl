using deMCMC
using Test
using TransformVariables, TransformedLogDensities, Random, Distributions

#easy problem that uses all the updates
function ld_normal(x)
    sum(-(x .* x)/2)
end
n_pars = 2;
ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal);
n_chains = 4;
rng = Random.MersenneTwister(1234);
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

diagnostic_checks = [
    ld_check(),
    acceptance_check()
];

@testset "composite sampler" begin
    n_its = 10;
    n_burnin = 10;
    output_mem = composite_sampler(
        ld, n_its, n_chains, true, initial_state, sampler_scheme;
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false,
        diagnostic_checks = diagnostic_checks
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
    epoch_size = 1000;
    warmup_epochs = 2;
    output = composite_sampler(
        ld, epoch_size, n_chains, true, initial_state, sampler_scheme, deMCMC.R̂_stopping_criteria(1.5);
        save_burnt = true, rng = rng, warmup_epochs = warmup_epochs, parallel = false,
        diagnostic_checks = diagnostic_checks
    )
    output_its = size(output.samples, 1)
    output_its_burnt = size(output.burnt_samples, 1)
    @test size(output.samples) == (output_its, n_chains, n_pars)
    @test size(output.ld) == (output_its, n_chains)
    @test size(output.burnt_samples) == (output_its_burnt, n_chains, n_pars)
    @test size(output.burnt_ld) == (output_its_burnt, n_chains)
    @test eltype(output.samples) == eltype(initial_state)
    @test eltype(output.ld) == eltype(initial_state)
end

@testset "regular deMCMC" begin
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

@testset "test rng states" begin
    #should give the same result
    n_its = 100;
    n_burnin = 100;
    rng = Random.MersenneTwister(112);
    output1 = composite_sampler(
        ld, n_its, n_chains, false, initial_state, sampler_scheme;
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )
    rng = Random.MersenneTwister(112);
    output2 = composite_sampler(
        ld, n_its, n_chains, false, initial_state, sampler_scheme;
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )

    @test isequal(output1, output2)
end