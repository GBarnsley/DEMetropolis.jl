@testset "test rng states" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x) / 2)
    end
    n_pars = 2;
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal);
    n_chains = 4;
    rng = Random.MersenneTwister(1234);
    initial_state = randn(rng, n_chains, n_pars);

    #should give the same result
    n_its = 100
    n_burnin = 100
    rng = Random.MersenneTwister(112)
    output1 = composite_sampler(
        ld, n_its,
        n_chains,
        false,
        initial_state,
        setup_sampler_scheme(
            setup_de_update(ld, deterministic_γ = false),
            setup_de_update(ld, deterministic_γ = true),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0)
        );
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )
    rng = Random.MersenneTwister(112)
    output2 = composite_sampler(
        ld, n_its,
        n_chains,
        false,
        initial_state,
        setup_sampler_scheme(
            setup_de_update(ld, deterministic_γ = false),
            setup_de_update(ld, deterministic_γ = true),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0)
        );
        save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
    )

    @test isequal(output1.sampler_scheme.updates[1], output2.sampler_scheme.updates[1])
    @test isequal(output1.sampler_scheme.updates[2], output2.sampler_scheme.updates[2])
    @test isequal(output1.sampler_scheme.updates[3], output2.sampler_scheme.updates[3])
    @test isequal(output1.sampler_scheme.updates[4], output2.sampler_scheme.updates[4])
    #bug to report?
    #@test isequal(output1.sampler_scheme, output2.sampler_scheme)
    #@test isequal(output1.sampler_scheme.updates[5], output2.sampler_scheme.updates[5])
    #@test isequal(output1.sampler_scheme.updates[6].adaptation, output2.sampler_scheme.updates[6].adaptation)
    @test isequal(output1.sampler_scheme.updates[6].adaptation.L,
        output2.sampler_scheme.updates[6].adaptation.L)
    @test isequal(output1.sampler_scheme.updates[6].adaptation.Δ,
        output2.sampler_scheme.updates[6].adaptation.Δ)
    @test isequal(output1.sampler_scheme.updates[6].adaptation.crs,
        output2.sampler_scheme.updates[6].adaptation.crs)

    @test isequal(output1.samples, output2.samples)
    @test isequal(output1.ld, output2.ld)
    @test isequal(output1.burnt_samples, output2.burnt_samples)
    @test isequal(output1.burnt_ld, output2.burnt_ld)
end
