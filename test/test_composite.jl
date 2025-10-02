@testset "composite sampler" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x) / 2)
    end
    n_pars = 2;
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal);
    n_chains = 4;
    rng = Random.MersenneTwister(1234);
    initial_state = randn(rng, n_chains, n_pars);
    sampler_scheme = setup_sampler_scheme(
        setup_de_update(ld, deterministic_γ = false),
        setup_de_update(ld, deterministic_γ = true),
        setup_snooker_update(deterministic_γ = false),
        setup_snooker_update(deterministic_γ = true),
        setup_subspace_sampling(),
        setup_subspace_sampling(γ = 1.0),
        setup_subspace_sampling(γ = 1.0, cr = 0.5);
        w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    diagnostic_checks = [
        ld_check(),
        acceptance_check()
    ];

    @testset "iterative" begin
        n_its = 10
        n_burnin = 10
        output_mem = composite_sampler(
            ld, n_its, n_chains, true,
            cat(initial_state, initial_state, initial_state, dims = 1), sampler_scheme;
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

    @testset "iterative_simple_update" begin
        n_its = 10
        n_burnin = 10
        output_mem = composite_sampler(
            ld, n_its, n_chains, true,
            cat(initial_state, initial_state, initial_state, dims = 1),
            setup_sampler_scheme(setup_subspace_sampling());
            save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false,
            diagnostic_checks = diagnostic_checks
        )
        output = composite_sampler(
            ld, n_its, n_chains, false, initial_state,
            setup_sampler_scheme(setup_subspace_sampling());
            save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
        )
        output_nob = composite_sampler(
            ld, n_its, n_chains, false, initial_state,
            setup_sampler_scheme(setup_subspace_sampling());
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

    @testset "until converged" begin
        #easy problem that uses all the updates
        epoch_size = 1000
        warmup_epochs = 2
        output = composite_sampler(
            ld, epoch_size, n_chains, true,
            cat(initial_state, initial_state, initial_state, dims = 1),
            sampler_scheme, R̂_stopping_criteria(1.5);
            save_burnt = true, rng = rng, warmup_epochs = warmup_epochs, parallel = false,
            diagnostic_checks = diagnostic_checks
        )
        output_nob = composite_sampler(
            ld, epoch_size, n_chains, true,
            cat(initial_state, initial_state, initial_state, dims = 1),
            sampler_scheme, R̂_stopping_criteria(1.5);
            save_burnt = false, rng = rng, warmup_epochs = warmup_epochs, parallel = false,
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
        @test !(:burnt_samples in keys(output_nob))
    end

    @testset "sampler errors" begin
        @test_throws ErrorException setup_sampler_scheme(
            setup_de_update(ld, deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(γ = 1.0),
            w = [1.0, 1.0, 1.0, 1.0]
        )
        @test_throws ErrorException setup_sampler_scheme(
            setup_de_update(ld, deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(γ = 1.0),
            w = [1.0, 1.0, -1.0]
        )
    end
    @testset "initial_state errors" begin
        n_its = 10
        n_burnin = 10
        @test_throws ErrorException composite_sampler(
            ld, n_its, n_chains, false, randn(rng, n_chains + 1, n_pars), sampler_scheme;
            save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
        )
        @test_throws ErrorException composite_sampler(
            ld, n_its, 3, false, randn(rng, 3, n_pars), sampler_scheme;
            save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
        )
        @test_throws ErrorException composite_sampler(
            ld, n_its, 3, true, randn(rng, 3 + 2, n_pars), sampler_scheme;
            save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
        )
        @test_throws ErrorException composite_sampler(
            ld, n_its, n_chains, false, randn(rng, n_chains, n_pars + 1), sampler_scheme;
            save_burnt = true, rng = rng, n_burnin = n_burnin, parallel = false
        )
    end
end
