@testset "test rng states" begin
    model = IsotropicNormalModel([-5.0, 5.0])

    # DE + Snooker composite
    de_sampler = setup_sampler_scheme(
        setup_de_update(),
        setup_de_update(n_dims = LogDensityProblems.dimension(model)),
        setup_snooker_update(deterministic_γ = false),
        setup_snooker_update(deterministic_γ = true),
        setup_subspace_sampling(),
        setup_subspace_sampling(γ = 1.0),
        setup_subspace_sampling(cr = DiscreteNonParametric([0.5, 1.0], [0.5, 0.5]))
    )

    de_sampler_old = deepcopy(de_sampler)

    #should be equal after deepcopy
    @test de_sampler.update_weights == de_sampler_old.update_weights
    for i in eachindex(de_sampler.updates)
        if isa(de_sampler.updates[i], DEMetropolis.AbstractDifferentialEvolutionSubspaceSampler)
            @test de_sampler.updates[i].cr_spl == de_sampler_old.updates[i].cr_spl
            @test de_sampler.updates[i].n_cr == de_sampler_old.updates[i].n_cr
            @test de_sampler.updates[i].δ_spl == de_sampler_old.updates[i].δ_spl
            @test de_sampler.updates[i].ϵ_spl == de_sampler_old.updates[i].ϵ_spl
            @test de_sampler.updates[i].e_spl == de_sampler_old.updates[i].e_spl
        else
            @test de_sampler.updates[i] == de_sampler_old.updates[i]
        end
    end

    #should give the same result
    n_its = 1000
    n_warmup = 0
    seed = 112
    output1 = sample(
        backwards_compat_rng(seed),
        AbstractMCMC.LogDensityModel(model),
        de_sampler,
        n_its;
        num_warmup = n_warmup,
        parallel = false,
        progress = false
    )

    output2 = sample(
        backwards_compat_rng(seed),
        AbstractMCMC.LogDensityModel(model),
        de_sampler,
        n_its;
        num_warmup = n_warmup,
        parallel = true,
        progress = false
    )

    #should still be equal after deepcopy
    @test de_sampler.update_weights == de_sampler_old.update_weights
    for i in eachindex(de_sampler.updates)
        if isa(de_sampler.updates[i], DEMetropolis.AbstractDifferentialEvolutionSubspaceSampler)
            @test de_sampler.updates[i].cr_spl == de_sampler_old.updates[i].cr_spl
            @test de_sampler.updates[i].n_cr == de_sampler_old.updates[i].n_cr
            @test de_sampler.updates[i].δ_spl == de_sampler_old.updates[i].δ_spl
            @test de_sampler.updates[i].ϵ_spl == de_sampler_old.updates[i].ϵ_spl
            @test de_sampler.updates[i].e_spl == de_sampler_old.updates[i].e_spl
        else
            @test de_sampler.updates[i] == de_sampler_old.updates[i]
        end
    end

    equality_x = [isequal(output1[i].x, output2[i].x) for i in 1:length(output1)]
    equality_ld = [isequal(output1[i].ld, output2[i].ld) for i in 1:length(output1)]
    @test all(equality_x)
    @test all(equality_ld)
end
