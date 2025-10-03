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
        setup_subspace_sampling(cr = DiscreteUniform(1, 2))
    )

    #should give the same result
    n_its = 1000
    n_warmup = 1000
    rng = Random.MersenneTwister(112)
    output1 = sample(
        rng,
        AbstractMCMC.LogDensityModel(model),
        de_sampler,
        n_its;
        num_warmup = n_warmup,
        progress=false
    )

    rng = Random.MersenneTwister(112)
    output2 = sample(
        rng,
        AbstractMCMC.LogDensityModel(model),
        de_sampler,
        n_its;
        num_warmup = n_warmup,
        parallel=true,
        progress=false
    )

    equality_x = [isequal(output1[i].x, output2[i].x) for i in 1:length(output1)]
    equality_ld = [isequal(output1[i].ld, output2[i].ld) for i in 1:length(output1)]
    @test all(equality_x)
    @test all(equality_ld)
end
