@testset "Subspace Update" begin
    @testset "Subspace Setup" begin
         dist = setup_subspace_sampling(
            γ = nothing,
            δ = 1
        )
        @test isa(dist.δ_spl, Dirac)
        @test isa(dist.cr_spl, Distributions.AliasTable)
        dist = setup_subspace_sampling(
            γ = 1.0,
            δ = Poisson(0.5),
            cr = 0.5
        )
        @test isa(dist.γ, Real)
        @test isa(dist.cr_spl, Dirac)
        @test isa(dist.δ_spl, Distributions.PoissonCountSampler)
    end

    @testset "Sample using regular Subspace" begin
        rng = MersenneTwister(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling(cr = DiscreteUniform(1, 5))

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = false, adapt = false)

        @test isa(sample_result, DEMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DEMetropolis.DifferentialEvolutionState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state)

        @test isa(sample_result, DEMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DEMetropolis.DifferentialEvolutionState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress=false,
            adapt=false
        )
        @test length(samples) == 100
        @test all(isa(x, DEMetropolis.DifferentialEvolutionSample) for x in samples)
    end

     @testset "Sample which will likely fail to pick a dimension atleast once" begin
        rng = MersenneTwister(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling(cr = Categorical([0.95, repeat([0.005], 10)...]))

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = false, adapt = false)

        @test isa(sample_result, DEMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DEMetropolis.DifferentialEvolutionState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})
    end

    @testset "Sample using memory Subspace" begin
        rng = MersenneTwister(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling(cr = DiscreteUniform(1, 5), γ = 1.0)

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = true, adapt = false)

        @test isa(sample_result, DEMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DEMetropolis.DifferentialEvolutionStateMemory)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state)

        @test isa(sample_result, DEMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DEMetropolis.DifferentialEvolutionStateMemory)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress=false,
            adapt=false
        )
        @test length(samples) == 100
        @test all(isa(x, DEMetropolis.DifferentialEvolutionSample) for x in samples)
    end
end
