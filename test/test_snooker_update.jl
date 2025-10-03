@testset "Snooker Update" begin
    @testset "Snooker Setup" begin
        double_dist = setup_snooker_update(
            γ = Normal(0.8, 1.2)
        )
        @test isa(double_dist.γ_spl, Normal)

        single_dist = setup_snooker_update(
            γ = 0.5
        )
        @test isa(single_dist.γ_spl, Dirac)
        det = setup_snooker_update()
        @test isa(det.γ_spl, Dirac)
        ran = setup_snooker_update(
            deterministic_γ = false
        )
        @test isa(ran.γ_spl, Uniform)
    end

    @testset "Sample using regular Snooker" begin
        rng = MersenneTwister(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_snooker_update()

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = false)

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
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress=false
        )
        @test length(samples) == 100
        @test all(isa(x, DEMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Sample using memory Snooker" begin
        rng = MersenneTwister(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_snooker_update(
            deterministic_γ = false
        )

        sample_result, initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = true)

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
            progress=false
        )
        @test length(samples) == 100
        @test all(isa(x, DEMetropolis.DifferentialEvolutionSample) for x in samples)
    end
end
