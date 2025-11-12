@testset "Test correctness with MCMCTesting.jl" begin
    #basic example from (https://arxiv.org/pdf/2001.06465) inspired by MCMCTesting.jl

    struct NormalNormalModel
        σ::Float64
        σ_ϵ::Float64
        y::Float64
    end

    function sample_joint(rng::Random.AbstractRNG, model::NormalNormalModel)
        θ = rand(rng, Normal(0, model.σ), 2)
        y = rand(rng, Normal(sum(θ), model.σ_ϵ))
        θ, y
    end

    function complete_conditional(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
        μ = σ² / (σ²_ϵ + σ²) * (y - θ)
        σ = 1 / sqrt(1 / σ²_ϵ + 1 / σ²)
        Normal(μ, σ)
    end

    function gibbs_sample_θ(rng::Random.AbstractRNG, model::NormalNormalModel, θ::Vector{Float64})
        θ = copy(θ)
        y = model.y
        σ² = model.σ^2
        σ²_ϵ = model.σ_ϵ^2
        θ[1] = rand(rng, complete_conditional(θ[2], σ², σ²_ϵ, y))
        θ[2] = rand(rng, complete_conditional(θ[1], σ², σ²_ϵ, y))
        θ
    end

    function LogDensityProblems.dimension(model::NormalNormalModel)
        return 2
    end

    function LogDensityProblems.logdensity(model::NormalNormalModel, θ::Vector{Float64})
        return logpdf(Normal(0, model.σ), θ[1]) +
            logpdf(Normal(0, model.σ), θ[2]) +
            logpdf(Normal(θ[1] + θ[2], model.σ_ϵ), model.y)
    end

    LogDensityProblems.capabilities(model::NormalNormalModel) = LogDensityProblems.LogDensityOrder{0}()

    function test_for_correctness(rng, update, memory, base_model, attempts, steps_per_attempt, sig; pt = false)
        if memory
            n_chains = 4
            n_extra = 4 * LogDensityProblems.dimension(base_model) - n_chains
            n_hot_chains = 0
        elseif pt
            n_chains = 4
            n_hot_chains = 3 * LogDensityProblems.dimension(base_model) - n_chains
            n_extra = 0
        else
            n_chains = 2 * LogDensityProblems.dimension(base_model)
            n_extra = 0
            n_hot_chains = 0
        end

        M_sampler = Distributions.sampler(DiscreteUniform(1, steps_per_attempt))
        #rank based on ld so its less stuff to store
        initial_positions = [Vector{Float64}(undef, LogDensityProblems.dimension(base_model)) for i in 1:(n_chains + n_extra + n_hot_chains)]
        complete_chain = Array{Float64, 2}(undef, steps_per_attempt, n_chains)
        ordinal_ranks = Array{Int, 2}(undef, attempts, n_chains)
        sum_of_rank_of_ranks = zeros(Int, n_chains)

        for attempt in 1:attempts
            M = rand(rng, M_sampler)
            initial_positions[1], y = sample_joint(rng, base_model)
            new_model = NormalNormalModel(base_model.σ, base_model.σ_ϵ, y)
            for i in 2:length(initial_positions)
                #gibbs sample to get initial positions with correct dependence on y
                initial_positions[i] .= gibbs_sample_θ(rng, new_model, initial_positions[i - 1])
            end

            #randomly permute the initial position so its a bit fairer
            initial_positions[1:n_chains] = initial_positions[randperm(rng, n_chains)]

            complete_chain[M, :] .= [LogDensityProblems.logdensity(new_model, initial_positions[i]) for i in 1:n_chains]
            if M < steps_per_attempt
                complete_chain[(M + 1):steps_per_attempt, :] .= sample(
                    rng,
                    AbstractMCMC.LogDensityModel(new_model),
                    update,
                    steps_per_attempt - M;
                    initial_position = initial_positions,
                    memory = memory,
                    N₀ = n_extra,
                    n_chains = n_chains,
                    progress = false,
                    chain_type = DifferentialEvolutionOutput,
                    silent = true,
                    n_hot_chains = n_hot_chains
                ).ld
            end

            if M > 1
                complete_chain[1:(M - 1), :] .= reverse(
                    sample(
                        rng,
                        AbstractMCMC.LogDensityModel(new_model),
                        update,
                        M - 1;
                        initial_position = initial_positions,
                        memory = memory,
                        N₀ = n_extra,
                        n_chains = n_chains,
                        progress = false,
                        chain_type = DifferentialEvolutionOutput,
                        silent = true,
                        n_hot_chains = n_hot_chains
                    ).ld; dims = 1
                )
            end

            for chain in 1:n_chains
                ordinal_ranks[attempt, chain] = ordinalrank(complete_chain[:, chain])[M]
            end

            sum_of_rank_of_ranks .+= ordinalrank(ordinal_ranks[attempt, :])
        end

        #test for uniformity within iterations
        friedman_statistic = ((12 / (attempts * n_chains * (n_chains + 1))) * sum(sum_of_rank_of_ranks .^ 2)) - 3 * attempts * (n_chains + 1)

        B = 10000
        running_count = 0
        res = zeros(Float64, attempts, n_chains)
        ranks = zeros(Int, attempts, n_chains)
        soror = similar(sum_of_rank_of_ranks)
        for _ in 1:B
            res .= randn(rng, attempts, n_chains)
            for i in 1:n_chains
                ranks[:, i] .= ordinalrank(res[:, i])
            end
            soror .= ordinalrank(ranks[1, :])
            for i in 2:attempts
                soror .+= ordinalrank(ranks[i, :])
            end
            running_count += (((12 / (attempts * n_chains * (n_chains + 1))) * sum(soror .^ 2)) - 3 * attempts * (n_chains + 1)) ≥ friedman_statistic
        end

        p_values_within = running_count /= B

        #across iterations, assuming they are all independent
        test_statistic = 0.0
        expected = length(ordinal_ranks[:]) / steps_per_attempt
        for v in 1:steps_per_attempt
            n_v = count(==(v), ordinal_ranks[:])
            test_statistic += (n_v - expected)^2 / expected
        end
        p_values_across = ccdf(Chisq(steps_per_attempt - 1), test_statistic)

        pass = true
        if p_values_within < sig
            @warn "Failed rank-uniformity between DE-chains test with p-value $p_values_within"
            W = friedman_statistic / (attempts * (n_chains - 1))
            if W > 0.01
                @warn "Large coefficient of concordance W = $W"
                pass = false
            else
                @warn "But small coefficient of concordance W = $W."
            end
        end
        if p_values_across < sig
            @warn "Failed rank-uniformity across all iterations test with p-value $p_values_across"
            pass = false
        end

        return pass
    end

    rng = backwards_compat_rng(1234)
    attempts = 300
    steps_per_attempt = 6000
    base_model = NormalNormalModel(1.0, 0.5, 0.0)
    sig = 0.05
    @testset "Without memory" begin
        @test test_for_correctness(rng, setup_de_update(n_dims = 2), false, base_model, attempts, steps_per_attempt, sig)
        @test test_for_correctness(rng, setup_snooker_update(), false, base_model, attempts, steps_per_attempt, sig)
        @test test_for_correctness(rng, setup_subspace_sampling(), false, base_model, attempts, steps_per_attempt, sig)
    end
    @testset "With memory" begin
        @test test_for_correctness(rng, setup_de_update(n_dims = 2), true, base_model, attempts, steps_per_attempt, sig)
        @test test_for_correctness(rng, setup_snooker_update(), true, base_model, attempts, steps_per_attempt, sig)
        @test test_for_correctness(rng, setup_subspace_sampling(), true, base_model, attempts, steps_per_attempt, sig)
    end
    @testset "With pt" begin
        @test test_for_correctness(rng, setup_de_update(n_dims = 2), false, base_model, attempts, steps_per_attempt, sig; pt = true)
        @test test_for_correctness(rng, setup_snooker_update(), false, base_model, attempts, steps_per_attempt, sig; pt = true)
        @test test_for_correctness(rng, setup_subspace_sampling(), false, base_model, attempts, steps_per_attempt, sig; pt = true)
    end

end
