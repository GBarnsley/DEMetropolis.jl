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

    function test_for_correctness!(
            rng, p_values, sum_of_rank_of_ranks, update, base_model, sample_kwargs, n, L,
            M_sampler, initial_positions, complete_chain, ordinal_ranks
        )
        d = LogDensityProblems.dimension(base_model)
        n_chains = sample_kwargs.n_chains
        for attempt in 1:n
            M = rand(rng, M_sampler)
            initial_positions[1], y = sample_joint(rng, base_model)
            new_model = NormalNormalModel(base_model.σ, base_model.σ_ϵ, y)
            for i in 2:length(initial_positions)
                #gibbs sample to get initial positions with correct dependence on y
                initial_positions[i] .= gibbs_sample_θ(rng, new_model, initial_positions[i - 1])
            end

            #randomly permute the initial position so its a bit fairer
            initial_positions[1:n_chains] = initial_positions[randperm(rng, n_chains)]

            if M < L
                res = sample(
                    rng,
                    AbstractMCMC.LogDensityModel(new_model),
                    update,
                    L - M + 1;
                    initial_position = initial_positions,
                    sample_kwargs...
                )
                complete_chain[M:L, :, 1:(end - 1)] .= res.samples
                complete_chain[M:L, :, end] .= res.ld
            else
                complete_chain[M, :, 1:(end - 1)] .= cat(initial_positions[1:n_chains]...; dims = 2)'
                complete_chain[M, :, end] .= [LogDensityProblems.logdensity(new_model, initial_positions[i]) for i in 1:n_chains]
            end

            if M > 1
                res = sample(
                    rng,
                    AbstractMCMC.LogDensityModel(new_model),
                    update,
                    M;
                    initial_position = initial_positions,
                    sample_kwargs...
                )
                #append but ignore the value at M, its already on there
                if res.samples[1, :, :] != complete_chain[M, :, 1:(end - 1)]
                    error("Something went wrong with chaining the results together")
                end
                complete_chain[1:(M - 1), :, 1:(end - 1)] .= reverse(res.samples; dims = 1)[1:(end - 1), :, :]
                complete_chain[1:(M - 1), :, end] .= reverse(res.ld; dims = 1)[1:(end - 1), :]
            end

            for chain in 1:n_chains
                for parameter in 1:(d + 1)
                    ordinal_ranks[attempt, chain, parameter] = ordinalrank(complete_chain[:, chain, parameter])[M]
                end
            end

            for parameter in 1:(d + 1)
                sum_of_rank_of_ranks[:, parameter] .+= ordinalrank(ordinal_ranks[attempt, :, parameter])
            end
        end

        #across iterations, assuming they are all independent, we'll use sum_of_rank_of_ranks to test within iterations later
        p_values .= 0.0 #holds test statistic for now
        expected = n * (n_chains) / L
        for v in 1:L
            for p in 1:(d + 1)
                n_v = count(==(v), ordinal_ranks[1:n, :, p][:])
                p_values[p] += (n_v - expected)^2 / expected
            end
        end
        #convert to p-values
        p_values .= ccdf.(Chisq(L - 1), p_values)

        return nothing
    end

    function friedman_statistic(sum_of_rank_of_ranks, attempts, n_chains)
        return ((12 / (attempts * n_chains * (n_chains + 1))) * sum(sum_of_rank_of_ranks .^ 2)) - 3 * attempts * (n_chains + 1)
    end

    function sequential_testing(
            rng, update, base_model, L, α, k, Δ, initial_n, memory::Bool;
            d = LogDensityProblems.dimension(base_model),
            n_chains = memory ? 5 : 3 * d,
            n_hot_chains = 0,
            N₀ = memory ? max(5 * d - n_chains, n_chains + n_hot_chains) : 0
        )
        #setup for model
        sample_kwargs = (
            memory = memory, N₀ = N₀, n_chains = n_chains, progress = false,
            chain_type = DifferentialEvolutionOutput, silent = true, n_hot_chains = n_hot_chains,
        )

        #setup for checks
        M_sampler = Distributions.sampler(DiscreteUniform(1, L))
        #rank based on values and ld so its less stuff to store
        n_positions = n_chains + N₀ + n_hot_chains
        initial_positions = [Vector{Float64}(undef, d) for i in 1:n_positions]
        complete_chain = Array{Float64, 3}(undef, L, n_chains, d + 1)
        ordinal_ranks = Array{Int, 3}(undef, Δ * initial_n, n_chains, d + 1)
        sum_of_rank_of_ranks = zeros(Int, n_chains, d + 1)

        β = α / k
        γ = β^(1 / k)
        n = initial_n

        p_values = Vector{Float64}(undef, d + 1)

        #sequential test for uniformity across iterations
        for i in 1:k
            test_for_correctness!(
                rng, p_values, sum_of_rank_of_ranks, update, base_model, sample_kwargs, n, L,
                M_sampler, initial_positions, complete_chain, ordinal_ranks
            )
            q = minimum(p_values) * (d + 1)
            if q ≤ β
                @warn "Failed rank-uniformity across all iterations test with p-value $(minimum(p_values))"
                return false
            elseif q > γ + β
                break
            else
                β = β / γ
                if i == 1
                    n *= Δ
                end
            end
        end

        #test for uniformity within iterations
        #since we can only rank 1:n_chains we can calculate the number of attempts as
        attempts = Int(sum(sum_of_rank_of_ranks, dims = 1)[1, 1] / sum(1:n_chains))

        statistic = maximum([friedman_statistic(sum_of_rank_of_ranks[:, p], attempts, n_chains) for p in 1:(d + 1)])

        # good approx when n_chains > 4 and attempts > 15
        p_value_within = ccdf(Chisq(n_chains - 1), statistic)

        if p_value_within < α
            @warn "Failed rank-uniformity between DE-chains test with p-value $p_value_within"
            W = statistic / (attempts * (n_chains - 1))
            if W > 0.01
                @warn "Large coefficient of concordance W = $W"
                return false
            else
                @warn "But small coefficient of concordance W = $W."
            end
        end

        return true
    end

    rng = backwards_compat_rng(1234)
    initial_n = 50
    Δ = 10
    L = 100
    base_model = NormalNormalModel(1.0, 0.5, 0.0)
    α = 0.05
    k = 10

    composite_update = setup_sampler_scheme(
        setup_de_update(n_dims = LogDensityProblems.dimension(base_model)),
        setup_snooker_update(),
        setup_subspace_sampling()
    )

    @testset "Without memory" begin
        @test sequential_testing(rng, setup_de_update(n_dims = LogDensityProblems.dimension(base_model)), base_model, L, α, k, Δ, initial_n, false)
        @test sequential_testing(rng, setup_snooker_update(), base_model, L, α, k, Δ, initial_n, false)
        @test sequential_testing(rng, setup_subspace_sampling(), base_model, L, α, k, Δ, initial_n, false)
        @test sequential_testing(rng, composite_update, base_model, L, α, k, Δ, initial_n, false)
    end
    @testset "With memory" begin
        @test sequential_testing(rng, setup_de_update(n_dims = LogDensityProblems.dimension(base_model)), base_model, L, α, k, Δ, initial_n, true)
        @test sequential_testing(rng, setup_snooker_update(), base_model, L, α, k, Δ, initial_n, true)
        @test sequential_testing(rng, setup_subspace_sampling(), base_model, L, α, k, Δ, initial_n, true)
        @test sequential_testing(rng, composite_update, base_model, L, α, k, Δ, initial_n, true)
    end
    @testset "With pt" begin
        @test sequential_testing(rng, setup_de_update(n_dims = LogDensityProblems.dimension(base_model)), base_model, L, α, k, Δ, initial_n, false; n_hot_chains = 5)
        @test sequential_testing(rng, setup_snooker_update(), base_model, L, α, k, Δ, initial_n, false; n_hot_chains = 5)
        @test sequential_testing(rng, setup_subspace_sampling(), base_model, L, α, k, Δ, initial_n, false; n_hot_chains = 5)
        @test sequential_testing(rng, composite_update, base_model, L, α, k, Δ, initial_n, false; n_hot_chains = 5)
    end

end
