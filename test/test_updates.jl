@testset "Updates" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x) / 2)
    end
    n_pars = 2;
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_normal);

    @testset "regular deMC" begin
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
        @test isa(single_dist.γ, Distributions.Dirac)
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
        @test isa(det.γ, Distributions.Dirac)
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
        @test isa(rel.γ, Distributions.Dirac)
        @test rel == setup_snooker_update(
            γ = 10,
            deterministic_γ = true
        )
        det = setup_snooker_update(
            deterministic_γ = true
        )
        @test isa(det.γ, Distributions.Dirac)
        ran = setup_snooker_update(
            deterministic_γ = false
        )
        @test isa(ran.γ, Distributions.Uniform)
    end

    @testset "subspace" begin
        dist = setup_subspace_sampling(
            γ = nothing,
            δ = 1
        )
        @test isa(dist.δ, Distributions.Dirac)
        @test isa(dist.cr, Distributions.DiscreteNonParametric)
        dist = setup_subspace_sampling(
            γ = 1.0,
            δ = Distributions.Poisson(0.5),
            cr = 0.5
        )
        @test isa(dist.γ, Real)
        @test isa(dist.cr, Distributions.Dirac)
        @test isa(dist.δ, Distributions.Poisson)
    end
end
