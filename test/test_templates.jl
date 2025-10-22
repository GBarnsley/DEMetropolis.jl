@testset "templates" begin
    ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))

    @testset "non memory runs" begin
        deMC(ld, 100, memory = false, save_burnt = true)
        deMCzs(ld, 1000; thin = 2, memory = false, epoch_limit = 3)
        DREAMz(ld, 1000; thin = 2, memory = false, epoch_limit = 3)
    end
    @testset "memory runs" begin
        deMC(ld, 100, memory = true)
        deMCzs(ld, 1000; thin = 2, memory = true, epoch_limit = 3)
        DREAMz(ld, 1000; thin = 2, memory = true, epoch_limit = 3)
    end
    @testset "parameter simplifying" begin
        deMC(ld, 100, memory = false, γ₁ = 0.5, γ₂ = 0.5)
        deMCzs(ld, 1000; thin = 2, memory = false, p_snooker = 0.0, epoch_limit = 3)
        DREAMz(ld, 1000; thin = 2, memory = true, p_γ₂ = 0.0, epoch_limit = 3)
    end
    @testset "annealing and parallel tempering" begin
        DREAMz(ld, 1000; thin = 2, memory = false, epoch_limit = 3, n_hot_chains = 10)
        deMC(ld, 100, memory = false; annealing = true)
        deMCzs(ld, 1000; thin = 2, memory = true, epoch_limit = 3, annealing = true)
    end
    @testset "warnings" begin
        #check for warning
        ld_wide = AbstractMCMC.LogDensityModel(IsotropicNormalModel([
            -5.0, 5.0, 0.0, 0.0, 0.0]));
        @test_logs @test_logs (:warn,) deMC(ld_wide, 1000; thin = 2, n_chains = 4)
    end
    @testset "parallel" begin
        DREAMz(ld, 1000; thin = 2, memory = true, parallel = true, epoch_limit = 3)
    end
    @testset "initial_state building" begin
        n_dims = LogDensityProblems.dimension(ld.logdensity)
        deMC(ld, 100, initial_state = [randn(n_dims) for _ in 1:(n_dims * 2)])
        @test_logs (:warn,) deMC(ld, 100, initial_state = [randn(n_dims)
                                                           for _ in 2:(n_dims * 2)])
        @test_logs (:warn,) deMC(
            ld, 100, initial_state = [randn(n_dims) for _ in 0:(n_dims * 2)], memory = true)
        @test_throws ErrorException deMC(ld, 100, initial_state = [randn(n_dims - 1)
                                                                   for _ in 1:(n_dims * 2)])

        deMC(ld, 100, initial_state = [randn(n_dims) for _ in 1:10],
            n_chains = 5, memory = false)
        @test_throws ErrorException deMC(
            ld, 100, initial_state = [randn(n_dims) for _ in 1:10],
            n_chains = 5, memory = false, n_hot_chains = 2)
        @test_logs (:warn,) deMC(
            ld, 100, initial_state = [randn(n_dims) for _ in 0:10], N₀ = 5, memory = true)
    end
end
