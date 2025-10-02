@testset "templates" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x) / 2)
    end
    n_dims = 2
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_dims), ld_normal);

    @testset "non memory runs" begin
        deMC(ld, 100, memory = false)
        deMCzs(ld, 1000; thin = 2, memory = false)
        DREAMz(ld, 1000; thin = 2, memory = false)
    end
    @testset "memory runs" begin
        deMC(ld, 100, memory = true)
        deMCzs(ld, 1000; thin = 2, memory = true)
        DREAMz(ld, 1000; thin = 2, memory = true)
    end
    @testset "parameter simplifying" begin
        deMC(ld, 100, memory = false, γ₁ = 0.5, γ₂ = 0.5)
        deMCzs(ld, 1000; thin = 2, memory = false, p_snooker = 0.0)
        DREAMz(ld, 1000; thin = 2, memory = true, p_γ₂ = 0.0)
    end
    @testset "warnings" begin
        #check for warning
        ld_wide = TransformedLogDensities.TransformedLogDensity(as(Array, 5), ld_normal);
        @test_logs @test_logs (:warn,) deMC(ld_wide, 1000; thin = 2, n_chains = 4)
        @test_logs @test_logs (:warn,) DREAMz(ld_wide, 1000; thin = 2, n_chains = 4)
        @test_logs @test_logs (:warn,) deMCzs(ld_wide, 1000; thin = 2, n_chains = 4)
    end
    @testset "parallel" begin
        DREAMz(ld, 1000; thin = 2, memory = true, parallel = true)
    end
    @testset "initial_state building" begin
        deMC(ld, 100, initial_state = randn(n_dims * 2, n_dims))
        @test_logs @test_logs (:warn,) deMC(ld, 100, initial_state = randn(n_dims * 2 - 1, n_dims))
        @test_logs @test_logs (:warn,) deMC(ld, 100, initial_state = randn(n_dims * 2 + 1, n_dims))
        @test_throws ErrorException deMC(ld, 100, initial_state = randn(n_dims * 2, n_dims -
                                                                                    1))
    end
end
