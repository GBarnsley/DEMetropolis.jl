using DEMetropolis
using Test
using TransformVariables, TransformedLogDensities, Random, Distributions
using Aqua

@testset "DEMetropolis.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(DEMetropolis)
    end

    # Include tests for specific distributions
    include("test_composite.jl")
    include("test_updates.jl")
    include("test_rng.jl")
    include("test_templates.jl")
    include("test_diagnostics.jl")

    #if VERSION ≥ v"1.11"
    #    @testset "JET" begin
    #        JET.test_package(DEMetropolis)
    #    end
    #end
end
