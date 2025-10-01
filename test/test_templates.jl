@testset "templates" begin
    #easy problem that uses all the updates
    function ld_normal(x)
        sum(-(x .* x) / 2)
    end
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, 2), ld_normal);

    deMC(ld, 100, memory = false)
    deMCzs(ld, 1000; thin = 2, memory = false)
    DREAMz(ld, 1000; thin = 2, memory = false)
    deMC(ld, 100, memory = true)
    deMCzs(ld, 1000; thin = 2, memory = true)
    DREAMz(ld, 1000; thin = 2, memory = true)
end
