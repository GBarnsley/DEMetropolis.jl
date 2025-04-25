module deMCMC
export composite_sampler, setup_de_update, setup_snooker_update, sampler_scheme_multi
import StatsBase, Random, TransformedLogDensities, LogDensityProblems, Distributions, Logging, ProgressMeter, LinearAlgebra, StatsBase, MCMCDiagnosticTools

include("population.jl")
include("updates.jl")
include("sampler.jl")
include("update_chain.jl")
include("diagnostics.jl")
include("utilities.jl")

end

#using .deMCMC
#using TransformVariables, Distributions, TransformedLogDensities
#function ld_raw(x)
#    # normal mixture
#    sum(Distributions.logpdf(Distributions.MixtureModel(Distributions.Normal[Distributions.Normal(-5.0, 1), Distributions.Normal(5.0, 1)], [1/3, 2/3]), x))
#end
#ld = TransformedLogDensities.TransformedLogDensity(as(Array, 8), ld_raw)
#rng = Random.MersenneTwister(1234);
#n_its = 1000;
#n_chains = 50;
#n_burnin = 1000;
#memory = true;
#parallel = true;
#initial_state = randn(n_chains, 8);
#sampler_scheme = sampler_scheme_multi(
#    [1.0, 1.0, 0.1, 0.1],
#    [
#        setup_de_update(ld, deterministic_γ = false),
#        setup_de_update(ld, deterministic_γ = true),
#        setup_snooker_update(deterministic_γ = false),
#        setup_snooker_update(deterministic_γ = true)
#    ]
#)
#