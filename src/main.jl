module deMCMC
export composite_sampler, setup_de_update, setup_snooker_update, sampler_scheme_multi, R̂_stopping_criteria
import StatsBase, Random, TransformedLogDensities, LogDensityProblems, Distributions, Logging, ProgressMeter, LinearAlgebra, StatsBase, MCMCDiagnosticTools

include("population.jl")
include("updates.jl")
include("stopping.jl")
include("sampler.jl")
include("update_chain.jl")
include("diagnostics.jl")
include("utilities.jl")

end

#using .deMCMC
#using TransformVariables, Distributions, TransformedLogDensities, Plots, Random
#function ld_raw(x)
#    # normal mixture
#    sum(Distributions.logpdf(Distributions.MixtureModel(Distributions.Normal[Distributions.Normal(-5.0, 1), Distributions.Normal(5.0, 1)], [1/3, 2/3]), x))
#end
#ld = TransformedLogDensities.TransformedLogDensity(as(Array, 8), ld_raw)
#rng = Random.MersenneTwister(1234);
#n_its = 10000;
#n_chains = 4;
#memory = true;
#parallel = true;
#initial_state = randn(n_chains, 8);
#sampler_scheme = deMCMC.sampler_scheme_multi(
#    [1.0, 1.0, 0.1, 0.1],
#    [
#        setup_de_update(ld, deterministic_γ = false),
#        setup_de_update(ld, deterministic_γ = true),
#        setup_snooker_update(deterministic_γ = false),
#        setup_snooker_update(deterministic_γ = true)
#    ]
#)
#output = composite_sampler(
#    ld, n_its, n_chains, memory, initial_state, sampler_scheme, deMCMC.R̂_stopping_criteria(1.1);
#    save_burnt = true, rng = rng, parallel = true
#)
#
#plot(cat(output.burnt_samples[:, :, 1], output.samples[:, :, 1], dims = 1))
#mean(output.samples[:, :, :], dims = (1, 2))
#mean(output.samples[:, :, :])