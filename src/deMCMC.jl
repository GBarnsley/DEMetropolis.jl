module deMCMC
export composite_sampler, setup_de_update, setup_snooker_update, setup_subspace_sampling, sampler_scheme_multi, R̂_stopping_criteria, ld_check, acceptance_check
import StatsBase, Statistics, Random, TransformedLogDensities, LogDensityProblems, Distributions, Logging, ProgressMeter, LinearAlgebra, MCMCDiagnosticTools

include("population.jl")
include("updates.jl")
include("stopping.jl")
include("diagnostics.jl")
include("sampler.jl")
include("update_chain.jl")
include("utilities.jl")

end

#using .deMCMC
#using TransformVariables, Distributions, TransformedLogDensities, Plots, Random
#function ld_raw(x)
#    # normal mixture
#    sum(Distributions.logpdf(Distributions.MixtureModel(Distributions.Normal[Distributions.Normal(-5.0, 1), Distributions.Normal(5.0, 1)], [1/3, 2/3]), x))
#end
#
#ld = TransformedLogDensities.TransformedLogDensity(as(Array, 8), ld_raw)
#rng = Random.MersenneTwister(555);
#n_its = 10000;
#n_chains = 8;
#memory = true;
#parallel = false;
#initial_state = randn(n_chains * 20, 8);
#sampler_scheme = deMCMC.sampler_scheme_multi(
#    [1.0, 1.0, 0.1, 0.1, 1.0, 1.0],
#    [
#        setup_de_update(ld, deterministic_γ = false),
#        setup_de_update(ld, deterministic_γ = true),
#        setup_snooker_update(deterministic_γ = false),
#        setup_snooker_update(deterministic_γ = true),
#        setup_subspace_sampling(),
#        setup_subspace_sampling(γ = 1.0, cr = Distributions.DiscreteNonParametric([0.4, 1.0], [0.5, 0.5]))
#    ]
#)
#sampler_scheme = deMCMC.sampler_scheme_multi(
#    [0.1, 0.2, 0.7],
#    [
#        setup_snooker_update(deterministic_γ = false),
#        setup_subspace_sampling(γ = 1.0),
#        setup_subspace_sampling(),
#    ]
#)
#diagnostics = [
#    ld_check(),
#    acceptance_check()
#]
#output = composite_sampler(
#    ld, n_its, n_chains, memory, initial_state, sampler_scheme, deMCMC.R̂_stopping_criteria(1.1);
#    save_burnt = true, rng = rng, parallel = true, diagnostic_checks = diagnostics
#)
#
#
#output.sampler_scheme.updates[2]
#output.sampler_scheme.updates[3]
#
#mean(output.samples[:, :, :], dims = (1, 2))
#mean(output.samples[:, :, :])
#plot(cat(output.burnt_samples[:, :, 1], output.samples[:, :, 1], dims = 1))