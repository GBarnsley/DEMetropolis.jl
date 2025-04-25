module deMCMC
export composite_sampler
import StatsBase, Random, TransformedLogDensities, LogDensityProblems, Distributions, Logging, ProgressMeter, LinearAlgebra, StatsBase, MCMCDiagnosticTools

include("population.jl")
include("updates.jl")
include("sampler.jl")
include("update_chain.jl")
include("diagnostics.jl")
include("utilities.jl")

using TransformVariables
function ld_raw(x)
    # normal mixture
    sum(Distributions.logpdf(Distributions.MixtureModel(Distributions.Normal[Distributions.Normal(-5.0, 1), Distributions.Normal(5.0, 1)], [1/3, 2/3]), x))
end
ld = TransformedLogDensities.TransformedLogDensity(as(Array, 8), ld_raw)
rng = Random.MersenneTwister(1234);
n_its = 1000;
n_chains = 50;
n_burnin = 1000;
memory = false;
parallel = true;
initial_state = randn(n_chains, 8);
sampler_scheme = sampler_scheme_multi(
    [1.0, 1.0, 0.1, 0.1],
    [
        setup_de_update(ld, deterministic_γ = false),
        setup_de_update(ld, deterministic_γ = true),
        setup_snooker_update(deterministic_γ = false),
        setup_snooker_update(deterministic_γ = true)
    ]
)

end

#import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter, LinearAlgebra, StatsBase, MCMCDiagnosticTools
#using MCMCDiagnosticTools, Plots, Distributions, BenchmarkTools
#function ld(x)
#    # normal mixture
#    sum(logpdf(MixtureModel(Normal[Normal(-5.0, 1), Normal(5.0, 1)], [1/3, 2/3]), x))
#end
#pars = 10;
#
##output = deMCMC.run_deMCMC(ld, pars; 
##    n_its = 1000, n_burn = 10000, n_thin = 1, n_chains = 100, deterministic_γ = false, memory = true, parallel = true, check_chain_epochs = 3, save_burnt = true, N₀ = 20
##);
#output = deMCMC.run_deMCMC_live(ld, pars; n_its = 1000, check_every = 10000, n_chains = 25, deterministic_γ = false, parallel = true, save_burnt = true, N₀ = 20, check_acceptance = true);
#size(output.samples)
#mean(output.samples, dims = (1, 2))
#
#mean(MixtureModel(Normal[Normal(-5.0, 1), Normal(5.0, 1)], [1/3, 2/3]))
#
#println(ess_rhat(output.samples))
#plot(
#    output.ld
#)
#plot(
#    output.samples[:, :, 1]
#)

