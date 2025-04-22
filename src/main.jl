module deMCMC
export run_deMCMC
import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter, LinearAlgebra, StatsBase, MCMCDiagnosticTools

include("deMCMC_live.jl")
include("deMCMC.jl")
include("DREAM.jl")
include("diagnostics.jl")
include("epoch.jl")
include("update_chain.jl")

#using Distributions
#(; n_its, n_chains, rng, save_burnt, fitting_parameters) = run_DREAM_defaults()
#n_chains = 20;
#function ld(x)
#    # normal mixture
#    sum(logpdf(MixtureModel(Normal[Normal(-5.0, 1), Normal(5.0, 1)], [1/3, 2/3]), x))
#end
#dim = 10;

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

