module deMCMC
export run_deMCMC
import Random, TransformedLogDensities, LogDensityProblems, Logging, ProgressMeter, LinearAlgebra, StatsBase, MCMCDiagnosticTools

include("deMCMC_live.jl")
include("deMCMC.jl")
include("diagnostics.jl")
include("epoch.jl")
include("update_chain.jl")

end

#using MCMCDiagnosticTools, Plots, Distributions, BenchmarkTools
#function ld(x)
#    # normal mixture
#    sum(logpdf(MixtureModel(Normal[Normal(-5.0, 1), Normal(5.0, 1)], [1/3, 2/3]), x))
#end
#pars = 20;
#
#output = deMCMC.run_deMCMC(ld, pars; n_its = 1000, n_burn = 100000, n_thin = 1, n_chains = 200, deterministic_γ = false, memory = true, parallel = true, check_chain_epochs = 5);
##output = deMCMC.run_deMCMC_live(ld, pars; n_its = 1000, check_every = 5000, n_chains = 100, deterministic_γ = false, parallel = true, save_burnt = true);
#
#mean(output.samples, dims = (1, 2))
#mean(output.samples, dims = (1, 2, 3))
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
#plot(
#    output.samples[:, :, 2]
#)