module DEMetropolis
export composite_sampler, setup_de_update, setup_snooker_update, setup_subspace_sampling, setup_sampler_scheme, R̂_stopping_criteria, ld_check, acceptance_check
export deMC, deMCzs, DREAMz

using StatsBase: mean, quantile, sample, wsample
using Statistics: var
using Random: default_rng
import Random
using LogDensityProblems: logdensity, dimension
import Distributions
using ProgressMeter: Progress, next!, finish!
using LinearAlgebra: norm, dot, normalize
using MCMCDiagnosticTools: rhat

include("population.jl")
include("updates.jl")
include("stopping.jl")
include("diagnostics.jl")
include("sampler.jl")
include("update_chain.jl")
include("utilities.jl")
include("templates.jl")

end
