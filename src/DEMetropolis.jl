module DEMetropolis
export composite_sampler, setup_de_update, setup_snooker_update, setup_subspace_sampling,
       setup_sampler_scheme, R̂_stopping_criteria, ld_check, acceptance_check
export deMC, deMCzs, DREAMz

import StatsBase: mean, quantile, sample, wsample
import Statistics: var
import Random: default_rng, AbstractRNG
import Random
import LogDensityProblems: logdensity, dimension
import TransformedLogDensities: TransformedLogDensity
import Distributions
import ProgressMeter: Progress, next!, finish!
import LinearAlgebra: norm, dot, normalize
import MCMCDiagnosticTools: rhat

include("population.jl")
include("updates.jl")
include("stopping.jl")
include("diagnostics.jl")
include("sampler.jl")
include("update_chain.jl")
include("utilities.jl")
include("templates.jl")

end
