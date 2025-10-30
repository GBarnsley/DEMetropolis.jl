module DEMetropolis
export setup_de_update, setup_snooker_update, setup_subspace_sampling, setup_sampler_scheme
export step, step_warmup, fix_sampler, fix_sampler_state
export r̂_stopping_criteria, process_outputs
export deMC, deMCzs, DREAMz

import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution,
                      ContinuousUnivariateDistribution, DiscreteNonParametricSampler
import Distributions: Sampleable, Discrete, Continuous, Univariate, sampler, params
import Distributions: Dirac, Uniform, DiscreteUniform, Normal, Categorical, AliasTable,
                      DiscreteNonParametric
import Distributions

import LogDensityProblems: logdensity, dimension
import StatsBase: wsample
import StatsBase
import LinearAlgebra: norm, normalize, dot
import Random: AbstractRNG, default_rng
import Random
import AbstractMCMC: LogDensityModel, AbstractSampler, step, step_warmup, AbstractModel,
                     sample
import MCMCDiagnosticTools: rhat

abstract type AbstractDifferentialEvolutionSampler <: AbstractSampler end

abstract type AbstractDifferentialEvolutionState{T, A, L, V, VV} end

abstract type AbstractDifferentialEvolutionAdaptiveState{T} end

abstract type AbstractDifferentialEvolutionTemperatureLadder{T} end

include("temperature.jl")
include("chains.jl")
include("differential_evolution_update.jl")
include("snooker_update.jl")
include("subspace_update.jl")
include("subspace_adaptive_update.jl")
include("composite_sampler.jl")
include("utilities.jl")
include("convergence.jl")
include("templates.jl")

end
