module DEMetropolis
export setup_de_update, step


import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: Sampleable, Discrete, Continuous, Univariate, sampler
import Distributions: Dirac, Uniform

import LogDensityProblems: logdensity, dimension
import StatsBase: sample
import Random: AbstractRNG, default_rng
import AbstractMCMC: LogDensityModel, AbstractSampler, step

abstract type AbstractDifferentialEvolutionSampler <: AbstractSampler end

abstract type AbstractDifferentialEvolutionState end

include("differential_evolution_update.jl")

end
