# DifferentialEvolutionMetropolis Documentation

Tools for sampling from log-densities using differential evolution algorithms.

See [Sampling from multimodal distributions](@ref) and [Customizing your sampler](@ref) to get started.

This package is built upon [LogDensityProblems.jl](https://www.tamaspapp.eu/LogDensityProblems.jl) so log-densities should be constructed using that package, and can be used with [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) to control the parameter space.

The other key dependency is [Distributions.jl](https://juliastats.org/Distributions.jl). Almost every parameter in proposals given here (see [Proposal Distributions](@ref)) are defined via customizable univariate distributions. Values that are fixed are specified via a [Dirac distribution](https://en.wikipedia.org/wiki/Dirac_delta_function), though in the API these can be specified with any real value. As a *warning* there are minimal checks on the given distributions, it is up to the user to ensure that they are suitable for the given parameter, i.e. there is nothing stopping you from having the noise term in the deMC proposal be centred around 100 instead of 0, or have the distribution for a probability be > 1.
Distributions can optionally be used to define your log-density, as in the examples given here. 

As far as I am aware, there is one other package that implements differential evolution MCMC in Julia, [DifferentialEvolutionMCMC.jl](https://github.com/itsdfish/DifferentialEvolutionMCMC.jl/tree/master).
I opted to implement my own version as I wanted a more flexible API and the subsampling scheme from DREAM. That's not to discredit DifferentialEvolutionMCMC.jl, it has many features this package does not, such as being able to work on optimization problems and parameter blocking.

## Next Steps

A few plans for this package, feel free to suggest features or improvements via [issues](https://github.com/GBarnsley/DifferentialEvolutionMetropolis/issues):
- Implement multi-try and delayed rejection DREAM, I avoided these so far since I have been using these samplers for costly log-densities with relatively few parameters, such as one that solve an ODE.
- Integrate with AbstractMCMC and MCMCChains, potentially not worth the cost since parrallelism in a deMCMC is within chains rather than across chains.

## Contents

```@contents
```

## Functions

### Implemented Sampling Schemes

```@docs
deMC
deMCzs
DREAM
```

### Tools for setting up your own sampler

```@docs
setup_sampler_scheme
composite_sampler
```

### Proposal Distributions

```@docs
setup_de_update
setup_snooker_update
setup_subspace_sampling
```

### Stopping Criteria

```@docs
R̂_stopping_criteria
```

### Diagnostics Checks with Resampling

```@docs
ld_check
acceptance_check
```

## Index

```@index
```