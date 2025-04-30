# DifferentialEvolutionMetropolis Documentation

Tools for sampling from log-densities using differential evolution algorithms.

See [Sampling from multimodal distributions](@ref) and [Customizing your sampler](@ref) to get started.

There is

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