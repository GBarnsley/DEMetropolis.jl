# DEMetropolis

[![Build Status](https://github.com/GBarnsley/DEMetropolis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GBarnsley/DEMetropolis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://GBarnsley.github.io/DEMetropolis.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://GBarnsley.github.io/DEMetropolis.jl/dev)


## Overview

This package implements various differential evolution MCMC samplers in the Julia language, with some minor tweaks to simplify parallelism.

These samplers excel at sampling from multimodal distributions. The package is designed to be flexible and extensible, allowing users to customize the sampling process by modifying the implemented samplers or defining your own samplers and sampler scheme.

Please start with the [documentation](https://GBarnsley.github.io/DEMetropolis.jl/stable).

## Example

```
using DEMetropolis, TransformedLogDensities, TransformVariables

function ld_normal(x)
    sum(-(x .* x)/2)
end
ld = TransformedLogDensity(as(Array, 4), ld_normal);

DREAMz(ld, 1000)
```

## Bibliography

Braak, C.J.F.T. A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces. Stat Comput 16, 239–249 (2006). https://doi.org/10.1007/s11222-006-8769-1

Braak, C.J.F.T., Vrugt, J.A. Differential Evolution Markov Chain with snooker updater and fewer chains. Stat Comput 18, 435–446 (2008). https://doi.org/10.1007/s11222-008-9104-9

Vrugt, J.A., Braak, C.J.F.T., Diks, C.G.H., Robinson, B.A., Hyman, J.M., Higdon, D. Accelerating Markov Chain Monte Carlo Simulation by Differential Evolution with Self-Adaptive Randomized Subspace Sampling. International Journal of Nonlinear Sciences and Numerical Simulation 10, no. 3, 273-290 (2009). https://doi.org/10.1515/IJNSNS.2009.10.3.273