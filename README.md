# DEMetropolis

[![Build Status](https://github.com/GBarnsley/DEMetropolis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GBarnsley/DEMetropolis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://GBarnsley.github.io/DEMetropolis.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://GBarnsley.github.io/DEMetropolis.jl/dev)


## Overview

This package implements various differential evolution MCMC samplers in the Julia language based on , with some minor tweaks to simplify parallelism.

These samplers excel at sampling from multimodal distributions. The package is designed to be flexible and extensible, allowing users to customize the sampling process by modifying the implemented samplers or defining your own samplers and sampler scheme.

Please start with the [documentation](https://GBarnsley.github.io/DEMetropolis.jl/stable).

## Example

```
using DEMetropolis, TransformedLogDensities, TransformVariables

function ld_normal(x)
    sum(-(x .* x)/2)
end
ld = TransformedLogDensity(as(Array, 4), ld_normal);

DREAM(ld, 1000)
```

## Bibliography

Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

Betancourt, M. (2016). Diagnosing suboptimal cotangent disintegrations in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1604.00695](https://arxiv.org/abs/1604.00695).

Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis. : CRC Press.

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models.

Hoffman, M. D., & Gelman, A. (2014). The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623.

McElreath, R. (2018). Statistical rethinking: A Bayesian course with examples in R and Stan. Chapman and Hall/CRC.


LINKS TO PAPERS

SIMPLE EXAMPLE

REFERNCE TO DOCUMENATION (TUTORIAL AND CUSTOMISATION)