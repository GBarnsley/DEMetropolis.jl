# Sampling from multimodal distributions

Let say you have a multimodal distribution, for example a mixture of two Gaussians.
`DEMetropolis` implements differential evolution MCMC samplers (including deMC-zs and DREAM) that are designed to sample from such distributions efficiently.
Roughly these samplers work by generating new proposals based on many separate chains (or a history of sampled chains).
In theory this allows the sampler to easily jump between modes of the distribution.

## Multimodal Distributions

First we need to implement a multimodal distribution, we'll use a mixture of two Gaussians for one parameter and two Exponential distributions for another, making this a 2D distribution.
We can easily implement this using the `Distributions` package, which underpins most of the functionality in `DEMetropolis`.

```julia
using Distributions

α_mixed_dist = MixtureModel([
    Normal(-5.0, 0.5),
    Normal(0.0, 0.5),
    Normal(5.0, 0.5)
], [1/4, 1/4, 1/2]);

β_mixed_dist = MixtureModel([
    LogNormal(-0.5, 0.5),
    LogNormal(1.75, 0.25)
], [1/6, 5/6]);

function multimodal_ld(θ)
    (; α, β) = θ
    logpdf(α_mixed_dist, α) +
        logpdf(β_mixed_dist, β)
end
```

Let's visualize this distribution using the `Plots` package.
In the first dimension:
```julia
using Plots
α = -10:0.1:10

y = [exp(multimodal_ld((α = xi, β = 1.0))) for xi in α]

plot(α, y, xlabel="α (First Dimension)", ylabel="Density", legend = false)
```

In the second dimension:
```julia
β = 0:0.1:10

y = [exp(multimodal_ld((α = 0.0, β = xi))) for xi in β]

plot(β, y, xlabel="β (Second Dimension)", ylabel="Density", legend = false)
```

All together:
```julia
y = [exp(multimodal_ld((α = x1i, β = x2i))) for x2i in β, x1i in α]

heatmap(α, β, y, xlabel="α", ylabel="β")
```

We can also transform our log density function, so we can provide real-valued inputs. This is much easier to work with.

```julia
using TransformedLogDensities, TransformVariables
transformation = as((α = asℝ, β = asℝ₊))
transformed_ld = TransformedLogDensity(transformation, multimodal_ld)
```

## Sampling with MCMC and HMC

We can attempt to sample from this using `AdaptiveMCMC.jl` which implements adaptive Metropolis-Hastings algorithms.

```julia
using AdaptiveMCMC, Statistics, LogDensityProblems

#can't use adaptive_rwm with transformed log densities, so we use the log density directly
log_p(x) = LogDensityProblems.logdensity(transformed_ld, x)

adaptive_mh = map(_ -> adaptive_rwm(zeros(2), log_p, 10000 + (10000 * 10); algorithm=:am, b = 10000, thin = 10), 1:3);
```

We can also sample from this distribution using Hamiltonian Monte Carlo (HMC) with `DynamicHMC.jl`.
This approach is often more efficient for high-dimensional distributions, but it requires the gradient of the log density function.

```julia
using DynamicHMC, LogDensityProblemsAD, DynamicHMC.Diagnostics, Random;

∇P = ADgradient(:ForwardDiff, transformed_ld);

dynamic_hmc = [mcmc_with_warmup(Random.default_rng(), ∇P, 10000; reporter = ProgressMeterReport()) for _ in 1:3];
```

## Using DREAM

We can use an implementation of the [`DREAM`](@ref) algorithm to sample from this distribution.
Here we increase the number of chains (the default is two times the number of dimensions) to 10, which allows the sampler to explore the multimodal distribution more effectively. DREAM also underperforms here, so we will thin the samples, storing every 10th sample.

```julia
using DEMetropolis

dream = DREAM(transformed_ld, 10000; thin = 2, n_chains = 50, parallel = true);
```

Other implementations of the differential evolution MCMC algorithm are available in `DEMetropolis.jl`, such as [`deMC`](@ref) and [`deMCzs`](@ref), which can be used similarly.

## Custom Scheme

DREAM was fairly slow to converge, an easy way to dramatically increase the performance is to use a memory based sampler. In the previous example DREAM only samples based on the current status of the chains, but we can also sample based on the history of the chains. This reduces the chance of chains getting stuck, meaning we can also use fewer chains.

We could tell DREAM to use a memory based sampling with `DREAM(..., memory = true)`, but we could also define our own sampler scheme, giving us more control over the sampling process.

```julia
using DEMetropolis

low_chains_sampler = setup_sampler_scheme(
    setup_subspace_sampling(), # a DREAM like sampler that uses subspace sampling
    setup_snooker_update(deterministic_γ = false); # a snooker update for better exploration
    w = [0.8, 0.2] # only use snooker 20% of the time
);

initial_state = randn(100, LogDensityProblems.dimension(transformed_ld));

custom = composite_sampler(
    transformed_ld, 10000, 5, true, initial_state, low_chains_sampler, R̂_stopping_criteria(1.05);
    diagnostic_checks = [ld_check()]
);
```

This achieves convergence to (a stricter criteria) within a single epoch, while DREAM couldn't converge within 20 epochs.

Other samplers are available, see `DEMetropolis Documentation`(@ref) for more details.
You can also define your own samplers, see `Customizing your sampler`(@ref) for more details.

## Compare distributions

Now let's compare the performance of the different samplers in terms of effective sample size (ESS) and R-hat diagnostic. We'll use the `MCMCChains` and `DataFrames` packages for this purpose.

```julia
using DataFrames, MCMCChains, Statistics

# Collect samples from each sampler (assuming each returns a vector of chains or samples)
chains = [
    cat([chain.X for chain in adaptive_mh]..., dims=3),
    cat([chain.posterior_matrix for chain in dynamic_hmc]..., dims=3),
    cat([chain.X for chain in dream]..., dims=3),
    cat([chain.X for chain in custom]..., dims=3)
]

sampler_names = ["Adaptive MH", "Dynamic HMC", "DREAM", "Custom Sampler"]

# Compute ESS and R-hat for each sampler
ess_values = [ess(permutedims(chain, (2, 3, 1))) for chain in chains]
rhat_values = [maximum(rhat(Chains(chain))) for chain in chains]

results_df = DataFrame(
    sampler = sampler_names,
    ess = ess_values,
    rhat = rhat_values
)

println(results_df)
```

## Trace plots and summary statistics

To visualize the samples and check convergence, you can plot the trace and compute summary statistics:

```julia
using Plots

# Example: plot trace for α and β from the first sampler
posterior = chains[1]  # shape: (parameters, iterations, chains)

plot(posterior[1, :, :], xlabel = "iteration", ylabel = "α", title = "Trace plot for α")
plot(posterior[2, :, :], xlabel = "iteration", ylabel = "β", title = "Trace plot for β")
```

To summarize the posterior:

```julia
# Compute median and 90% credible interval for each parameter
posterior_flat = reshape(posterior, size(posterior, 1), :)
for i in 1:size(posterior_flat, 1)
    med = median(posterior_flat[i, :])
    q05, q95 = quantile(posterior_flat[i, :], [0.05, 0.95])
    println("Parameter $i: median = $med, 90% CI = ($q05, $q95)")
end
```

## Heatmaps of the samples

You can also visualize the joint distribution of the samples:

```julia
# Scatter plot of α vs β for all samples
scatter(vec(posterior[1, :, :]), vec(posterior[2, :, :]), xlabel = "α", ylabel = "β", title = "Joint posterior samples")
```

## Summary and next steps

This tutorial introduced how to use `DEMetropolis` and related packages to sample from challenging multimodal distributions. You learned how to:

- Define and visualize a multimodal target distribution
- Use adaptive Metropolis-Hastings, Hamiltonian Monte Carlo, and differential evolution MCMC (DREAM) samplers
- Customize your own sampler scheme for improved performance
- Compare samplers using effective sample size and R-hat diagnostics
- Visualize and summarize posterior samples

For more details, see the [DEMetropolis Documentation](@ref) and the [Customizing your sampler](@ref) section. For general Julia documentation and best practices, refer to the [Julia documentation manual](https://docs.julialang.org/en/v1/manual/documentation/).

Happy sampling!
