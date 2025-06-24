# Sampling from multimodal distributions

Let say you have a multimodal distribution, for example a mixture of two Gaussians.
`DEMetropolis` implements differential evolution MCMC samplers (including deMC-zs and DREAMz) that are designed to sample from such distributions efficiently.
Roughly these samplers work by generating new proposals based on many separate chains (or a history of sampled chains).
In theory this allows the sampler to easily jump between modes of the distribution.

## Multimodal Distributions

First we need to implement a multimodal distribution. We'll use a mixture of three Gaussians for one parameter and two LogNormal distributions for another, making this a 2D distribution. We can easily implement this using the `Distributions` package, which underpins most of the functionality in `DEMetropolis`.

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

We can also transform our log density function, so we can provide real-valued inputs. This is much easier to work with.

```julia
using TransformedLogDensities, TransformVariables
transformation = as((α = asℝ, β = asℝ₊))
transformed_ld = TransformedLogDensity(transformation, multimodal_ld)
```

## Sampling with DEMetropolis

Now let's use `DEMetropolis` to sample from this multimodal distribution. Here we use the DREAMz sampler, which is well-suited for exploring complex, multimodal spaces. We increase the number of chains to allow the sampler to explore the distribution more effectively.

```julia
using DEMetropolis

dreamz = DREAMz(transformed_ld, 10000; thin = 2, n_chains = 5);
```

Other implementations of the differential evolution MCMC algorithm are available in `DEMetropolis.jl`, such as `deMC` and `deMCzs`, which can be used similarly.

## Custom Scheme

DREAMz can be further customized. For example, we could include snooker updates alongside the DREAMz-like subspace sampling.

You can also modify aspects of the implemented sampling, for example tell DREAMz to use not memory-based sampling with `DREAMz(..., memory = false)`, or you can define your own sampler scheme for more control over the sampling process.

```julia
using DEMetropolis, LogDensityProblems

low_chains_sampler = setup_sampler_scheme(
    setup_subspace_sampling(), # a DREAM-like sampler that uses subspace sampling
    setup_snooker_update(deterministic_γ = false); # a snooker update for better exploration
    w = [0.8, 0.2] # only use snooker 20% of the time
);

initial_state = randn(100, LogDensityProblems.dimension(transformed_ld));

custom = composite_sampler(
    transformed_ld, 10000, 5, true, initial_state, low_chains_sampler, R̂_stopping_criteria(1.05);
    diagnostic_checks = [ld_check()]
);
```

You can also define your own samplers for more specialized use cases.

## Interpreting Results

After running the sampler, you will have a collection of samples from the target distribution. These samples can be used to estimate summary statistics, credible intervals, and to assess the quality of your sampling.

### Assessing Sampler Performance: ESS and R-hat

To evaluate how well your sampler is performing, you can compute the effective sample size (ESS) and the R-hat diagnostic. These metrics help you determine if your chains have mixed well and if your estimates are reliable.

- **Effective Sample Size (ESS):** This measures the number of independent samples your chains are equivalent to. Higher ESS values indicate more reliable estimates.
- **R-hat Diagnostic:** Also known as the Gelman-Rubin statistic, R-hat compares the variance within each chain to the variance between chains. Values close to 1 suggest good mixing and convergence; values much greater than 1 indicate potential problems.

Below is an example of how to compute these diagnostics for two DEMetropolis samplers, `dreamz` and `custom`, using `MCMCChains`:

```julia
using Statistics, MCMCDiagnosticTools

samplers = [dreamz, custom]
sampler_names = ["DREAMz", "Custom Sampler"]

for (sampler, name) in zip(samplers, sampler_names)
    samples = sampler.samples  # shape: (iterations, chains, parameters)
    ess_val = ess(samples)./size(samples, 1)
    rhat_val = maximum(rhat(samples))
    println("$name diagnostics:")
    println("  ESS per iteration: $ess_val")
    println("  R-hat: $rhat_val\n")
end
```

### Summarizing Posterior Samples

Once you have confirmed good mixing and convergence, you can summarize your posterior samples. For each parameter, you may want to compute the median and a credible interval (such as the 90% interval):

```julia
# Example: summarize the DREAMz sampler's posterior
samples = dreamz.samples
n_params = size(samples, 3)

# Flatten the samples across all chains and iterations for each parameter
flat_samples = [vec(samples[:, :, param]) for param in 1:n_params]

for (i, param_samples) in enumerate(flat_samples)
    med = median(param_samples)
    q05, q95 = quantile(param_samples, [0.05, 0.95])
    println("Parameter $i: median = $med, 90% CI = ($q05, $q95)")
end
```

This will print the median and 90% credible interval for each parameter, giving you a summary of the posterior distribution.

For more details, see the [DEMetropolis Documentation](@ref) and the [Customizing your sampler](@ref) section. For general Julia documentation and best practices, refer to the [Julia documentation manual](https://docs.julialang.org/en/v1/manual/documentation/).

