# Sampling from multimodal distributions

Let say you have a multimodal distribution, for example a mixture of two Gaussians.
`DEMetropolis` implements differential evolution MCMC samplers (including deMC-zs and DREAMz) that are designed to sample from such distributions efficiently.
Roughly these samplers work by generating new proposals based on many separate chains (or a history of sampled chains).
In theory this allows the sampler to easily jump between modes of the distribution.

## Multimodal Distributions

First we need to implement a multimodal distribution. We'll use a mixture of three Gaussians for one parameter and two LogNormal distributions for another, making this a 2D distribution. We can easily implement this using the `Distributions` package, which underpins most of the functionality in `DEMetropolis`.

```@example MMSampler
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

```@example MMSampler
using TransformedLogDensities, TransformVariables
transformation = as((α = asℝ, β = asℝ₊))
transformed_ld = TransformedLogDensity(transformation, multimodal_ld)
```

## Sampling with DEMetropolis

Now let's use `DEMetropolis` to sample from this multimodal distribution. Here we use the DREAMz sampler, which is well-suited for exploring complex, multimodal spaces. We increase the number of chains to allow the sampler to explore the distribution more effectively.

```@example MMSampler
using DEMetropolis, AbstractMCMC, Random

model = AbstractMCMC.LogDensityModel(transformed_ld)

Random.seed!(1234)

# Sample using DREAMz with adaptive stopping based on convergence
dreamz = DREAMz(model, 10000; n_chains = 6, progress = true);
```

Other implementations of the differential evolution MCMC algorithm are available in `DEMetropolis.jl`, such as `deMC` and `deMCzs`, which can be used similarly.

## Custom Scheme

DREAMz can be further customized. For example, we could include snooker updates alongside the DREAMz-like subspace sampling.

You can also modify aspects of the implemented sampling, for example tell DREAMz to use non-memory-based sampling with `DREAMz(...; memory = false)`, or you can define your own sampler scheme for more control over the sampling process.

```@example MMSampler
# Create a custom sampler scheme combining different update types
custom_sampler = setup_sampler_scheme(
    setup_subspace_sampling(), # a DREAM-like sampler that uses subspace sampling
    setup_snooker_update(deterministic_γ = false), # a snooker update for better exploration
    setup_de_update(); # standard DE update
    w = [0.6, 0.2, 0.2] # weights for each update type
);

# Sample using AbstractMCMC.sample with custom stopping criteria
custom_result = sample(
    model,
    custom_sampler,
    r̂_stopping_criteria;
    check_every = 1000,
    maximum_R̂ = 1.05,
    n_chains = 4,
    memory = true,
    parallel = true,
    num_warmup = 10000
);
```

You can also define your own samplers for more specialized use cases by extending the abstract types.

## Interpreting Results

After running the sampler, you will have a collection of samples from the target distribution. These samples can be used to estimate summary statistics, credible intervals, and to assess the quality of your sampling.

### Assessing Sampler Performance: ESS and R-hat

To evaluate how well your sampler is performing, you can compute the effective sample size (ESS) and the R-hat diagnostic. These metrics help you determine if your chains have mixed well and if your estimates are reliable.

- **Effective Sample Size (ESS):** This measures the number of independent samples your chains are equivalent to. Higher ESS values indicate more reliable estimates.
- **R-hat Diagnostic:** Also known as the Gelman-Rubin statistic, R-hat compares the variance within each chain to the variance between chains. Values close to 1 suggest good mixing and convergence; values much greater than 1 indicate potential problems.

Below is an example of how to compute these diagnostics for the DEMetropolis samplers using `MCMCDiagnosticTools`:

```@example MMSampler
using Statistics, MCMCDiagnosticTools

# Compute diagnostics for DREAMz results
ess_val = ess(dreamz.samples) ./ size(dreamz.samples, 1)
rhat_val = maximum(rhat(dreamz.samples))

println("DREAMz diagnostics:")
println("  ESS per iteration: $ess_val")
println("  R-hat: $rhat_val")
```

### Summarizing Posterior Samples

Once you have confirmed good mixing and convergence, you can summarize your posterior samples. For each parameter, you may want to compute the median and a credible interval (such as the 90% interval):

```@example MMSampler
# Example: summarize the custom sampler's posterior
custom_results = process_outputs(custom_result)

# Flatten the samples across all chains and iterations for each parameter
n_params = size(custom_results.samples, 3)

reduced_samples = custom_results.samples[(size(custom_results.samples, 1) ÷ 2):size(custom_results.samples, 1), :, :]
transformed_samples = [
    transform(transformation, vec(reduced_samples[iteration, chain, :])) for 
        iteration in axes(reduced_samples, 1),
        chain in axes(reduced_samples, 2)
][:]
flat_samples = [vec([transformed_samples[i][param] for i in axes(transformed_samples, 1)]) for param in 1:n_params]

med = median(flat_samples[1])
q05, q95 = quantile(flat_samples[1], [0.05, 0.95])
println("Parameter α: median = $med, 90% CI = ($q05, $q95)")
println("True α: median = $(median(α_mixed_dist)), 90% CI = ($(quantile(α_mixed_dist, 0.05)), $(quantile(α_mixed_dist, 0.95)))")

med = median(flat_samples[2])
q05, q95 = quantile(flat_samples[2], [0.05, 0.95])
println("Parameter β: median = $med, 90% CI = ($q05, $q95)")
println("True β: median = $(median(β_mixed_dist)), 90% CI = ($(quantile(β_mixed_dist, 0.05)), $(quantile(β_mixed_dist, 0.95)))")
```

For more details, see the [DEMetropolis Documentation](@ref) and [Customizing your sampler](@ref).
