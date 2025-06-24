# Customizing your sampler

This document describes how to extend `DEMetropolis.jl` with your own custom components. You can define custom stopping criteria, diagnostic checks, and proposal distributions (updates).

## Custom Stopping Criteria

To create a custom stopping criterion, you need to define a new struct that is a subtype of `DEMetropolis.stopping_criteria_struct` and then implement the `DEMetropolis.stop_sampling` method for your new type.

The `stop_sampling` method has the following signature:
`stop_sampling(stopping_criteria::YourCriteria, chains::chains_struct, sample_from::Int, last_iteration::Int)`

- `stopping_criteria`: An instance of your custom stopping criterion struct.
- `chains`: The `chains_struct` containing the state of all chains. You can use `DEMetropolis.population_to_samples` to extract samples.
- `sample_from`: The iteration number at which sampling (post-warmup) began.
- `last_iteration`: The total number of iterations completed so far (per chain).

The function should return `true` if sampling should stop, and `false` otherwise.

Here is an example of a stopping criterion that stops sampling after a maximum number of iterations has been reached.

```julia
using DEMetropolis

# Define the struct for the stopping criterion
struct MaxIterationsStopping <: DEMetropolis.stopping_criteria_struct
    max_iters::Int
end

# Implement the stop_sampling function
function DEMetropolis.stop_sampling(criterion::MaxIterationsStopping, chains, sample_from, last_iteration)
    if length(DEMetropolis.get_sampling_indices(sample_from, last_iteration)) >= criterion.max_iters
        println("Reached maximum iterations, stopping.")
        return true
    end
    return false
end
```

## Custom Diagnostic Checks

You can implement custom diagnostic checks that are run during the warmup/burn-in phase. This is useful for monitoring chain behavior and potentially correcting issues on the fly, like resetting outlier chains.

To create a custom diagnostic, define a struct that subtypes `DEMetropolis.diagnostic_check_struct` and implement `DEMetropolis.run_diagnostic_check!` for it.

The method signature is:
`run_diagnostic_check!(chains, diagnostic_check::YourCheck, rngs, current_iteration::Int)`

- `chains`: The `chains_struct`, which can be modified within the function.
- `diagnostic_check`: An instance of your custom diagnostic struct.
- `rngs`: A vector of random number generators, one for each chain.
- `current_iteration`: The current warmup iteration number.

Here is an example of a simple diagnostic that just prints a message. A more advanced check could, for example, calculate acceptance rates and reset chains that are not mixing well.

```julia
using DEMetropolis

# Define the struct for the diagnostic check
struct MyCustomDiagnostic <: DEMetropolis.diagnostic_check_struct
    bad_number::Float64
end

# Implement the run_diagnostic_check! function
function DEMetropolis.run_diagnostic_check!(chains, check::MyCustomDiagnostic, rngs, current_iteration)
    println("Running custom diagnostic at warmup iteration: ", current_iteration)
    # This is where you would add logic to inspect and modify chains

    # As an example we'll resample all chains that include a bad number (since these are floats it'll probably never happen)
    X = DEMetropolis.population_to_samples(chains, DEMetropolis.get_sampling_indices(1, current_iteration))
    
    chains = 1:chains.n_chains;

    outliers = setdiff([findfirst(check.bad_number .== X[:, chain, :]) for chain in chains], [nothing]);
    fine_chains = setdiff(chains, outliers);

    if length(outliers) > 0
        @warn string(length(outliers)) * " outlier chains detected, setting to random chains"

        resampled = [rand(rngs[outlier], fine_chains) for outlier in outliers];

        chains.ld[chains.current_position[outliers], :] .= chains.ld[chains.current_position[resampled], :];
        chains.X[chains.current_position[outliers], :] .= chains.X[chains.current_position[resampled], :];
    end
end
```

## Custom Proposal Distributions

The core of the samplers in `DEMetropolis.jl` are the update steps, which propose new parameter values. You can create your own proposal distributions by defining a new update type.

To do this, create a struct that subtypes `DEMetropolis.update_struct` and implement the `DEMetropolis.update!` method for it.

The method signature for the update is:
`update!(update::YourUpdate, chains, ld, rng, chain::Int)`

- `update`: An instance of your custom update struct.
- `chains`: The `chains_struct`. Helper functions like `DEMetropolis.get_value` and `DEMetropolis.update_value!` are available to get the current state and to update it after the Metropolis-Hastings step.
- `ld`: The log-density function from `LogDensityProblems.jl`.
- `rng`: The random number generator for the current chain.
- `chain`: The index of the chain to be updated.

If your proposal distribution requires adaptation (e.g., tuning a step size during warmup), you can also implement `DEMetropolis.adapt_update!(update::YourUpdate, chains)`.

Here is an example of a simple Metropolis-Hastings random walk update with a fixed step size.

```julia
using DEMetropolis, LogDensityProblems, Distributions, LinearAlgebra

# Define the struct for the update
struct MetropolisHastingsUpdate <: DEMetropolis.update_struct
    proposal_distribution::MvNormal
end
# Implement the update! function
function DEMetropolis.update!(update::MetropolisHastingsUpdate, chains, ld, rng, chain)
    # Get the current state of the chain
    x_current = DEMetropolis.get_value(chains, chain)
    
    # Propose a new point using a random walk
    x_proposal = x_current .+ rand(rng, update.proposal_distribution);
    
    # Calculate the log-density of the proposed point
    ld_proposal = LogDensityProblems.logdensity(ld, x_proposal)
    
    # The proposal is symmetric, so the Hastings factor is 0 in log-space. Other this could be included via offset = ...
    # update_value! handles the acceptance/rejection step.
    DEMetropolis.update_value!(chains, rng, chain, x_proposal, ld_proposal)
end
```

### Adaptive Metropolis-Hastings Update

For more complex problems, an adaptive proposal can be much more efficient. The following example shows how to create a Metropolis-Hastings update that adapts its proposal distribution during the warmup phase. It uses the covariance of the samples drawn so far to shape the proposal, which is a common technique in adaptive MCMC.

To achieve this, we will implement the `DEMetropolis.adapt_update!` method, which is called periodically during the sampling process.

```julia
using DEMetropolis, LogDensityProblems, Distributions, LinearAlgebra, Statistics

# The struct needs to be mutable to allow the proposal distribution to be updated.
mutable struct AdaptiveMetropolisUpdate <: DEMetropolis.update_struct
    proposal_distribution::MvNormal
    # Parameters to control the adaptation
    adapt_after::Int  # Start adapting after this many iterations
    adapt_every::Int  # Adapt every N iterations
    adapt_scale::Float64 # Scale factor for the covariance
end

# A constructor to set up the initial state
function AdaptiveMetropolisUpdate(
    n_pars::Int;
    initial_std::Float64=0.1,
    adapt_after::Int=200,
    adapt_every::Int=100,
    adapt_scale::Float64=2.38^2
)
    # Start with a simple isotropic proposal
    initial_cov = (initial_std^2) * I(n_pars)
    proposal = MvNormal(zeros(n_pars), initial_cov)
    return AdaptiveMetropolisUpdate(proposal, adapt_after, adapt_every, adapt_scale / n_pars)
end

# The update! method is the same as for the non-adaptive version
function DEMetropolis.update!(update::AdaptiveMetropolisUpdate, chains, ld, rng, chain)
    x_current = DEMetropolis.get_value(chains, chain)
    # Propose a new point using the current proposal distribution
    x_proposal = x_current .+ rand(rng, update.proposal_distribution)
    ld_proposal = LogDensityProblems.logdensity(ld, x_proposal)
    DEMetropolis.update_value!(chains, rng, chain, x_proposal, ld_proposal)
end

# The adapt_update! method contains the adaptation logic
function DEMetropolis.adapt_update!(update::AdaptiveMetropolisUpdate, chains)
    # Only adapt during warmup, after a burn-in period, and at specified intervals
    if !chains.warmup || chains.samples < update.adapt_after || chains.samples % update.adapt_every != 0
        return
    end

    println("Adapting proposal at iteration ", chains.samples)

    # Get all the warmup samples up to the current point
    warmup_samples_3d = DEMetropolis.population_to_samples(chains, 1:(chains.samples-1))
    
    # Reshape to a 2D matrix (n_samples, n_params)
    n_params = size(warmup_samples_3d, 3)
    warmup_samples_2d = reshape(warmup_samples_3d, :, n_params)

    # Calculate the covariance of the samples and regularize it
    cov_matrix = cov(warmup_samples_2d)
    regularized_cov = cov_matrix + 1e-6 * I

    # Update the proposal distribution
    update.proposal_distribution = MvNormal(zeros(n_params), update.adapt_scale .* regularized_cov)
end
```

## Example: Using Custom Components

Here is a complete example that shows how to use all the custom components defined above with `composite_sampler`.

```julia
using DEMetropolis, LogDensityProblems, TransformVariables, Distributions, TransformedLogDensities

# Set up and run the sampler

# A simple log-density to sample from (a 2D standard normal distribution)
ld = TransformedLogDensity(as(Array, 2), x -> -sum(x.^2) / 2);

# Use our custom Metropolis-Hastings update
my_sampler_scheme = setup_sampler_scheme(
    MetropolisHastingsUpdate(MvNormal([0.0, 0.0], I)), AdaptiveMetropolisUpdate(2)
);

# Use our custom stopping criterion
my_stopping_criterion = MaxIterationsStopping(2000);

# Use our custom diagnostic check
my_diagnostics = [MyCustomDiagnostic(-100.0)];

# Set up initial state for the chains (10 chains for a 2-parameter model)
# For memoryless sampling, the number of rows is the number of chains.
initial_state = randn(10, 2);

# Run the composite sampler with all our custom components
# epoch_size is the number of samples per chain, per epoch
output = composite_sampler(
    ld,
    1000, 
    10,
    false, # memoryless
    initial_state,
    my_sampler_scheme,
    my_stopping_criterion;
    warmup_epochs = 5,
    diagnostic_checks = my_diagnostics
);

println("Sampling finished. Total samples per chain: ", size(output.samples, 1))
println("Adapted Covariance: ", output.sampler_scheme.updates[2].proposal_distribution.Σ)
```

This will print the median and 90% credible interval for each parameter, giving you a summary of the posterior distribution.

For more details, see the [DEMetropolis Documentation](@ref) and the [Customizing your sampler](@ref) section. For general Julia documentation and best practices, refer to the [Julia documentation manual](https://docs.julialang.org/en/v1/manual/documentation/).