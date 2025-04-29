function build_initial_state(rng, ld, initial_state, N₀)
    if isnothing(initial_state)
        return randn(rng, N₀, LogDensityProblems.dimension(ld))
    else
        current_N = size(initial_state, 1)
        current_pars = size(initial_state, 2)
        if current_pars != LogDensityProblems.dimension(ld)
            error("Number of parameters in initial state must be equal to the number of parameters in the log density")
        end
        if current_N == N₀
            return initial_state
        elseif current_N < N₀
            warning("Initial state is smaller than the number of chains. Expanding initial state.")
            return cat(
                randn(rng, eltype(initial_state), N₀ - current_N, current_pars),
                initial_state,
                dims = 1
            )
        else
            warning("Initial state is larger than the number of chains. Shrinking initial state.")
            #shrink initial state
            return initial_state[1:N₀, :, :]
        end
    end
end

#DOI 10.1007/s11222-006-8769-1
function deMC(
    ld, n_its;
    n_burnin = n_its * 5,
    n_chains = LogDensityProblems.dimension(ld) * 2,
    initial_state = nothing,
    memory = false,
    save_burnt = true,
    parallel = false,
    rng = Random.default_rng(),
    diagnostic_checks = nothing,
    check_epochs = 1,
    γ₁ = nothing,
    γ₂ = 1.0,
    p_γ₂ = 0.1,
    β = Distributions.Uniform(-1e-4, 1e-4)
)

    if n_chains < LogDensityProblems.dimension(ld)
        warning("Number of chains should be greater than or equal to the number of parameters")
    end

    #setup initial state
    initial_state = build_initial_state(rng, ld, initial_state, n_chains)

    #build sampler scheme
    if γ₁ != γ₂
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ₂, β = β),
            setup_de_update(ld, γ = γ₁, β = β),
            w = [p_γ₂, 1 - p_γ₂],
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ₁, β = β)
        )
    end

    composite_sampler(
        ld, n_its, n_chains, memory, initial_state, sampler_scheme;
        save_burnt = save_burnt, rng = rng, n_burnin = n_burnin, parallel = parallel,
        diagnostic_checks = diagnostic_checks, check_epochs = check_epochs
    )

end

#DOI 10.1007/s11222-008-9104-9
function deMCzs(
    ld, epoch_size;
    warmup_epochs = 5,
    epoch_limit = 20,
    n_chains = LogDensityProblems.dimension(ld) * 2,
    N₀ = n_chains * 2,
    initial_state = nothing,
    memory = true,
    save_burnt = true,
    parallel = false,
    rng = Random.default_rng(),
    diagnostic_checks = nothing,
    stopping_criteria = R̂_stopping_criteria(),
    γ = nothing,
    γₛ = nothing,
    p_snooker = 0.1,
    β = Distributions.Uniform(-1e-4, 1e-4)
)

    if n_chains < LogDensityProblems.dimension(ld)
        warning("Number of chains should be greater than or equal to the number of parameters")
    end

    #setup initial state
    initial_state = build_initial_state(rng, ld, initial_state, N₀);

    #build sampler scheme
    if p_snooker == 0
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ, β = β)
        )
    else 
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(ld, γ = γ, β = β),
            setup_snooker_update(γ = γₛ),
            w = [1 - p_snooker, p_snooker],
        )
    end

    composite_sampler(
        ld, epoch_size, n_chains, memory, initial_state, sampler_scheme, stopping_criteria;
        save_burnt = save_burnt, rng = rng, warmup_epochs = warmup_epochs,
        parallel = parallel, epoch_limit = epoch_limit, diagnostic_checks = diagnostic_checks
    )

end