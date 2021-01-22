using LinearAlgebra
using DrWatson
using Kronecker
using Distributions
using Random

function perform_kalman(observations, A, H, m0, P0, Q, R)
    m = copy(m0)
    P = copy(P0)
    v = observations[:, 1] - H * m
    S = H * P * H' + R
    K = P * H' \ S
    T = size(observations, 2)
    _xd = length(m0)
    filtered_state = zeros(length(m0), T)
    filtered_cov = zeros(length(m0), length(m0), T)
    l_like_est = 0.0
    offness = 0.0
    H = Matrix(Hermitian(H))
    for t = 1:T
        m = A * m
        P = A * P * transpose(A) + Q
        v = observations[:, t] - H * m
        S = H * P * transpose(H) + R
        offness += norm(S - Matrix(Hermitian(S)), 1)
        S = Matrix(Hermitian(S))
        K = (P * transpose(H)) * inv(S)
        # l_like_est += logpdf(MvNormal(H * m, S), observations[:, t])
        # unstable, need to implement in sqrt form
        m = m + K * v
        P = (I(_xd) - K * H) * P * (I(_xd) - K * H)' + K * R * K'
        filtered_state[:, t] = copy(m)
        filtered_cov[:, :, t] = copy(P)
    end
    return (filtered_state, filtered_cov, l_like_est, offness)
end

function perform_rts(kalman_out, A, H, Q, R)
    kal_means = kalman_out[1]
    kal_covs = kalman_out[2]
    T = size(kal_means, 2)
    rts_means = zeros(size(kal_means))
    rts_covs = zeros(size(kal_covs))
    rts_means[:, T] = kal_means[:, T]
    rts_covs[:, :, T] = kal_covs[:, :, T]
    # just preallocation, values not important
    m_bar = A * kal_means[:, T]
    P_bar = A * kal_covs[:, :, T] * A' + Q
    G = kal_covs[:, :, T] * A' / P_bar
    G_ks = zeros(size(G)..., T)
    G_ks[:, :, T] .= G
    m = copy(m_bar)
    P = copy(P_bar)
    for k = (T-1):-1:1
        m_bar .= A * kal_means[:, k]
        P_bar .= A * kal_covs[:, :, k] * A' + Q
        G .= kal_covs[:, :, k] * A' / P_bar
        G_ks[:, :, k] .= G
        m = kal_means[:, k] + G * (rts_means[:, k+1] - m_bar)
        P = kal_covs[:, :, k] + G * (rts_covs[:, :, k+1] - P_bar) * G'
        rts_means[:, k] .= m
        rts_covs[:, :, k] .= P
    end
    return (rts_means, rts_covs, G_ks)
end

function Q_func(observations, A′, H, m0, P0, Q, R, _lp)
    kal = perform_kalman(observations, A′, H, m0, P0, Q, R)
    rts = perform_rts(kal, A′, H, Q, R)
    rts_means = rts[1]
    rts_covs = rts[2]
    rts_G = rts[3]

    Σ = zeros(size(P0))
    Φ = zeros(size(P0))
    B = zeros(size(observations[:, 1] * rts_means[:, 1]'))
    C = zeros(size(m0 * m0'))
    D = zeros(size(observations[:, 1] * observations[:, 1]'))

    K = size(observations, 2)

    for k = 2:K
        B += observations[:, k] * rts_means[:, k]'
        Σ += rts_covs[:, :, k] + (rts_means[:, k] * rts_means[:, k]')
        Φ += rts_covs[:, :, k-1] + (rts_means[:, k-1] * rts_means[:, k-1]')
        C += (rts_covs[:, :, k] * rts_G[:, :, k-1]') + (rts_means[:, k] * rts_means[:, k-1]')
        D += observations[:, k] * observations[:, k]'
    end
    B ./= K
    Σ ./= K
    Φ ./= K
    C ./= K
    D ./= K

    _f1(A) = (K / 2.0) * tr(inv(Q) * (Σ - C * A' - A * C' + A * Φ * A'))
    _f2(A) = _lp(A)
    Qf(A) = _f1(A) + _f2(A)
    val_dict = @dict Σ Φ C B D
    return (Qf, _f1, _f2, val_dict)
end

function isotropic_proxf1(A, θ, K, Q, val_dict)
    C = val_dict[:C]
    Φ = val_dict[:Φ]

    σ = Q[1, 1]
    id = 1.0 .* Matrix(I(size(Q, 1)))
    tKs = θ .* K ./ σ
    _t1 = tKs .* C + A
    _t2 = inv(tKs .* Φ + id)
    return _t1 * _t2
end

function _proxf1(A, θ, K, Q, val_dict)
    C = val_dict[:C]
    Φ = val_dict[:Φ]
    Q_inv = inv(Q)
    Φ_inv = inv(Φ)

    id = 1.0 .* Matrix(I(size(Q, 1)))

    _t1 = id ⊗ (K * Q_inv) + (θ * Φ_inv) ⊗ id
    _t2 = vec(K * Q_inv * C * Φ_inv)
    rv = _t1 \ _t2
    rvl = isqrt(length(rv))
    return reshape(rv, (rvl, rvl))
end

function _proxf2(A, θ)
    maximator = max.(abs.(A) .- θ, 0.0)
    return sign.(A) .* maximator
end


function DR_opt(f1, f2, proxf1, proxf2, θ, K, Q, val_dict, Z0, ϵ, γ; maxiters = 100)
    difference = 2 * ϵ
    Z = copy(Z0)
    A = proxf2(Z, θ)
    A_old = copy(A)
    V = proxf1(2A - Z, θ, K, Q, val_dict)
    Z .= Z + θ .* (V - A)
    # println(Z + θ .* (V - A))
    iters = 0
    while (difference >= ϵ) && (iters < maxiters)
        A .= proxf2(Z, γ)
        V .= proxf1(2.0 .* A - Z, θ, K, Q, val_dict)
        # @info(V, isotropic_proxf1(2.0 .* A - Z, θ, K, Q, val_dict))
        Z .= Z + θ .* (V - A)
        difference = abs.(f1(A) + f2(A) - f1(A_old) - f2(A_old))
        A_old .= A
        iters += 1
    end
    return A
end

function generate_lagged_obs(Y::Vector{Float64}, lag::Integer; lagmin::Integer = 0)
    num_obs = length(Y)
    num_gen_obs = num_obs - lag
    generated_obs = Matrix{Float64}(undef, lag + 1 - lagmin, num_gen_obs)
    for i = 1:num_gen_obs
        generated_obs[:, i] = Y[(i):(i+lag-lagmin)]
    end
    return generated_obs
end

function kalmanesq_MMH_A(
    U::AbstractMatrix,
    V::AbstractMatrix,
    P,
    Q,
    R,
    H,
    m0,
    observations;
    steps::Integer = 1000,
    A0 = 1.0 * Matrix(I(size(P, 1))),
)
    # flat prior, eliminates p(A) term
    # symmetric walk, eliminates q term
    # propose based on LR only
    out_A = 0.0 .* A0
    A = copy(A0)
    A′ = copy(A)
    M = zeros(size(A))
    pert_dist = MatrixNormal(M, U, V)
    l_pya = perform_kalman(observations, A, H, m0, P, Q, R)[3]
    l_pyap = copy(l_pya)
    l_accrat = 0.0
    for n = 1:steps
        A′ = A .+ rand(pert_dist)
        l_pyap = perform_kalman(observations, A′, H, m0, P, Q, R)[3]
        l_accrat = l_pyap - l_pya
        l_rand = log(rand())
        if (l_rand < l_accrat)
            A = copy(A′)
            l_pya = copy(l_pyap)
        end
        out_A .+= (A ./ steps)
    end
    return out_A
end

function perf_em(dimA, steps, Y, H, m0, P, Q, R)
    A_gem = rand(dimA, dimA)
    a_size = size(A_gem)
    a_nelem = prod(a_size)
    A_gem_vec = reshape(A_gem, a_nelem)
    for s = 1:em_steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)[1]
        Q_optim(A) = Qf(reshape(A, a_size))
        optimres = optimize(Q_optim, A_gem_vec, NelderMead())
        A_gem_vec = optimres.minimizer
        A_gem = reshape(A_gem_vec, a_size)
    end
    return A_gem
end

function oneshot_H(val_dict)
    B = val_dict[:B]
    Σ = val_dict[:Σ]
    H_map = B / Σ
    return H_map
end

function oneshot_Q(val_dict, A)
    Σ = val_dict[:Σ]
    C = val_dict[:C]
    Φ = val_dict[:Φ]
    Q_map = Σ - C * A' - A * C' + A * Φ * A'
    return Q_map
end

function oneshot_R(val_dict, H)
    Σ = val_dict[:Σ]
    D = val_dict[:D]
    B = val_dict[:B]
    R_map = D - H * B' - B * H' + H * Σ * H'
    return R_map
end

function em_dr(dimA, steps, Y, H, m0, P, Q, R, γ)
    A_gem = rand(dimA, dimA)
    θ = 1.0
    T = size(Y, 2)
    osh = zeros(size(H))
    for s = 1:steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3, γ)
        osh .= oneshot_H(Qf[4])
    end
    return (A_gem, osh)
end

# A = [1. 1.; 0. 1.]
# H = [1. 0.; 0. 1.]
# P = [1. 0.; 0. 1.]
# Q = [0.1 0.0; 0.0 0.1]
# R = [1. 0.; 0. 1.]
#
# m0 = [1., 0.]
# T = 50
#
# X = zeros(2, T)
# Y = zeros(2, T)
#
# prs_noise = MvNormal(Q)
# obs_noise = MvNormal(R)
# prior_state = MvNormal(m0, P)
#
# X[:, 1] = rand(prior_state)
# Y[:, 1] = H * X[:, 1] .+ rand(obs_noise)
#
# for t in 2:T
#     X[:, t] = A * X[:, t-1] .+ rand(prs_noise)
#     Y[:, t] = H * X[:, t] .+ rand(obs_noise)
# end
#
# filtered = perform_kalman(Y, A, H, m0, P, Q, R)
# smoothed = perform_rts(filtered, A, H, Q, R)
#
# plot(X[1, :], legend = false)
# # plot!(filtered[1][1, :], legend = false)
# plot!(smoothed[1][1, :], legend = false)
function Z_constructor(X, L)
    d = size(X, 2)
    T = size(X, 1)
    Z_rows = T - L
    Z_cols = 1 + (d * L)
    Z = Matrix{Float64}(undef, Z_rows, Z_cols)
    Z[:, 1] .= 1.0
    for k = 0:(L-1)
        start_idx = k * d + 1
        fin_idx = (k + 1) * d + 1
        x_rsind = L - k
        x_reind = x_rsind + T - L - 1
        xoint = X[x_rsind:x_reind, :]
        Z[:, (1+start_idx):fin_idx] .= xoint
    end
    return Z
end

function slarac_aggregator!(A, A_full, L, d)
    ij = 1:d
    for j in ij, i in ij
        A[i, j] = maximum(A_full[i, j.+(0:(L-1)).*d])
    end
    return A
end

function perform_slarac(X::Matrix, L::Integer, B::Integer)
    @assert L > 0
    @assert B > 0
    T = size(X, 1)
    d = size(X, 2)
    A_full = zeros(d, d * L)
    A = Matrix{Float64}(undef, d, d)
    β = zeros(d * L + 1, d)
    INV_GR = 2.0 / (1.0 + sqrt(5.0))
    subsample_sizes = [1.0, 2.0, 3.0, 6.0]
    subsample_sizes .= T .* INV_GR .^ inv.(subsample_sizes)
    subsample_sizes_absol = round.(Int64, sample(subsample_sizes, B, replace = true))
    Y_t = X[(L+1):end, :]
    Z_t = Z_constructor(X, L)
    # @info(Z_t)
    for b = 1:B
        β[:, :] .= 0.0
        lags = rand(1:L)
        # lags = L
        eff_lags = lags * d + 1
        ps_boot = size(Y_t, 1)
        t_bootstrap = sample(1:ps_boot, subsample_sizes_absol[b], replace = true)
        # t_bootstrap = 1:ps_boot
        # t_bootstrap = sample((lags+1):T, 100, replace = true)
        ico = (rand(1:lags) * d + 1)
        Y_b = Y_t[t_bootstrap, :]
        Z_b = Z_t[t_bootstrap, :]
        beta = svd(Z_b) \ Y_b
        β[1:size(beta, 1), :] .= beta
        # @info(size(beta))
        A_full .+= abs.(β[2:end, :]')
    end
    A .= slarac_aggregator!(A, A_full ./ B, L, d)
    return (A', A_full)
end
