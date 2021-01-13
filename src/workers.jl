using LinearAlgebra
using DrWatson
using Kronecker

function perform_kalman(observations, A, H, m0, P0, Q, R)
    m = copy(m0)
    P = copy(P0)
    v = observations[:, 1] - H * m
    S = H * P * H' + R
    K = P * H' \ S
    T = size(observations, 2)
    filtered_state = zeros(length(m0), T)
    filtered_cov = zeros(length(m0), length(m0), T)
    for t = 1:T
        m = A * m
        P = A * P * A' + Q
        v = observations[:, t] - H * m
        S = H * P * H' + R
        K = (P * H') / S
        m = m + K * v
        P = P - K * S * K'
        filtered_state[:, t] .= m
        filtered_cov[:, :, t] .= P
    end
    return (filtered_state, filtered_cov)
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
    Σ = zeros(size(P))
    Φ = zeros(size(P))
    C = zeros(size(m0 * m0'))
    K = size(observations, 2)

    rts_means = rts[1]
    rts_covs = rts[2]
    rts_G = rts[3]

    for k = 2:K
        Σ += rts_covs[:, :, k] + (rts_means[:, k] * rts_means[:, k]')
        Φ += rts_covs[:, :, k-1] + (rts_means[:, k-1] * rts_means[:, k-1]')
        C += rts_covs[:, :, k] * rts_G[:, :, k-1]' + rts_means[:, k] * rts_means[:, k-1]'
    end
    Σ ./= K
    Φ ./= K
    C ./= K
    _f1(A) = (K / 2.0) * tr(inv(Q) * (Σ - C * A' - A * C' + A * Φ * A'))
    _f2(A) = _lp(A)
    Qf(A) = _f1(A) .+ _f2(A)
    val_dict = @dict Σ Φ C
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

    _t1 = inv(id ⊗ (K .* Q_inv) + (θ .* Φ_inv) ⊗ id)
    _t2 = vec(K * Q_inv * C * Φ_inv)
    rv = _t1 * _t2
    rvl = isqrt(length(rv))
    return reshape(rv, (rvl, rvl))
end

function _proxf2(A, θ)
    maximator = max.(abs.(A) .- θ, 0)
    return sign.(A) .* maximator
end


function DR_opt(f1, f2, proxf1, proxf2, θ, K, Q, val_dict, Z0, ϵ)
    difference = 2 * ϵ
    Z = copy(Z0)
    A = proxf2(Z, θ)
    A_old = copy(A)
    V = proxf1(2A - Z, θ, K, Q, val_dict)
    Z .= Z + θ .* (V - A)
    # println(Z + θ .* (V - A))
    while difference >= ϵ
        # @info("DRSTEP")
        A .= proxf2(Z, θ)
        # @info(A)
        V .= proxf1(2.0 .* A - Z, θ, K, Q, val_dict)
        # @info(V)
        Z .= Z + θ .* (V - A)
        # @info(Z)
        # @info(abs(f1(A) + f2(A) - f1(A_old) - f2(A_old)))
        difference = abs(f1(A) + f2(A) - f1(A_old) - f2(A_old))
        A_old .= A
    end
    return A
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
