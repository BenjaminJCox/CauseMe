using LinearAlgebra
using Distributions
using Random

function X_past_constructor_old(X, t_bootstrap, lags)
    d = size(X, 2)
    v = length(t_bootstrap)
    X_past = Matrix{Float64}(undef, v, 1+(lags*d))
    X_past[:,1] .= 1.0
    for lag in 1:lags
        cols = (2+(d*(lag-1))):(1+(d*(lag)))
        X_past[:,cols] = X[t_bootstrap.-lag,:]
    end
    X_past
end

function Z_constructor(X, L)
    d = size(X, 2)
    T = size(X, 1)
    Z_rows = T - L
    Z_cols = 1 + (d * L)
    Z = Matrix{Float64}(undef, Z_rows, Z_cols)
    Z[:,1] .= 1.0
    for k in 0:(L-1)
        start_idx = k*d+1
        fin_idx = (k+1)*d+1
        x_rsind = L-k
        x_reind = x_rsind + T - L - 1
        xoint = X[x_rsind:x_reind,:]
        Z[:,(1+start_idx):fin_idx] .= xoint
    end
    return Z
end

function slarac_aggregator!(A, A_full, L, d)
    ij = 1:d
    for j in ij, i in ij
        A[i,j] = maximum(A_full[i, j .+ (0:(L-1)).*d])
    end
    return A
end



# X = [-1.39294  -0.629396; 1.14853 0.0176775; -1.5745 -0.082081; -1.01497  -0.288307]


function perform_slarac_old(X::Matrix, L::Integer, B::Integer, bootstrap_sizes::Vector)
    @assert length(bootstrap_sizes) == B
    @assert L > 0
    @assert B > 0
    T = size(X, 1)
    d = size(X, 2)
    A_full = zeros(d, d*L)
    A = Matrix{Float64}(undef, d, d)
    Z = hcat(ones(T), X)
    β = zeros(d * L + 1, d)
    for b in 1:B
        β[:,:] .= 0.0
        lags = rand(1:L)
        ico = min(T - lags + 1, T-1)
        t_bootstrap = (lags+1):T
        Y_b = X[t_bootstrap,:]
        X_past_b = X_past_constructor(X, t_bootstrap, lags)[:,1:(ico)]
        β[1:ico,:] = X_past_b \ Y_b
        A_full .+= abs.(β[2:end,:]')
    end
    A .= slarac_aggregator!(A, A_full ./ B, L, d)
    return A'
end

function perform_slarac(X::Matrix, L::Integer, B::Integer)
    @assert L > 0
    @assert B > 0
    T = size(X, 1)
    d = size(X, 2)
    A_full = zeros(d, d*L)
    A = Matrix{Float64}(undef, d, d)
    β = zeros(d * L + 1, d)
    INV_GR = 2.0 / (1.0 + sqrt(5.0))
    subsample_sizes = [1.0, 2.0, 3.0, 6.0]
    subsample_sizes .= T .* INV_GR .^ inv.(subsample_sizes)
    subsample_sizes_absol = round.(Int64, sample(subsample_sizes, B, replace = true))
    Y_t = X[(L+1):end,:]
    Z_t = Z_constructor(X, L)
    # @info(Z_t)
    for b in 1:B
        β[:,:] .= 0.0
        lags = rand(1:L)
        # lags = L
        eff_lags = lags * d + 1
        ps_boot = size(Y_t, 1)
        t_bootstrap = sample(1:ps_boot, subsample_sizes_absol[b], replace = true)
        # t_bootstrap = 1:ps_boot
        # t_bootstrap = sample((lags+1):T, 100, replace = true)
        ico = (rand(1:lags) * d + 1)
        Y_b = Y_t[t_bootstrap,:]
        Z_b = Z_t[t_bootstrap,:]
        beta = svd(Z_b) \ Y_b
        β[1:size(beta,1),:] .= beta
        # @info(size(beta))
        A_full .+= abs.(β[2:end,:]')
    end
    A .= slarac_aggregator!(A, A_full ./ B, L, d)
    return (A', A_full)
end

b = perform_slarac(X, 3, 2000);
display(b[1])
