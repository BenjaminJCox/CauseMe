using LinearAlgebra
using Distributions
using Random
using MultivariateStats
using StatsBase

function ew_quantile(X, q)
    rv = zeros(size(X,1), size(X,2))
    _ax = axes(X)
    _cols = _ax[2]
    _rows = _ax[1]
    for j = _cols, i = _rows
        rv[i,j] = quantile(X[i,j,:], q)
    end
    return rv
end

function perform_qrbs(X::AbstractMatrix, B::Integer, λ::AbstractFloat, q::AbstractFloat)
    @assert 0.0 <= q <= 1.0
    @assert B > 0
    @assert λ > 0
    T = size(X, 1)
    d = size(X, 2)
    A_store = zeros(d, d, B)
    Y_static = diff(X, dims = 1)
    X_static = X[1:(end-1),:]
    @info(Y_static)
    @info(X_static)
    n_samples = round(Int64, 0.7 .* T)
    for b = 1:B
        t_boot = sample(1:(T-1), n_samples, replace = true)
        Y_b = Y_static[t_boot,:]
        X_b = X_static[t_boot,:]
        # X_c = fit(ZScoreTransform, X_b; dims = 2, center = true, scale = false)
        # X_b = StatsBase.transform(X_c, X_b)
        A_store[:,:,b] .= abs.(ridge(X_b, Y_b, λ, bias = true)[1:(end-1),:])
    end
    A_qrbs = ew_quantile(A_store, q)
    return A_qrbs
end




X = [-1.39294  -0.629396; 1.14853 0.0176775; -1.5745 -0.082081; -1.01497  -0.288307]
A = perform_qrbs(X, 600, 0.005, 0.75)
display(A)
