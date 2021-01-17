using LinearAlgebra
using Distributions
using Random

function X_past_constructor(X, t_bootstrap, lags)
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

X = [-1.39294  -0.629396; 1.14853 0.0176775; -1.5745 -0.082081; -1.01497  -0.288307]


function perform_slarac(X::Matrix, L::Integer, B::Integer, bootstrap_sizes::Vector)
    @assert length(bootstrap_sizes) == B
    @assert L > 0
    @assert B > 0
    T = size(X, 1)
    d = size(X, 2)
    A_full = zeros(d, d*L)
    A = Matrix{Float64}(undef, d, d)
    Z = hcat(ones(T), X)
    for b in 1:B
        # lags = rand(1:L)
        lags = L
        # t_bootstrap = sample((lags+1):T, bootstrap_sizes[b], replace = true)
        ico = min(T - lags + 1, T-1)
        t_bootstrap = (lags+1):T
        Y_b = X[t_bootstrap,:]
        X_past_b = X_past_constructor(X, t_bootstrap, lags)[:,1:(ico)]
        β = X_past_b \ Y_b
        return β
    end
end

b = perform_slarac(X, 3, 1, [0]); display(b)
