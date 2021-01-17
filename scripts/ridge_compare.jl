using DataFrames
using Plots
using CSV
using MultivariateStats
using DrWatson

plotlyjs()

include(srcdir("workers.jl"))

data_file = datadir("exp_raw/TestCLIMnoise_N-5_T-100/TestCLIMnoise_N-5_T-100_0001.txt")

full_set = DataFrame(CSV.File(data_file; header = false))
n_series = ncol(full_set)
n_data = nrow(full_set)

feature_mat = Matrix(full_set[:, 2:end])
obs = full_set[:,1]

λ = exp(3)

ridge_relations = ridge(feature_mat, obs, λ, bias = false)

obs_pred = feature_mat * ridge_relations

# gr()
plot(obs_pred, label = "Ridge regression estimates", legend = :outerbottom, size = (1000, 500))
plot!(obs, label = "Noisy Observations")
# savefig("ridge_obs.pdf")
# plotlyjs()

function hat_matrix(X)
    p = size(X, 2)
    qrqx = qr(X).Q
    return qrqx[:,1:p] * qrqx[:,1:p]'
end

function ann_matrix(X)
    n = size(X, 1)
    return I(n) - hat_matrix(X)
end

function sigma2_stat(X, y)
    n = size(X, 1)
    _t1 = y' * ann_matrix(X) * y
    return _t1 ./ n
end

function ridge_varest(X, y, λ)
    n = size(X, 1)
    p = size(X, 2)
    σ² = sigma2_stat(X, y)
    _t2 = X' * X
    _t1 = inv(_t2 + λ .* I(p))
    return diag(σ² .* (_t1 * _t2 * _t1'))
end

function ridgeenator(full_set::DataFrame, λ)
    n_series = ncol(full_set)
    n_data = nrow(full_set)
    r_matrix = zeros(n_series, n_series)
    v_matrix = zeros(n_series, n_series)
    r_matrix .+= I(n_series)
    for n in 1:n_series
        subset = Matrix(select(full_set, Not(n)))
        obsset = Matrix(select(full_set, n))
        _ridgevec = ridge(subset, obsset, λ, bias = false)
        r_matrix[n, Not(n)] = _ridgevec
        v_matrix[n, Not(n)] = ridge_varest(subset, obsset, λ)
    end
    return (r_matrix, v_matrix)
end

RN = ridgeenator(full_set, λ)
cr_mat = RN[1]
pr_mat = RN[2]
