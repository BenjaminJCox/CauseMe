## -- preamble
using DrWatson
using Optim
using Distributions
using Random
using DataFrames
using Plots
using CSV

plotlyjs()

include(srcdir("workers.jl"))

# Random.seed!(0x63617573656d65)

γ = exp(3)
l1_penalty(A) = γ * norm(A, 1)

em_steps = 50
function perf_em(dimA, steps, Y, H, m0, P, Q, R)
    A_gem = rand(dimA,dimA)
    a_size = size(A_gem)
    a_nelem = prod(a_size)
    A_gem_vec = reshape(A_gem, a_nelem)
    for s in 1:em_steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)[1]
        Q_optim(A) = Qf(reshape(A, a_size))
        optimres = optimize(Q_optim, A_gem_vec, NelderMead())
        A_gem_vec = optimres.minimizer
        A_gem = reshape(A_gem_vec, a_size)
    end
    return A_gem
end

dr_steps = 50
function em_dr(dimA, steps, Y, H, m0, P, Q, R)
    A_gem = I(dimA) + randn(dimA, dimA)
    θ = 1.0
    T = size(Y, 2)
    for s in 1:steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3)
    end
    return A_gem
end

data_file = datadir("exp_raw/TestCLIMnoise_N-5_T-100/TestCLIMnoise_N-5_T-100_0001.txt")

full_set = Matrix(Matrix(DataFrame(CSV.File(data_file; header = false)))')
n_series = size(full_set, 1)
n_data = size(full_set, 2)
plot_obs = full_set[1,:]
## -- GraphEM algorithm
sta_var = 0.10^2
P_full = sta_var .* Matrix(I(n_series))
Q_full = sta_var .* Matrix(I(n_series))

obs_var = 0.15^2
R_full = obs_var .* Matrix(I(n_series))

H_full = 1. .* Matrix(I(n_series))

m0_full = full_set[:, 1]

graphem_runs_full = 300
## --
genem_samples = zeros(n_series, n_series, graphem_runs_full)
for i in 1:graphem_runs_full
    genem_samples[:, :, i] .+= em_dr(n_series, 50, full_set, H_full, m0_full, P_full, Q_full, R_full)
end

## --
genem_full = (mean(genem_samples, dims = 3))[:,:,1]
genem_var = (var(genem_samples, dims = 3))[:, :, 1]

genfil_gem = perform_kalman(full_set, genem_full, H_full, m0_full, P_full, Q_full, R_full)
plot_series_gem = genfil_gem[1][1,:]
gr()
plot(plot_series_gem, label = "GEM Causal Filter", legend = :outerbottom, size = (1000, 500))
plot!(plot_obs, label = "Noisy Observations")
savefig("initial_gem_kf.pdf")
plotlyjs()
# ## -- MMH with GEM as starting point
#
# U_prop = ones(n_series, n_series) .+ I(n_series)
# V_prop = copy(U_prop)
# var_mul = 0.025
# U_prop .*= var_mul
# V_prop .*= var_mul
#
# A0 = genem_full
# A_mmh = kalmanesq_MMH_A(U_prop, V_prop, P_full, Q_full, R_full, H_full, m0_full, full_set; A0 = A0, steps = 5_000)
# genfil_mmh = perform_kalman(full_set, A_mmh, H_full, m0_full, P_full, Q_full, R_full)
# plot_series = genfil_mmh[1][1,:]
# plot(plot_series, label = "GEMMMH Causal Filter")
# plot!(plot_obs, label = "Noisy Observations")

## --
