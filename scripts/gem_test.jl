using DrWatson
using Optim
using Distributions
using Random
using DataFrames
using Plots

plotlyjs()

include(srcdir("workers.jl"))

γ = exp(2)
l1_penalty(A) = γ * norm(A, 1)


A = [1. 1.; 0. 1.]
H = [1. 0.; 0. 1.]
P = [1. 0.; 0. 1.]
Q = [0.1 0.0; 0.0 0.1]
R = [1. 0.; 0. 1.]

m0 = [1., 0.]
T = 50

X = zeros(2, T)
Y = zeros(2, T)

prs_noise = MvNormal(Q)
obs_noise = MvNormal(R)
prior_state = MvNormal(m0, P)

X[:, 1] = rand(prior_state)
Y[:, 1] = H * X[:, 1] .+ rand(obs_noise)

for t in 2:T
    X[:, t] = A * X[:, t-1] .+ rand(prs_noise)
    Y[:, t] = H * X[:, t] .+ rand(obs_noise)
end

filtered = perform_kalman(Y, A, H, m0, P, Q, R)
smoothed = perform_rts(filtered, A, H, Q, R)

em_steps = 50

function perf_em()
    A_gem = rand(2,2)
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
    A_gem = rand(dimA, dimA)
    θ = 1.0
    for s in 1:steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3)
    end
    return A_gem
end


# A_graphem1 = perf_em()
# A_graphem2 = perf_em()
# A_graphem3 = perf_em()
# A_graphem4 = perf_em()

A_graphem_dr = em_dr(2, dr_steps, Y, H, m0, P, Q, R)

data_file = datadir("exp_pro/TestCLIM_N-5_T-100.csv")
colsww = [1]
data_csv = CSV.File(data_file; select=colsww, header = false)
raw_obs = vec(Matrix(DataFrame(data_csv)))

l_data = length(raw_obs)

lag = 3
implied_obs = generate_lagged_obs(raw_obs, lag)

mpt = maximum(raw_obs) - minimum(raw_obs)

H_mat = 1. .* Matrix(I(lag+1))

var = mpt / 10.
P_mat = var * Matrix(I(lag+1))
Q_mat = var * Matrix(I(lag+1))
R_mat = 0.01 * var * Matrix(I(lag+1))

m0_cd = implied_obs[:, 1]

genem = em_dr(lag+1, 100, implied_obs, H_mat, m0_cd, P_mat, Q_mat, R_mat)

genfil = perform_kalman(implied_obs, genem, H_mat, m0_cd, P_mat, Q_mat, R_mat)

varoint = genfil[1][lag+1,:]

gr()

plot((lag+1):l_data, varoint, label = "Generative Filter")
plot!(raw_obs, label = "Data")
