using DrWatson
using Optim
using Distributions
using Random
using DataFrames
using Plots
using CSV

plotlyjs()

include(srcdir("workers.jl"))

Random.seed!(0x296c1bfc)

γ = 0.5
l1_penalty(A) = γ * norm(A, 1)


A = [0.5 -0.5; 0.0 1.0]
H = [1. 0.; 0. 1.]
P = [1. 0.; 0. 1.]
Q = [0.1 0.0; 0.0 0.1]
R = [0.1 0.0; 0.0 0.1]

m0 = [1., 0.]
T = 75

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
function em_dr(dimA, steps, Y, H, m0, P, Q, R, γ)
    A_gem = rand(dimA, dimA)
    θ = 1.0
    for s in 1:steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3, γ)
    end
    return A_gem
end

graphem_runs_full = 25
genem_samples = zeros(2, 2, graphem_runs_full)
for i in 1:graphem_runs_full
    genem_samples[:, :, i] .+= em_dr(2, dr_steps, Y, H, m0, P, Q, R, γ)
end
A_graphem_dr = mean(genem_samples, dims = 3)[:, :, 1]
display(A_graphem_dr)
