using DrWatson
using Optim
using Distributions
using Random
using DataFrames
using Plots
using CSV
using Turing: filldist

gr()

include(srcdir("workers.jl"))

# certified random number
Random.seed!(0x8d8e1b4c2169a717f8c9fe4ccd6d5f9f)

γ = 0.9
l1_penalty(A) = γ * norm(A, 1)


A = [0.5 0.5 0.0; 0.0 0.5 0.5; 0.1 0.0 0.1]
Q = Matrix(1.0 .* I(3))
H = P = R = Q

m0 = ones(3)

T = 50

X = zeros(3, T)
Y = zeros(3, T)

prs_noise = MvNormal(Q)
obs_noise = MvNormal(R)
prior_state = MvNormal(m0, P)

X[:, 1] = rand(prior_state)
Y[:, 1] = H * X[:, 1] .+ rand(obs_noise)

for t = 2:T
    X[:, t] = A * X[:, t-1] .+ rand(prs_noise)
    Y[:, t] = H * X[:, t] .+ rand(obs_noise)
end

filtered = perform_kalman(Y, A, H, m0, P, Q, R)
smoothed = perform_rts(filtered, A, H, Q, R)

em_steps = 50
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

dr_steps = 50
function em_dr(dimA, steps, Y, H, m0, P, Q, R, γ)
    A_gem = rand(dimA, dimA)
    θ = 1.0
    for s = 1:steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3, γ)
    end
    return A_gem
end

graphem_runs_full = 25
genem_samples = zeros(3, 3, graphem_runs_full)
for i = 1:graphem_runs_full
    genem_samples[:, :, i] .+= em_dr(3, dr_steps, Y, H, m0, P, Q, R, γ)
end
A_graphem_dr = mean(genem_samples, dims = 3)[:, :, 1]
display(A_graphem_dr)

slarac_score = perform_slarac(Matrix(Y'), 1, 10000)
# display(slarac_score[1])

function kalmanesq_MMH_A_sparse(P,
    Q,
    R,
    H,
    m0,
    observations;
    steps::Integer = 1000,
    A0 = 1.0 * Matrix(I(size(P, 1))),
    burnin::Integer = fld(steps, 2),
)

    out_A = 0.0 .* A0
    N = size(A0,1)
    A = copy(A0)
    A′ = copy(A)
    M = zeros(size(A))
    pert_dist = filldist(Laplace(0,0.1), N, N)
    penalty(a) = exp(1) .* norm(a,1)

    n_step = steps - burnin

    l_pya = perform_kalman(observations, A, H, m0, P, Q, R)[3]
    l_pyap = copy(l_pya)
    l_accrat = 0.0
    for n = 1:steps
        # a_new = vec(A) .+ rand(pert_dist)
        # A′ = reshape(a_new, N, N)
        A′ = A .+ rand(pert_dist)
        l_pyap = perform_kalman(observations, A′, H, m0, P, Q, R)[3]
        l_accrat = l_pyap - l_pya - penalty(A′) + penalty(A)
        l_rand = log(rand())
        if (l_rand < l_accrat)
            A = copy(A′)
            l_pya = copy(l_pyap)
        end
        # @info A
        if n > burnin
            out_A .+= (A ./ n_step)
        end
    end
    return out_A
end

genA_mmh = kalmanesq_MMH_A_sparse(P, Q, R, H, m0, Y; steps = 10_000)
display(genA_mmh)

genfil_gem = perform_kalman(Y, A_graphem_dr, H, m0, P, Q, R)
genfil_mmh = perform_kalman(Y, genA_mmh, H, m0, P, Q, R)
opt_fil = perform_kalman(Y, A, H, m0, P, Q, R)

voi = 1
plot_series_gem = genfil_gem[1][voi, :]
plot_series_mmh = genfil_mmh[1][voi, :]
plot_series_opt = opt_fil[1][voi, :]
plot_obs = X[voi,:]


p1 = plot(plot_series_gem, label = "GEM Causal Filter")
plot!(plot_obs, label = "Truth")
plot!(plot_series_mmh, label = "GEMMMH Causal Filter")
plot!(plot_series_opt, label = "Optimal Filter")

plot(p1, legend = :outerright, size = (1000, 750))
