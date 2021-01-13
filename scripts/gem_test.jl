using DrWatson
using Optim

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
function em_dr()
    A_gem = rand(2, 2)
    θ = 1.0
    for s in 1:dr_steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3)
    end
    return A_gem
end


# A_graphem1 = perf_em()
# A_graphem2 = perf_em()
# A_graphem3 = perf_em()
# A_graphem4 = perf_em()

A_graphem_dr = em_dr()

qco = Q_func(Y, rand(2,2), H, m0, P, Q, R, l1_penalty)
θ = 1.0

pf1 = isotropic_proxf1(A, θ, T, Q, qco[4])
nit_pf1 = proxf1(A, θ, T, Q, qco[4])
