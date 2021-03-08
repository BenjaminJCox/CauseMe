using DrWatson
using Optim
using Distributions
using Random
using DataFrames
using Plots
using CSV
using Turing: filldist
using BlockDiagonals

gr()

include(srcdir("workers.jl"))
include(srcdir("greenforsparse.jl"))

# certified random number
Random.seed!(0x8d8e1b4c2169a71cf)

γ = 1.0
l1_penalty(A) = γ * norm(A, 1)

A_blocks = [rand(2,2) .+ 2.0 for _ in 1:2]
a_dim = 4
A = BlockDiagonal(A_blocks)
A = Matrix(A)
A ./= 1.1 .* eigmax(A)
# A = [0.8 0.2 0.0; 0.0 0.7 0.3; 0.1 0.0 0.9]
Q = Matrix(1.0^2 .* I(a_dim))
R = Matrix(1.0^2 .* I(a_dim))
H = Matrix(1.0 .* I(a_dim))
P = Matrix(1e-8 .* I(a_dim))

m0 = ones(a_dim)

T = 100

X = zeros(a_dim, T)
Y = zeros(a_dim, T)

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

dr_steps = 25
function em_dr(dimA, steps, Y, H, m0, P, Q, R, γ)
    A_gem = rand(dimA, dimA)
    θ = 1.0
    for s = 1:steps
        Qf = Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-1, γ, maxiters = 5)
    end
    return A_gem
end

graphem_runs_full = 25
genem_samples = zeros(a_dim, a_dim, graphem_runs_full)
gem_stat = Dict(:f1 => 0.0, :spec => 0.0, :prec => 0.0, :rec => 0.0)

for i = 1:graphem_runs_full
    genem_samples[:, :, i] .+= em_dr(a_dim, dr_steps, Y, H, m0, P, Q, R, γ)
    gem_rs =  prec_rec_graphem(A, genem_samples[:, :, i])
    for st in keys(gem_rs)
        gem_stat[st] += gem_rs[st]
    end
end

for st in keys(gem_stat)
    gem_stat[st] /= graphem_runs_full
end

A_graphem_dr = mean(genem_samples, dims = 3)[:, :, 1]
display(A_graphem_dr)

graphem_runs_full_init = 25
genem_samples_init = zeros(a_dim, a_dim, graphem_runs_full)
for i = 1:graphem_runs_full_init
    genem_samples_init[:, :, i] .+= em_dr(a_dim, dr_steps, Y, H, m0, P, Q, R, 0.0)
end
A_graphem_dr_init = mean(genem_samples_init, dims = 3)[:, :, 1]
# display(A_graphem_dr)

# slarac_score = perform_slarac(Matrix(Y'), 1, 10_000)
# display(slarac_score[1])

function kalmanesq_MMH_A_sparse(
    P,
    Q,
    R,
    H,
    m0,
    observations;
    steps::Integer = 1000,
    A0 = 1.0 * Matrix(I(size(P, 1))),
    burnin::Integer = fld(steps, 2),
    σ = 0.1,
    λ = exp(1),
)

    out_A = 0.0 .* A0
    N = size(A0, 1)
    A = copy(A0)
    A′ = copy(A)
    M = zeros(size(A))

    pert_dist = filldist(Normal(0, σ), N, N)
    penalty(a) = λ .* norm(a, 1)

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

genA_mmh = kalmanesq_MMH_A_sparse(P, Q, R, H, m0, Y; steps = 10_000, σ = 0.1)
display(genA_mmh)

pot_sparse = findall(abs.(genA_mmh) .< 0.3)
@info(pot_sparse)
stp = 25_000
# gf_sparse = kalman_sample_sparse(P, Q, R, H, m0, Y, A_graphem_dr, pot_sparse, steps = stp, no_change_prob = 0.8, sparser_prob = 0.8, penalty = x -> exp(1.5) .* norm(x, 1))

burnin = 15_000
n_s = stp - burnin + 1

# gfs_mean = mean(gf_sparse[:, :, burnin:end], dims = 3)[:, :, 1]
# num_sparse = sum(gf_sparse[:, :, burnin:end] .== 0.0, dims = 3)[:, :, 1] ./ n_s
# display(gfs_mean)
# display(num_sparse)


n_runs = 10
ssmcmc_stat = Dict(:f1 => 0.0, :spec => 0.0, :prec => 0.0, :rec => 0.0)
gfs_mean = genA_mmh

for run in 1:n_runs
    pot_sparse = findall(abs.(gfs_mean) .< 0.3)
    gf_sparse = kalman_sample_sparse(P, Q, R, H, m0, Y, gfs_mean, pot_sparse, steps = stp, no_change_prob = 0.7, sparser_prob = 0.5, penalty = x -> exp(-1) .* norm(x, 1))
    global gfs_mean = mean(gf_sparse[:, :, burnin:end], dims = 3)[:, :, 1]
    num_sparse = sum(gf_sparse[:, :, burnin:end] .== 0.0, dims = 3)[:, :, 1] ./ n_s
    display(num_sparse)
    ssmcmc_rs = prec_rec_serjmcmc(A, num_sparse, threshold = 0.4)
    for st in keys(ssmcmc_rs)
        ssmcmc_stat[st] += ssmcmc_rs[st]
    end
end

for st in keys(gem_stat)
    ssmcmc_stat[st] /= n_runs
end




genfil_gem = perform_kalman(Y, A_graphem_dr, H, m0, P, Q, R)
genfil_mmh = perform_kalman(Y, genA_mmh, H, m0, P, Q, R)
genfil_spmmh = perform_kalman(Y, gfs_mean, H, m0, P, Q, R)
opt_fil = perform_kalman(Y, A, H, m0, P, Q, R)

vois = 1:a_dim
plot_arr = Array{Any,1}(undef, length(vois))
for voi in vois
    plot_series_gem = genfil_gem[1][voi, :]
    plot_series_mmh = genfil_mmh[1][voi, :]
    plot_series_spmmh = genfil_spmmh[1][voi, :]
    plot_series_opt = opt_fil[1][voi, :]
    plot_obs = X[voi, :]

    p1 = plot(plot_obs, label = "Truth")
    # plot!(plot_series_gem, label = "GEM Filter")
    # plot!(plot_series_mmh, label = "MMH Filter")
    plot!(plot_series_spmmh, label = "SSMCMC Filter")
    # plot!(plot_series_opt, label = "Optimal Filter")
    plot_arr[voi] = p1
end


norm(A - A_graphem_dr, 2)
norm(A - genA_mmh, 2)
norm(A - gfs_mean, 2)

# latexify(gfs_mean, fmt = "%.4f")

# tvec = [10, 30, 40, 50, 70, 100]

# tested = T_test(tvec, A, P, Q, R, H, m0, Y)
# tested_threaded = T_test(tvec, A, P, Q, R, H, m0, Y)
# tested_threaded[:plt]

# @info prec_rec_serjmcmc(A, num_sparse, threshold = 0.5)
# @info prec_rec_graphem(A, A_graphem_dr)
plot(plot_arr..., legend = false, size = (1000, 750), layout = (:, 1))

using LightGraphs, GraphRecipes

diA = DiGraph(A)
diRec = DiGraph(num_sparse .< 0.3)
diGem = DiGraph(A_graphem_dr .> 0)
plot(diA, names = 1:12, size = (500, 500), nodeshape = :circle, nodesize = 0.1)
savefig("true_graph.pdf")
plot(diRec, names = 1:12, size = (500, 500), nodeshape = :circle, nodesize = 0.1)
savefig("ssmcmc_graph.pdf")
plot(diGem, names = 1:12, size = (500, 500), nodeshape = :circle, nodesize = 0.1)
savefig("graphem_graph.pdf")

@info("Average GraphEM stats for", T, gem_stat)
@info("Average SSMCMC stats for", T, ssmcmc_stat)
