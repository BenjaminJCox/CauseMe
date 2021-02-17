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
include(srcdir("greenforsparse.jl"))

# certified random number
Random.seed!(0x8d8e1b4c2169a717f8c9fcf)

γ = 0.9
l1_penalty(A) = γ * norm(A, 1)


# A = [0.8 0.2 0.0; 0.0 0.7 0.3; 0.1 0.0 0.9]
A = [0.8 0.0 0.0; 0.0 0.7 0.0; 0.0 0.0 0.9]
# A = [0.4 0.2 0.4; 0.3 0.4 0.3; 0.1 0.3 0.6]
Q = Matrix(1 .* I(3))
R = Matrix(1 .* I(3))
H = P =  Matrix(1.0 .* I(3))

m0 = ones(3)

T = 100

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

slarac_score = perform_slarac(Matrix(Y'), 1, 10_000)
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
)

    out_A = 0.0 .* A0
    N = size(A0, 1)
    A = copy(A0)
    A′ = copy(A)
    M = zeros(size(A))

    pert_dist = filldist(Laplace(0, 0.1), N, N)
    penalty(a) = exp(1) .* norm(a, 1)

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

pot_sparse = findall(abs.(genA_mmh) .< 0.3)
@info(pot_sparse)
stp = 25_000
gf_sparse = kalman_sample_sparse(P, Q, R, H, m0, Y, genA_mmh, pot_sparse, steps = stp, no_change_prob = 0.7, sparser_prob = 0.8, penalty = x -> exp(2) .* norm(x, 1))

burnin = 15_000
n_s = stp - burnin + 1

gfs_mean = mean(gf_sparse[:, :, burnin:end], dims = 3)[:, :, 1]
num_sparse = sum(gf_sparse[:, :, burnin:end] .== 0.0, dims = 3)[:, :, 1] ./ n_s
display(gfs_mean)
display(num_sparse)

genfil_gem = perform_kalman(Y, A_graphem_dr, H, m0, P, Q, R)
genfil_mmh = perform_kalman(Y, genA_mmh, H, m0, P, Q, R)
genfil_spmmh = perform_kalman(Y, gfs_mean, H, m0, P, Q, R)
opt_fil = perform_kalman(Y, A, H, m0, P, Q, R)

vois = 1:3
plot_arr = Array{Any,1}(undef, length(vois))
for voi in vois
    plot_series_gem = genfil_gem[1][voi, :]
    plot_series_mmh = genfil_mmh[1][voi, :]
    plot_series_spmmh = genfil_spmmh[1][voi, :]
    plot_series_opt = opt_fil[1][voi, :]
    plot_obs = X[voi, :]

    p1 = plot(plot_obs, label = "Truth")
    plot!(plot_series_gem, label = "GEM Filter")
    # plot!(plot_series_mmh, label = "MMH Filter")
    plot!(plot_series_spmmh, label = "SeMMH Filter")
    plot!(plot_series_opt, label = "Optimal Filter")
    plot_arr[voi] = p1
end


norm(A - A_graphem_dr, 2)
norm(A - genA_mmh, 2)
norm(A - gfs_mean, 2)

# latexify(gfs_mean, fmt = "%.4f")

function T_test(T_vec, A, P, Q, R, H, m0, Y)
    nT = length(T_vec)
    fscores_serjmcmc = Array{Float64,1}(undef,nT)
    fscores_graphem = Array{Float64,1}(undef,nT)
    @assert maximum(T_vec) .<= size(Y, 2)
    @info("Spinning off threads!")
    Threads.@threads for t_ind in 1:nT
        T = T_vec[t_ind]
        @info("Currently running for", T)
        y = Y[:, 1:T]

        graphem_runs_full = 25
        genem_samples = zeros(3, 3, graphem_runs_full)
        for i = 1:graphem_runs_full
            genem_samples[:, :, i] .+= em_dr(3, dr_steps, y, H, m0, P, Q, R, γ)
        end
        A_graphem = mean(genem_samples, dims = 3)[:, :, 1]
        fscores_graphem[t_ind] = prec_rec_graphem(A, A_graphem)[:f1]


        genA_mmh = kalmanesq_MMH_A_sparse(P, Q, R, H, m0, y; steps = 10_000)
        pot_sparse = findall(abs.(genA_mmh) .< 0.3)
        stp = 20_000
        gf_sparse = kalman_sample_sparse(P, Q, R, H, m0, y, genA_mmh, pot_sparse, steps = stp, no_change_prob = 0.9, sparser_prob = 0.6)

        burnin = 12_000
        n_s = stp - burnin + 1

        gfs_mean = mean(gf_sparse[:, :, burnin:end], dims = 3)[:, :, 1]
        num_sparse = sum(gf_sparse[:, :, burnin:end] .== 0.0, dims = 3)[:, :, 1] ./ n_s
        @info(T, num_sparse)

        fscores_serjmcmc[t_ind] = prec_rec_serjmcmc(A, num_sparse, threshold = 0.3)[:f1]
    end
    plt = plot(T_vec, fscores_graphem, label = "GraphEM", xlabel = "Series Length", ylabel = "F1 Score", legend = :outerbottom)
    plot!(T_vec, fscores_serjmcmc, label = "SERJMCMC")
    @dict plt fscores_graphem fscores_serjmcmc
end

# tvec = [10, 30, 40, 50, 70, 100]

# tested = T_test(tvec, A, P, Q, R, H, m0, Y)
# tested_threaded = T_test(tvec, A, P, Q, R, H, m0, Y)
# tested_threaded[:plt]

norm(gfs_mean, 1) * exp(2)
genfil_spmmh[3]

@info prec_rec_serjmcmc(A, num_sparse, threshold = 0.3)
@info prec_rec_graphem(A, A_graphem_dr)
plot(plot_arr..., legend = :outerright, size = (1000, 750), layout = (3, 1))
