using DrWatson
using Distributions
using Random
using Turing: filldist

function kalman_sample_sparse(P, Q, R, H, m0, y, A0, sparse_inds::Array{CartesianIndex{2},1}; steps::Integer = 1_000, MH_scale::Float64 = 0.1, penalty::Function = x -> exp(1) .* norm(x, 1), change_prob::Float64 = 0.5, sparser_prob::Float64 = 0.5)
    @assert 0 < change_prob < 1
    @assert 0 < sparser_prob < 1
    N = size(A0, 1)
    max_sparse_n = length(sparse_inds)
    A_samples = zeros(N, N, steps)
    θ_pert = filldist(Normal(0, MH_scale), N, N)

    currently_sparse = BitArray(zeros(max_sparse_n))

    l_pya = perform_kalman(y, A0, H, m0, P, Q, R)[3]
    l_pyap = copy(l_pya)
    l_accrat = 0.0

    A = copy(A0)
    for n = 1:steps
        # change sparseness?
        A′ = copy(A)
        if (rand() < change_prob)
            # no change sparseness, draw from walk
            change_indices = Not(sparse_inds[currently_sparse])
            A′[change_indices] .+= rand(θ_pert)[change_indices]
            l_pyap = perform_kalman(y, A′, H, m0, P, Q, R)[3]
            l_accrat = l_pyap - l_pya - penalty(A′) + penalty(A)
        else
            if (rand() < sparser_prob) && (sum(currently_sparse) < max_sparse_n)
                # get sparser, drop higher terms
            else
                # get denser, draw higher terms from walk (about zero)
            end
        end
        l_rand = log(rand())
        if (l_rand < l_accrat)
            A = copy(A′)
            l_pya = copy(l_pyap)
        end
        A_samples[:, :, n] .= A
    end

end
