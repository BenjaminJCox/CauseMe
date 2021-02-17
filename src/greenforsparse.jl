using DrWatson
using Distributions
using Random
using Turing: filldist
using MLBase

function kalman_sample_sparse(
    P,
    Q,
    R,
    H,
    m0,
    y,
    A0,
    sparse_inds::Array{CartesianIndex{2},1};
    steps::Integer = 1_000,
    MH_scale::Float64 = 0.1,
    penalty::Function = x -> exp(2) .* norm(x, 1),
    no_change_prob::Float64 = 0.9,
    sparser_prob::Float64 = 0.8,
    corrections::Bool = true,
    subcorrections::Bool = true,
)
    @assert 0 < no_change_prob < 1
    @assert 0 < sparser_prob < 1
    N = size(A0, 1)
    max_sparse_n = length(sparse_inds)
    A_samples = zeros(N, N, steps)
    symwald = Laplace(0, MH_scale)
    t_pert = filldist(symwald, N, N)

    currently_sparse = BitArray(zeros(max_sparse_n))
    sparse_ind_inds = 1:max_sparse_n


    l_pya = perform_kalman(y, A0, H, m0, P, Q, R)[3]
    l_pyap = copy(l_pya)
    l_accrat = 0.0

    r_draw_diff = zero(A0)
    can_sparser = true
    can_denser = false
    doing_sparser = false
    correction = 0.0

    A = copy(A0)
    A_unzero = copy(A0)
    pi_p = log(1.0 - sparser_prob) - log(sparser_prob)
    for n = 1:steps
        # change sparseness?
        correction = 0.0
        Ap = copy(A)
        currently_sparse_op = copy(currently_sparse)
        correction = 0.0
        n_sparse = sum(currently_sparse)
        n_dense = max_sparse_n - n_sparse
        can_sparser = n_sparse < max_sparse_n
        can_denser = n_sparse > 0
        if (rand() < no_change_prob)
            # no change sparseness, draw from walk
            change_indices = Not(sparse_inds[currently_sparse])
            r_draw_diff .= rand(t_pert)
            Ap[change_indices] .+= r_draw_diff[change_indices]
            l_pyap = perform_kalman(y, Ap, H, m0, P, Q, R)[3]
            l_accrat = l_pyap - l_pya - penalty(Ap) + penalty(A)
        else
            # must look over all corrections, when these are allowed things get weird
            if (can_sparser && can_denser)
                doing_sparser = (rand() < sparser_prob)
                if doing_sparser
                    correction = corrections * pi_p + subcorrections * (log(n_dense) - log(n_sparse + 1))
                else
                    correction = corrections * -pi_p + subcorrections * (log(n_sparse) - log(n_dense + 1))
                end
                # if stepping to sparsest
                if doing_sparser && (n_sparse == max_sparse_n - 1)
                    correction = corrections * -log(sparser_prob) - subcorrections * log(max_sparse_n)
                    # if stepping to densest
                elseif !doing_sparser && (n_sparse == 1)
                    correction = corrections * -log(1.0 - sparser_prob) - subcorrections * log(max_sparse_n)
                end
                # if stepping from densest
            elseif can_sparser
                correction = corrections * log(1.0 - sparser_prob) + subcorrections * log(max_sparse_n)
                doing_sparser = true
                # if stepping from sparsest
            elseif can_denser
                correction = corrections * log(sparser_prob) + subcorrections * log(max_sparse_n)
                doing_sparser = false
            else
                @info("This is really bad")
            end
            if doing_sparser
                # get sparser, drop higher terms
                # draw from indices of not yet sparse
                if (length(sparse_ind_inds[Not(currently_sparse)]) == 1)
                    make_sparse_ind = sparse_ind_inds[Not(currently_sparse)][1]
                else
                    make_sparse_ind = rand(sparse_ind_inds[Not(currently_sparse)])[1]
                end
                A_unzero[sparse_inds[make_sparse_ind]] = A[sparse_inds[make_sparse_ind]]
                Ap[sparse_inds[make_sparse_ind]] = 0.0
                currently_sparse_op[make_sparse_ind] = true
                l_pyap = perform_kalman(y, Ap, H, m0, P, Q, R)[3]
                l_accrat =
                    l_pyap - l_pya - penalty(Ap) +
                    penalty(A) +
                    correction +
                    logpdf(symwald, A[sparse_inds[make_sparse_ind]])
            else
                # get denser, draw higher terms from walk (about zero)
                if (length(sparse_ind_inds[currently_sparse]) == 1)
                    make_dense_ind = sparse_ind_inds[currently_sparse][1]
                else
                    make_dense_ind = rand(sparse_ind_inds[currently_sparse])[1]
                end
                tper = rand(symwald)
                Ap[sparse_inds[make_dense_ind]] = tper
                # Ap[sparse_inds[make_dense_ind]] = A_unzero[sparse_inds[make_dense_ind]] + tper
                currently_sparse_op[make_dense_ind] = false
                l_pyap = perform_kalman(y, Ap, H, m0, P, Q, R)[3]
                l_accrat =
                    l_pyap - l_pya - penalty(Ap) + penalty(A) + correction -
                    logpdf(symwald, tper)
            end
        end
        l_rand = log(rand())
        if (l_rand < l_accrat)
            A = copy(Ap)
            l_pya = copy(l_pyap)
            currently_sparse = copy(currently_sparse_op)
        end
        A_samples[:, :, n] .= A
    end
    return A_samples
end

function prec_rec_serjmcmc(true_A, sp_A; threshold::Float64 = 0.40)
    # @assert 0.0 .<= sp_A .<= 1.0
    true_sparse = (true_A .== 0.0)
    est_sparse = (sp_A .>= threshold)
    ts_vec = vec(true_sparse)
    es_vec = vec(est_sparse)
    eroc = roc(ts_vec, es_vec)
    prec = precision(eroc)
    rec = recall(eroc)
    f1 = f1score(eroc)
    return @dict prec rec f1
end

function prec_rec_graphem(true_A, gem_A)
    true_sparse = (true_A .== 0.0)
    est_sparse = (abs.(gem_A) .== 0.0)
    ts_vec = vec(true_sparse)
    es_vec = vec(est_sparse)
    eroc = roc(ts_vec, es_vec)
    prec = precision(eroc)
    rec = recall(eroc)
    f1 = f1score(eroc)
    return @dict prec rec f1
end
