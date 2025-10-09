using TensorAlgebra: contract, svd
using MatrixAlgebraKit: truncrank, trunctol
using GradedArrays: O2, dual, gradedrange
using FusionTensors: FusionTensorAxes, to_fusiontensor
using LinearAlgebra: normalize!


# =============================  simulation parameters   ==========================================
n_sites = 100
τ = 1.0e-3
Dmax = 20
rtol = 1.0e-11
β = 1.0

# =============================  TEBD algorithm   ==========================================
function absorb_weights(u, s, v, ::Val{:right})
    labels_out = tuplemortar((-1, ntuple(identity, ndims(v) - 1)))
    labels_v = tuplemortar((-2, ntuple(identity, ndims(v) - 1)))
    return u, contract(labels_out, s, (-1, -2), v, labels_v)
end

function absorb_weights(u, s, v, ::Val{:left})
    labels_out = tuplemortar((-1, ntuple(identity, ndims(v) - 1)))
    labels_u = tuplemortar((-2, ntuple(identity, ndims(v) - 1)))
    return contract(labels_out, u, labels_u, s, (-1, -2)), v
end

function tebd_step(left, right, gate, trunc, absorb)
    # phys-left-right
    leftright = contract((1, 3, 2, 4), left, (1, 2, -1), right, (2, 3, -1))
    t = contract((1, 3, 2, 4), leftright, (1, 2, 3, 4), gate, (1, 2, 3, 4))
    u, s, v = svd(t, (1, 2, 3, 4), (1, 2), (3, 4), trunc)
    normalize!(s)
    new_left, new_right_mat = absorb_weights(u, s, v, absorb)
    new_right = permutedims(new_right_mat, (2, 1), (3,))
    return new_left, new_right
end

function canonical_left(left, right)
    qmat, rmat = qr(left, (1, 2, 3, 4), (1, 2), (3, 4))
    permuted_right = permutedims(right, (2, 1, 3))
    new_right_mat = rmat * reshape(permuted_right, DM, dR * DR)
    new_right = permutedims(reshape(new_right_mat, (:, dR, DR)), (2, 1, 3))
    return new_left, new_right
end

function canonical_right(left, right)
    lmat, qmat = lq(right)
    new_right = permutedims(qmat, (2, 1, 3))
    new_left_mat = left * lmat
    new_left = reshape(new_left_mat, (dL, DL, :))
    return new_left, new_right
end

function canonical(mps0, i)
    n = length(mps0)
    newmps = deepcopy(mps0)
    for j in 1:(i - 1)
        newmps[j], newmps[j + 1] = canonical_left(newmps[j], newmps[j + 1])
    end
    for j in n:-1:(i + 1)
        newmps[j - 1], newmps[j] = canonical_right(newmps[j - 1], newmps[j])
    end
    return newmps
end

function tebd_sweep(mps0, gate, trunc)
    n = length(mps0)
    @assert iseven(n)
    newmps = deepcopy(mps0)
    for i in 1:(n ÷ 2 - 1)
        i1, i2, i3 = 2i - 1, 2i, 2i + 1
        newmps[i1], temp = tebd_step(newmps[i1], newmps[i2], gate, trunc, Val(:right))
        newmps[i2], newmps[i3] = canonical_left(temp, newmps[i3])
    end
    newmps[n - 1], newmps[n] = tebd_step(newmps[n - 1], newmps[n], gate, trunc, Val(:left))
    for i in 1:(n ÷ 2 - 1)
        i1, i2, i3 = n - 2i - 1, n - 2i, n - 2i + 1
        temp, newmps[i3] = tebd_step(newmps[i2], newmps[i3], gate, trunc, Val(:left))
        newmps[i1], newmps[i2] = canonical_right(newmps[i1], temp)
    end
    return newmps
end

function env_1site(mps, i)
    canonicalized = canonical(mps, i)
    center = canonicalized[i]
    right = canonicalized[i + 1]
    @tensor env[-1, -2, -3, -4] :=
        center[-1, 1, 2] * right[-2, 2, 3] * conj(center)[-3, 1, 4] * conj(right)[-4, 4, 3]
    return env
end

function env_2sites(mps, i)
    canonicalized = canonical(mps, i)
    left = canonicalized[i - 1]
    center = canonicalized[i]
    right = canonicalized[i + 1]
    @tensor ket[-4, -1, -2, -3, -5] := left[-1, -4, 1] * center[-2, 1, 2] * right[-3, 2, -5]
    @tensor env[-1, -2, -3, -4, -5, -6] := ket[1, -1, -2, -3, 2] * conj(ket)[1, -4, -5, -6, 2]
    return env
end

# ===========================  initialize XX chain  ========================================
trunc_strategy = truncrank(Dmax) # & trunctol(; rtol=1e-11, atol=0)

hxx_mat = [
    0.0 0.0 0.0 0.0
    0.0 0.0 -0.5 0.0
    0.0 -0.5 0.0 0.0
    0.0 0.0 0.0 0.0
]
hxx_tens = reshape(hxx_mat, (2, 2, 2, 2))
gd = gradedrange([O2(1 // 2) => 1])
gD = gradedrange([O2(0) => 1])
gate = exp(-tau * hxx)
hxx = to_fusiontensor(hxx_tens, (gd, gd), (dual(gd), dual(gd)))
t0 = zeros(FusionTensorAxes((gd, dual(gd)), (gD, dual(gD))))  # just 1 element
for (f1, f2) in fusiontrees(t0)
    t0[f1, f2] .= √2
end

ψ = fill(t0, (n_sites,))

# ===========================  run TEBD ========================================
for i in Int(β / τ)
    ψ = tebd_sweep(ψ, gate, trunc_strategy)
end

# ===========================  compute energy ========================================

# =========================== compare with exact result ====================================
exact_energy = 0.0
