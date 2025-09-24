using Test: @test, @testset
using TestExtras: @constinferred
using LinearAlgebra: LinearAlgebra, isdiag
using BlockSparseArrays: BlockSparseArray, eachstoredblock
using FusionTensors:
  FusionTensor, FusionTensorAxes, domain_axes, codomain_axes, to_fusiontensor
using GradedArrays:
  SU2, U1, checkspaces, checkspaces_dual, dual, flip, gradedrange, isdual, space_isequal
using TensorAlgebra: svd, tuplemortar
using MatrixAlgebraKit: truncrank

include("setup.jl")

function check_svd(ft, u, s, v)
  check_sanity(u)
  check_sanity(s)
  check_sanity(v)
  @test u isa FusionTensor{eltype(ft),ndims_codomain(ft) + 1}
  @test s isa FusionTensor{real(eltype(ft)),2}
  @test !isdual(axes(s, 1))
  @test isdual(axes(s, 2))
  @test v isa FusionTensor{eltype(ft),ndims_domain(ft) + 1}
  @test checkspaces(codomain_axes(ft), codomain_axes(u))
  @test checkspaces_dual(domain_axes(u), codomain_axes(s))
  @test checkspaces_dual(domain_axes(s), codomain_axes(s))
  @test checkspaces_dual(domain_axes(s), codomain_axes(v))
  @test checkspaces(domain_axes(v), domain_axes(ft))

  foreach(eachstoredblock(data_matrix(s))) do b
    @test isdiag(b)
    @test all(b .>= 0)
  end
  foreach(eachstoredblock(data_matrix(u))) do b
    @test b' * b ≈ LinearAlgebra.I(size(b, 2))
  end
  foreach(eachstoredblock(data_matrix(v))) do b
    @test b * b' ≈ LinearAlgebra.I(size(b, 1))
  end
  return true
end

function check_full_svd(ft, u, s, v)
  check_svd(ft, u, s, v)
  @test norm(s) ≈ norm(ft)
  @test ft ≈ u * s * v
  return true
end

@testset "full SVD (elt=$elt)" for elt in (Float64, ComplexF64)
  # matrix
  g1 = gradedrange([U1(1) => 1, U1(2) => 3])
  g2 = gradedrange([U1(1) => 2, U1(2) => 2])
  ft = randn(elt, FusionTensorAxes((g1,), (dual(g2),)))
  u, s, v = @constinferred svd(ft, (1, 2), (1,), (2,))
  @test check_full_svd(ft, u, s, v)
  @test checkspaces(codomain_axes(s), (gradedrange([U1(1) => 1, U1(2) => 2]),))

  # matrix with non canonical arrows
  g1 = gradedrange([U1(1) => 1, U1(2) => 3])
  ft = randn(FusionTensorAxes((dual(g1),), (flip(g2),)))
  u, s, v = svd(ft, (1, 2), (1,), (2,))
  @test check_full_svd(ft, u, s, v)
  @test checkspaces(codomain_axes(s), (gradedrange([U1(-2) => 2, U1(-1) => 1]),))

  # 4-dim tensor
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
  ft = randn(FusionTensorAxes((g1, dual(g2)), (dual(g3), g4)))
  u, s, v = @constinferred svd(ft, (1, 2, 3, 4), (1, 2), (3, 4))
  @test check_full_svd(ft, u, s, v)
  @test checkspaces(codomain_axes(s), (gradedrange([U1(-1) => 2, U1(0) => 4, U1(1) => 3]),))

  # non-abelian
  g2 = gradedrange([SU2(1//2) => 1])
  sds22 = reshape(
    [
      [0.25, 0.0, 0.0, 0.0]
      [0.0, -0.25, 0.5, 0.0]
      [0.0, 0.5, -0.25, 0.0]
      [0.0, 0.0, 0.0, 0.25]
    ],
    (2, 2, 2, 2),
  )
  ft = to_fusiontensor(sds22, (g2, g2), (dual(g2), dual(g2)))
  u, s, v = @constinferred svd(ft, (1, 2, 3, 4), (1, 2), (3, 4))
  @test check_full_svd(ft, u, s, v)
  @test checkspaces(codomain_axes(s), (gradedrange([SU2(0) => 1, SU2(1) => 1]),))
  @test data_matrix(s) ≈ [3/4 0; 0 1/4]
end

@testset "Abelian truncated SVD (elt=$elt)" for elt in (Float64, ComplexF64)
  # only test matrices; matricize handles higer dimension tensors
  g1 = gradedrange([U1(1) => 5, U1(2) => 6])
  g2 = gradedrange([U1(1) => 8, U1(2) => 4])
  ft = randn(elt, FusionTensorAxes((g1,), (dual(g2),)))

  trunc = truncrank(5)

  u, s, v = svd(ft, (1, 2), (1,), (2,))
  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc)
  @test check_svd(ft, utrunc, strunc, vtrunc)
  @test size(strunc) == tuplemortar(((5,), (5,)))
  ft_trunc = utrunc * strunc * vtrunc
  # TBD check norm(ft-ft_trunc)?

  # trunc above total number of values
  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc=truncrank(100))
  @test check_full_svd(ft, utrunc, strunc, vtrunc)

  # keep 0 values
  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc=truncrank(0))
  # TBD fix isdual(length = 0) or not worth it?
  #@test check_svd(ft, utrunc, strunc, vtrunc)
  @test size(strunc) == tuplemortar(((0,), (0,)))

  # add forbidden blocks
  g1 = gradedrange([U1(0) => 2, U1(1) => 5, U1(2) => 6])
  g2 = gradedrange([U1(1) => 8, U1(2) => 4, U1(4) => 3])
  ft = randn(FusionTensorAxes((g1,), (dual(g2),)))
  u, s, v = svd(ft, (1, 2), (1,), (2,))
  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc)
  @test check_svd(ft, utrunc, strunc, vtrunc)
  @test size(strunc) == tuplemortar(((5,), (5,)))
  ft_trunc = utrunc * strunc * vtrunc
end

random_unitary(t::Tuple) = random_unitary(t...)
random_unitary(m, n) = Matrix(LinearAlgebra.qr(randn(m, n)).Q)

@testset "Non-abelian truncated SVD" begin
  g1 = gradedrange([SU2(0) => 1, SU2(1//2) => 6, SU2(1) => 8])
  g2 = gradedrange([SU2(1//2) => 5, SU2(1) => 6, SU2(3//2) => 1])
  ft = zeros(FusionTensorAxes((g1,), (dual(g2),)))

  u1 = random_unitary(6, 5)
  v1 = random_unitary(5, 5)
  s1 = LinearAlgebra.diagm(1:5)
  u2 = random_unitary(8, 6)
  v2 = random_unitary(6, 6)
  s2 = LinearAlgebra.diagm(6:11)
  data_matrix(ft)[Block(2, 1)] .= u1 * s1 * v1
  data_matrix(ft)[Block(3, 2)] .= u2 * s2 * v2

  u, s, v = svd(ft, (1, 2), (1,), (2,))
  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc=truncrank(5))
  @test check_svd(ft, utrunc, strunc, vtrunc)
  @test size(strunc) == tuplemortar(((3,), (3,)))
  @test space_isequal(axes(strunc, 1), gradedrange([SU2(1) => 1]))

  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc=truncrank(6))
  @test check_svd(ft, utrunc, strunc, vtrunc)
  @test size(strunc) == tuplemortar(((6,), (6,)))
  @test space_isequal(axes(strunc, 1), gradedrange([SU2(1) => 2]))
  ft_trunc = utrunc * strunc * vtrunc
  # TBD check norm(ft-ft_trunc)?

  # trunc above total number of values
  utrunc, strunc, vtrunc = svd(ft, (1, 2), (1,), (2,); trunc=truncrank(500))
  @test check_full_svd(ft, utrunc, strunc, vtrunc)
end
