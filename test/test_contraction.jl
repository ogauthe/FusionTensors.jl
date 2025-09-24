using LinearAlgebra: mul!
using Test: @test, @testset, @test_broken

using BlockSparseArrays: BlockSparseArray
using FusionTensors:
  FusionMatrix, FusionTensor, FusionTensorAxes, domain_axes, codomain_axes
using GradedArrays: U1, dual, gradedrange
using TensorAlgebra: contract, matricize, permmortar, tuplemortar, unmatricize, unmatricize!

include("setup.jl")

@testset "matricize" begin
  # TODO add non-abelian test
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = randn(FusionTensorAxes((g1, g2), (dual(g3), dual(g4))))
  m = matricize(ft1, (1, 2), (3, 4))
  @test m isa FusionMatrix
  ft2 = unmatricize(m, axes(ft1))
  @test ft1 ≈ ft2

  biperm = permmortar(((3,), (1, 2, 4)))
  m2 = matricize(ft1, biperm)
  ft_dest = FusionTensor{eltype(ft1)}(undef, axes(ft1)[biperm])
  unmatricize!(ft_dest, m2, permmortar(((1,), (2, 3, 4))))
  @test ft_dest ≈ permutedims(ft1, biperm)
  @test ft_dest ≈ permutedims(ft1, biperm)

  ft2 = similar(ft1)
  unmatricize!(ft2, m2, biperm)
  @test ft1 ≈ ft2
end

@testset "contraction" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = FusionTensor{Float64}(undef, (g1, g2), (g3, g4))
  @test isnothing(check_sanity(ft1))

  ft2 = FusionTensor{Float64}(undef, dual.((g3, g4)), (g1,))
  @test isnothing(check_sanity(ft2))

  ft3 = ft1 * ft2  # tensor contraction
  @test isnothing(check_sanity(ft3))
  @test domain_axes(ft3) === domain_axes(ft2)
  @test codomain_axes(ft3) === codomain_axes(ft1)

  # test LinearAlgebra.mul! with in-place matrix product
  m1 = randn(FusionTensorAxes((g1,), (g2,)))
  m2 = randn(FusionTensorAxes((dual(g2),), (g3,)))
  m3 = FusionTensor{Float64}(undef, (g1,), (g3,))

  mul!(m3, m1, m2, 2.0, 0.0)
  @test m3 ≈ 2m1 * m2
end

@testset "TensorAlgebra interface" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = randn(FusionTensorAxes((g1, g2), (g3, g4)))
  ft2 = randn(FusionTensorAxes(dual.((g3, g4)), (dual(g1),)))
  ft3 = randn(FusionTensorAxes(dual.((g3, g4)), dual.((g1, g2))))

  ft4, legs = contract(ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test legs == tuplemortar(((1, 2), (5,)))
  @test isnothing(check_sanity(ft4))
  @test domain_axes(ft4) === domain_axes(ft2)
  @test codomain_axes(ft4) === codomain_axes(ft1)
  @test ft4 ≈ ft1 * ft2

  ft5 = contract((1, 2, 5), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test isnothing(check_sanity(ft5))
  @test ndims_codomain(ft5) === 3
  @test ndims_domain(ft5) === 0
  @test permutedims(ft5, (1, 2), (3,)) ≈ ft4

  ft6 = contract(tuplemortar(((1, 2), (5,))), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test isnothing(check_sanity(ft6))
  @test ft4 ≈ ft6

  @test permutedims(ft1, (), (1, 2, 3, 4)) * permutedims(ft3, (3, 4, 1, 2), ()) isa
    FusionTensor{Float64,0}
  ft7, legs = contract(ft1, (1, 2, 3, 4), ft3, (3, 4, 1, 2))
  @test legs == tuplemortar(((), ()))
  @test ft7 isa FusionTensor{Float64,0}

  # include permutations
  ft6 = contract(tuplemortar(((5, 1), (2,))), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test isnothing(check_sanity(ft6))
  @test permutedims(ft6, (2, 3), (1,)) ≈ ft4

  ft8 = contract(
    tuplemortar(((-3,), (-1, -2, -4))), ft1, (-1, 1, -2, 2), ft3, (-3, 2, -4, 1)
  )
  left = permutedims(ft1, (1, 3), (2, 4))
  right = permutedims(ft3, (4, 2), (1, 3))
  lrprod = left * right
  newft = permutedims(lrprod, (3,), (1, 2, 4))
  @test newft ≈ ft8
end
