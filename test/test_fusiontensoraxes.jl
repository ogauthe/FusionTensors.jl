using LinearAlgebra: norm
using Test: @test, @testset, @test_broken

using BlockArrays: blocklengths
using TensorAlgebra: BlockedTuple, tuplemortar

using FusionTensors: FusionTensorAxes, dummy_axis, ndims_domain, ndims_codomain
using GradedArrays:
  O2, U1, SectorProduct, SU2, dual, gradedrange, sector_type, space_isequal
using TensorAlgebra: BlockedPermutation, blockedperm, blockpermute, tuplemortar

@testset "misc FusionTensors.jl" begin
  @test space_isequal(dummy_axis(), gradedrange([TrivialSector() => 1]))
  @test space_isequal(dummy_axis(U1), gradedrange([TrivialSector() => 1]))
end

@testset "FusionTensorAxes" begin
  g2 = gradedrange([SU2(1//2) => 1])
  g2b = dual(g2)

  bt = tuplemortar(((g2, g2), (g2b, g2b)))
  fta = FusionTensorAxes(bt)

  @test fta isa FusionTensorAxes
  @test length(fta) == 2
  @test space_isequal(fta[1], g2)
  @test space_isequal(fta[2], g2)
  @test space_isequal(fta[3], g2b)
  @test space_isequal(fta[4], g2b)

  @test Tuple(fta) == Tuple(bt)
end
