using TensorAlgebra: TensorAlgebra

using GradedArrays: eachblockaxis, mortar_axis

function TensorAlgebra.MatrixAlgebra.svd!(A::FusionMatrix; kwargs...)
  um, sm, vm = TensorAlgebra.MatrixAlgebra.svd!(data_matrix(A); kwargs...)
  sm_blocks = collect(map(b -> Int(first(Tuple(b))), eachblockstoredindex(sm)))
  s_axes = map(
    Iterators.filter(
      b -> Int(last(Tuple(b))) in sm_blocks, Iterators.flatten(eachblockstoredindex(um))
    ),
  ) do b
    return sectors(axes(A, 1))[Int(first(Tuple(b)))] => size(um[b], 2)
  end

  # TODO prune forbidden blocks
  s_axis = gradedrange(s_axes)
  u = FusionTensor(um, (axes(A, 1),), (dual(s_axis),))
  s = FusionTensor(sm, (s_axis,), (dual(s_axis),))
  v = FusionTensor(vm, (s_axis,), (axes(A, 2),))
  return u, s, v
end
