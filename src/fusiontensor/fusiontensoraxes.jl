using GradedArrays: GradedArrays, AbstractSector, TrivialSector, dual, sector_type, trivial
using TensorAlgebra: BlockedTuple
using TensorProducts: ⊗

# =======================================  Misc  ===========================================

dummy_axis() = dummy_axis(TrivialSector)
dummy_axis(x) = dummy_axis(typeof(x))
dummy_axis(::Type{T}) where {T} = dummy_axis(sector_type(T))
dummy_axis(::Type{S}) where {S<:AbstractSector} = gradedrange([trivial(S) => 1])

# ====================================  Definitions  =======================================

struct FusionTensorAxes{BT<:BlockedTuple{2}}
  outer_axes::BT

  function FusionTensorAxes{BT}(bt) where {BT}
    return new{BT}(bt)
  end
end

# =====================================  Accessors  ========================================

outer_axes(fta::FusionTensorAxes) = fta.outer_axes

# ====================================  Constructors  ======================================

function FusionTensorAxes(
  bt::BlockedTuple{2,<:Any,<:NTuple{<:Any,<:AbstractGradedUnitRange}}
)
  return FusionTensorAxes{typeof(bt)}(bt)
end

# ==================================  Base interface  ======================================

for f in [:(length), :(axes), :(Tuple), :(iterate)]
  @eval Base.$f(fta::FusionTensorAxes) = $f(outer_axes(fta))
end

for f in [:(getindex), :(iterate)]
  @eval Base.$f(fta::FusionTensorAxes, i) = $f(outer_axes(fta), i)
end

# ==============================  GradedArrays interface  ==================================

function GradedArrays.sector_type(fta::FusionTensorAxes)
  return isempty(fta) ? TrivialSector : sector_type(first(fta))
end

# ==============================  FusionTensor interface  ==================================

codomain_axes(fta::FusionTensorAxes) = fta[Block(1)]

domain_axes(fta::FusionTensorAxes) = fta[Block(2)]

dummy_axis(ft::FusionTensorAxes) = dummy_axis(sector_type(ft))

function codomain_axis(fta::FusionTensorAxes)
  if ndims_codomain(fta) == 0
    return dummy_axis(fta)
  end
  return ⊗(codomain_axes(fta)...)
end

function domain_axis(fta::FusionTensorAxes)
  if ndims_domain(fta) == 0
    return dummy_axis(fta)
  end
  return dual(⊗(dual.(domain_axes(fta))...))
end

ndims_codomain(fta::FusionTensorAxes) = length(codomain_axes(fta))

ndims_domain(fta::FusionTensorAxes) = length(domain_axes(fta))
