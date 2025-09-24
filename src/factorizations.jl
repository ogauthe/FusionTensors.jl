using TensorAlgebra: TensorAlgebra, MatrixAlgebra
using MatrixAlgebraKit:
  MatrixAlgebraKit,
  TruncationStrategy,
  TruncationKeepSorted,
  default_svd_algorithm,
  findtruncated,
  svd_compact!,
  svd_trunc!
using GradedArrays:
  GradedArrays,
  BlockDiagonalAlgorithm,
  eachblockaxis,
  findfirstblock_sector,
  isdual,
  mortar_axis,
  quantum_dimension,
  sector
using BlockSparseArrays:
  BlockPermutedDiagonalAlgorithm, blockdiagonalize, eachblockstoredindex
using LinearAlgebra: LinearAlgebra, diag

function MatrixAlgebraKit.default_svd_algorithm(::Type{<:FusionMatrix}; kwargs...)
  return BlockPermutedDiagonalAlgorithm() do block
    return default_svd_algorithm(block; kwargs...)
  end
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_compact!), A::FusionMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  # cannot be done in default_svd_algorithm which acts on type domain.
  @assert !isdual(axes(A, 1)) && isdual(axes(A, 2))

  # we know initialized blocks are exactly invariant blocks
  m = data_matrix(A)
  s_axes = map(Iterators.flatten(eachblockstoredindex(m))) do b
    return sector(axes(A, 1)[first(Tuple(b))]) => min(size(m[b])...)
  end
  s_axis = gradedrange(s_axes)

  U = similar(A, eltype(A), ((axes(A, 1),), (dual(s_axis),)))
  S = similar(A, eltype(A), ((s_axis,), (dual(s_axis),)))
  Vᴴ = similar(A, eltype(A), ((s_axis,), (axes(A, 2),)))
  return U, S, Vᴴ
end

function BlockSparseArrays.blockdiagonalize(A::FusionMatrix)
  m = data_matrix(A)
  rowcolblocks = map(Iterators.flatten(eachblockstoredindex(m))) do b
    return Tuple(b)
  end
  rowselect = first.(rowcolblocks)
  colselect = last.(rowcolblocks)
  return m[rowselect, colselect], (rowselect, colselect)
end

function MatrixAlgebraKit.svd_compact!(
  A::FusionMatrix, (U, S, Vᴴ), alg::BlockPermutedDiagonalAlgorithm
)
  Ad, (rowselect, colselect) = blockdiagonalize(A)
  Um, Sm, Vᴴm = svd_compact!(Ad, BlockDiagonalAlgorithm(alg))
  data_matrix(U)[rowselect, :] .= Um
  data_matrix(S) .= Sm
  data_matrix(Vᴴ)[:, colselect] .= Vᴴm
  return U, S, Vᴴ
end

function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::Tuple{FusionTensor,FusionTensor,FusionTensor},
  strategy::TruncationStrategy,
)
  @show "HI MatrixAlgebraKit.truncate!"
  m = data_matrix(S)
  g0 = codomain_axis(S)
  vals = mapreduce(vcat, Iterators.flatten(eachblockstoredindex(m))) do b
    b1 = first(Tuple(b))
    d = quantum_dimension(sector(g0[b1]))
    return Multiplet.(diag(m[b]), b1, d)
  end
  sort!(vals; rev=true)
  indices_collection = findtruncated(vals, strategy)
  kept_vals = vals[indices_collection]
  kept_blocks = sort(unique(Block.(kept_vals)))
  kept_sectors = map(b -> sector(g0[b]), kept_blocks)
  # BlockIndexRange only allows contiguous ranges
  kept_blockwise = [count(==(b), Block.(kept_vals)) for b in kept_blocks]

  g = gradedrange(kept_sectors .=> kept_blockwise)
  Utrunc = similar(U, (codomain_axes(U), (dual(g),)))
  Strunc = similar(S, ((g,), (dual(g),)))
  Vᴴtrunc = similar(U, ((g,), domain_axes(Vᴴ)))

  # TBD: think in terms of block or of sectors?
  for (i, b) in enumerate(kept_blocks)
    r = Base.oneto(kept_blockwise[i])
    bu = findfirstblock_sector(codomain_axis(U), kept_sectors[i])
    bv = findfirstblock_sector(domain_axis(Vᴴ), kept_sectors[i])
    data_matrix(Utrunc)[bu, Block(i)] .= data_matrix(U)[bu, b][:, r]
    data_matrix(Strunc)[Block(i, i)] .= data_matrix(S)[b, b][r, r]
    data_matrix(Vᴴtrunc)[Block(i), bv] .= data_matrix(Vᴴ)[bv, b][r, :]
  end

  return Utrunc, Strunc, Vᴴtrunc
end

# ==========================================================================================

struct Multiplet{T<:Real,B,I}
  val::T
  blockval::B
  quantum_dimension::I
end

Base.real(m::Multiplet) = m.val
Base.real(::Type{M}) where {T,M<:Multiplet{T}} = T

Base.isless(m1::Multiplet, m2::Multiplet) = real(m1) < real(m2)

BlockArrays.Block(m::Multiplet) = m.blockval

# TBD use Base.length?
GradedArrays.quantum_dimension(m::Multiplet) = m.quantum_dimension

# TBD
LinearAlgebra.norm(m::Multiplet, p::Real=2) = abs(real(m)) * (quantum_dimension(m)^(1 / p))

function MatrixAlgebra.findtruncated(
  vals::AbstractVector{<:Multiplet}, strategy::TruncationKeepSorted
)
  Base.require_one_based_indexing(vals)
  issorted(vals; rev=true) || error("Not sorted.")
  cs = cumsum(quantum_dimension.(vals))
  k = findfirst(>(strategy.howmany), cs)
  isnothing(k) && return Base.oneto(length(vals))
  return Base.oneto(k - 1)  # remove last multiplet to have kept rank <= strategy.howmany
end
