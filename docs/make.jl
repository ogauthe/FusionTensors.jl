using FusionTensors: FusionTensors
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(FusionTensors, :DocTestSetup, :(using FusionTensors); recursive=true)

include("make_index.jl")

makedocs(;
  modules=[FusionTensors],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="FusionTensors.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/FusionTensors.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/FusionTensors.jl", devbranch="main", push_preview=true
)
