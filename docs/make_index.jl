using Literate: Literate
using FusionTensors: FusionTensors

Literate.markdown(
  joinpath(pkgdir(FusionTensors), "examples", "README.jl"),
  joinpath(pkgdir(FusionTensors), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
