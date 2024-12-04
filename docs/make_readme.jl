using Literate: Literate
using FusionTensors: FusionTensors

Literate.markdown(
  joinpath(pkgdir(FusionTensors), "examples", "README.jl"),
  joinpath(pkgdir(FusionTensors));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
