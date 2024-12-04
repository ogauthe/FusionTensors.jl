@eval module $(gensym())
using FusionTensors: FusionTensors
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  Aqua.test_ambiguities([FusionTensors])
  Aqua.test_unbound_args(FusionTensors)
  Aqua.test_undefined_exports(FusionTensors)
  Aqua.test_project_extras(FusionTensors)
  Aqua.test_deps_compat(FusionTensors)
  Aqua.test_piracies(FusionTensors)
  Aqua.test_persistent_tasks(FusionTensors)

  # TODO fix test_stale_deps
  # TODO replace with Aqua.test_all
  # Aqua.test_stale_deps(FusionTensors)
end
end
