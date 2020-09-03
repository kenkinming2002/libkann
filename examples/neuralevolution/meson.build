neuralevolution_sources = [
  'Config.cpp',
  'PhantomBody.cpp',
  'Renderer.cpp',
  'main.cpp',
  'StaticBody.cpp',
  'BerryBush.cpp',
  'App.cpp',
  'Creature.cpp',
  'World.cpp',
  'Ray.cpp'
]
sfml_dep = dependency('sfml-all')
neuralevolution = executable('neuralevolution', neuralevolution_sources,
  dependencies : [libkann_dep, sfml_dep, nlohmann_json_dep],
  cpp_args : ['-fopenmp'],
  link_args : ['-fopenmp']
)


