neuralevolution_sources = [
  'Config.cpp',
  'Renderer.cpp',
  'main.cpp',
  'Entity.cpp',
  'BerryBush.cpp',
  'App.cpp',
  'Creature.cpp',
  'World.cpp'
]

box2d_dep = dependency('box2d')
sfml_dep = dependency('sfml-all')
neuralevolution = executable('neuralevolution', neuralevolution_sources,
  dependencies : [libkann_dep, sfml_dep, nlohmann_json_dep, box2d_dep],
  cpp_args : ['-fopenmp'],
  link_args : ['-fopenmp']
)


