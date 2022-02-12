chinesechess_sources = [
  'Population.cpp',
  'Human.cpp',
  'Renderer.cpp',
  'main.cpp',
  'Game.cpp',
  'Board.cpp',
  'AIAgent.cpp',
  'MinimaxAgent.cpp',
  'HumanAgent.cpp',
]
sfml_dep = dependency('sfml-all')
chinesechess = executable('chinesechess', chinesechess_sources,
  dependencies : [libkann_dep, sfml_dep],
  cpp_args : ['-fopenmp'],
  link_args : ['-fopenmp']
)

