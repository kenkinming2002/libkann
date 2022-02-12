chinesechess_sources = [
  'Population.cpp',
  'Human.cpp',
  'Renderer.cpp',
  'main.cpp',
  'Game.cpp',
  'NewGame.cpp',
  'Board.cpp',
  'App/Hell.cpp',
  'App/Match/AgentPlayer.cpp',
  'App/Match/Agents.cpp',
  'App/Match.cpp',
  'Agent.cpp',
  'AIAgent.cpp',
  'HumanAgent.cpp',
  'Match.cpp'
]
sfml_dep = dependency('sfml-all')
chinesechess = executable('chinesechess', chinesechess_sources,
  dependencies : [libkann_dep, sfml_dep],
  cpp_args : ['-fopenmp'],
  link_args : ['-fopenmp']
)

