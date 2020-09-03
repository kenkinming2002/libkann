
libkann_sources = [
  'Connection.cpp',
  'NeuralNetwork.cpp',
  'Layer.cpp',
  'RecurrentNeuralNetwork.cpp',
  'ActivationFunction.cpp'
]

eigen_dep = dependency('eigen3')

libkann_lib = static_library('libkann', libkann_sources,
  include_directories : ['../include'], 
  dependencies        : [eigen_dep],
  cpp_args            : ['-fvisibility=hidden']
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [eigen_dep],
  link_with : libkann_lib, 
)
