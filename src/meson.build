
libkann_sources = [
  'neural_networks/NeuralNetwork.cpp',
  'neural_networks/RecurrentNeuralNetwork.cpp',
  'layers/WeightLayer.cpp',
  'layers/ActivationLayer.cpp',
  'layers/ConvolutionalLayer.cpp',
  'datasets/IDXFile.cpp',
  'datasets/MNISTDataSet.cpp',
  'ActivationFunction.cpp'
]

eigen_dep = dependency('eigen3')

libkann_lib = static_library('libkann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [cereal_dep, eigen_dep],
  cpp_args            : ['-fvisibility=hidden']
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [cereal_dep, eigen_dep],
  link_with : libkann_lib,
)
