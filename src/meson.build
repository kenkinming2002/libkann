
libkann_sources = [
  'layers/SequentialLayer.cpp',
  'layers/RecurrentLayer.cpp',
  'layers/IdentityLayer.cpp',
  'layers/WeightLayer.cpp',
  'layers/ActivationLayer.cpp',
  'layers/ConvolutionalLayer.cpp',
  'layers/DeconvolutionalLayer.cpp',
  'datasets/IDXFile.cpp',
  'datasets/MNISTDataSet.cpp',
  'datasets/RandomDataSet.cpp',
  'datasets/write.cpp',
  'Layer.cpp',
  'Model.cpp',
  'Loader.cpp',
  'Graph.cpp',
  'Algorithm.cpp',
  'ActivationFunction.cpp',
  'Executor.cpp',
  'executors/DefaultExecutor.cpp',
  'executors/ThreadedExecutor.cpp',
  'optimizers/SimpleOptimizer.cpp',
  'optimizers/AdamOptimizer.cpp',
  'Differentiate.cpp',
  'operations/IdentityOperation.cpp',
  'operations/ReduceOperation.cpp',
  'operations/MatrixMultiplyOperation.cpp',
  'operations/CWiseProductOperation.cpp',
  'operations/MultiplyOperation.cpp',
  'operations/SubtractOperation.cpp',
  'operations/ConvolutionOperation.cpp',
  'operations/DeconvolutionOperation.cpp',
]

eigen_dep = dependency('eigen3')
threads_dep = dependency('threads')
yaml_cpp_dep = dependency('yaml-cpp')

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

libkann_lib = library('kann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [cereal_dep, eigen_dep, sfml_dep, threads_dep, yaml_cpp_dep]
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [cereal_dep, eigen_dep, sfml_dep],
  link_with : libkann_lib,
)
