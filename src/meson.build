libkann_sources = [
  'Initialize.cpp',
  # Operations
  'Differentiate.cpp',
  'operations/IdentityOperation.cpp',
  'operations/ReduceOperation.cpp',
  'operations/MatrixMultiplyOperation.cpp',
  'operations/CWiseProductOperation.cpp',
  'operations/MultiplyOperation.cpp',
  'operations/SubtractOperation.cpp',
  'operations/ConvolutionOperation.cpp',
  'operations/DeconvolutionOperation.cpp',

  # Executors
  'Graph.cpp',
  'Executor.cpp',
  'executors/DefaultExecutor.cpp',
  'executors/ThreadedExecutor.cpp',

  # Layers
  'Layer.cpp',
  'LayerDef.cpp',
  'ActivationFunction.cpp',
  'layer_defs/Sequential.cpp',
  'layer_defs/Recurrent.cpp',
  'layer_defs/Identity.cpp',
  'layer_defs/Activation.cpp',
  'layer_defs/Weight.cpp',
  'layer_defs/Convolutional.cpp',
  'layer_defs/Deconvolutional.cpp',

  # Datasets
  'datasets/MNIST.cpp',
  'datasets/Random.cpp',
  'datasets/write.cpp',

  # Optimizers
  'optimizers/SimpleOptimizer.cpp',
  'optimizers/AdamOptimizer.cpp',

  # Actual Machine Learning Training Algorithm
  'algorithms/Helpers.cpp',
  'algorithms/Predict.cpp',
  'algorithms/Optimize.cpp',
]

range_v3_dep = dependency('range-v3')
eigen_dep = dependency('eigen3')
threads_dep = dependency('threads')
yaml_cpp_dep = dependency('yaml-cpp')

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

libkann_lib = library('kann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [range_v3_dep, eigen_dep, sfml_dep, threads_dep, yaml_cpp_dep]
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [eigen_dep, sfml_dep],
  link_with : libkann_lib,
)
