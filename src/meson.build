libkann_sources = [
  'Initialize.cpp',

  'Tensor.cpp',

  # Operations
  'operations/Add.cpp',
  'operations/CrossCorrelation.cpp',
  'operations/Scale.cpp',
  'operations/Subtract.cpp',
  'operations/TensorProduct.cpp',

  # Executors
  'Graph.cpp',
  'executors/DefaultExecutor.cpp',

  # Layers
  'Layer.cpp',
  'LayerDef.cpp',
  'layer_defs/Activation.cpp',
  'layer_defs/Convolutional.cpp',
  'layer_defs/Dense.cpp',
  'layer_defs/Sequential.cpp',

  # Datasets
  'datasets/MNIST.cpp',
  'datasets/Random.cpp',
  'datasets/write.cpp',

  # Optimizers
  'optimizers/SimpleOptimizer.cpp',
  'optimizers/AdamOptimizer.cpp',

  # Actual Machine Learning Training Algorithm
  'algorithms/ProgressBar.cpp',
  'algorithms/Context.cpp',
  'algorithms/Predict.cpp',
  'algorithms/Optimize.cpp',
]

fmt_dep = dependency('fmt')
range_v3_dep = dependency('range-v3')

eigen_dep = dependency('eigen3')
threads_dep = dependency('threads')
yaml_cpp_dep = dependency('yaml-cpp')

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

libkann_lib = library('kann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [fmt_dep, range_v3_dep, eigen_dep, sfml_dep, threads_dep, yaml_cpp_dep],
  gnu_symbol_visibility : 'hidden'
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [eigen_dep, sfml_dep],
  link_with : libkann_lib,
)
