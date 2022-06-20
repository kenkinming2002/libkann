libkann_sources = [
  #'Initialize.cpp',

  'Math.cpp',
  'Utils.cpp',

  # Operations
  #'operations/Add.cpp',
  #'operations/Broadcast.cpp',
  #'operations/CrossCorrelation.cpp',
  #'operations/Scale.cpp',
  #'operations/Subtract.cpp',
  #'operations/TensorProduct.cpp',

  # Executors
  #'Graph.cpp',
  #'executors/DefaultExecutor.cpp',

  # Layers
  'LayerStorage.cpp',
  'LayerDef.cpp',
  #'layer_defs/Activation.cpp',
  #'layer_defs/Convolutional.cpp',
  #'layer_defs/Dense.cpp',
  #'layer_defs/Sequential.cpp',

  # Datasets
  'datasets/MNIST.cpp',
  'datasets/Random.cpp',
  'datasets/write.cpp',

  # Optimizers
  #'Optimizer.cpp',
  #'optimizers/SimpleOptimizer.cpp',
  #'optimizers/AdamOptimizer.cpp',

  # Actual Machine Learning Training Algorithm
  'algorithms/ProgressBar.cpp',
  #'algorithms/Context.cpp',
  #'algorithms/Batch.cpp',
  #'algorithms/Predict.cpp',
  #'algorithms/Optimize.cpp',
]

fmt_dep = dependency('fmt')
range_v3_dep = dependency('range-v3')
eigen_dep = dependency('eigen3')
yaml_cpp_dep = dependency('yaml-cpp')

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

libkann_lib = library('kann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [fmt_dep, range_v3_dep, eigen_dep, sfml_dep, yaml_cpp_dep],
  gnu_symbol_visibility : 'hidden'
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [sfml_dep],
  link_with : libkann_lib,
)
