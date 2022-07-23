libkann_sources = [
  'Initialize.cpp',

  # Layers
  'Layer.cpp',
  'LayerDef.cpp',
  'layers/Activation.cpp',
  'layers/Convolutional.cpp',
  'layers/Dense.cpp',
  'layers/Sequential.cpp',
  'layers/SoftMax.cpp',

  # Datasets
  'datasets/MNIST.cpp',
  'datasets/Random.cpp',

  # Optimizers
  #'Optimizer.cpp',
  'optimizers/SimpleOptimizer.cpp',
  'optimizers/AdamOptimizer.cpp',

  # Loss functions
  'loss_functions/Lp.cpp',
  'loss_functions/CrossEntropy.cpp',

  'ProgressBar.cpp',
]

libkann_lib = library('kann', libkann_sources,
  include_directories : ['../include'],
  cpp_pch             : ['pch/pch.hpp'],
  dependencies        : [libtensor_dep, fmt_dep, range_v3_dep, yaml_cpp_dep],
  gnu_symbol_visibility : 'hidden'
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [libtensor_dep],
  link_with : libkann_lib,
)
