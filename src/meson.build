
libkann_sources = [
  'LayerVariable.cpp',
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
  'Model.cpp',
  'Build.cpp',
  'Algorithm.cpp',
  'ActivationFunction.cpp',
  'Executor.cpp',
  'executors/GraphExecutor.cpp',
  'executors/DefaultExecutor.cpp',
  'executors/ThreadedExecutor.cpp',
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
boost_dep = dependency('boost', modules : ['graph'])
threads_dep = dependency('threads')

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

libkann_lib = library('kann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [cereal_dep, eigen_dep, boost_dep, sfml_dep, threads_dep]
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [cereal_dep, eigen_dep, sfml_dep],
  link_with : libkann_lib,
)
