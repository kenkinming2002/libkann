visualization_autoencoder_sources = [
  'main.cpp',
]

threads_dep = dependency('threads')

visualization_autoencoder = executable('autoencoder', visualization_autoencoder_sources,
  dependencies : [libkann_dep, sfml_dep, threads_dep]
)
