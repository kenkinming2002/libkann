project('libkann', 'cpp', version : '0.1.0', default_options : ['cpp_std=c++2a'])

add_global_arguments('-fconcepts', language : 'cpp')
add_global_arguments('-fcoroutines', language : 'cpp')
add_global_arguments('-Wno-deprecated-enum-enum-conversion', language : 'cpp')

subdir('src')
subdir('include')
subdir('examples')
