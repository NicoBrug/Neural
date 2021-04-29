cmake_minimum_required(VERSION 3.18.6 FATAL_ERROR)

option( CORE_USE_CUDA    "Use CUDA to speed up certain parts of the code."              ON )

project(Neural
    VERSION 1.0
    DESCRIPTION "Neural is a library developed in C++ allowing the realization of artificial neural networks"
    LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)


find_package(PkgConfig REQUIRED)
find_package(sciplot)

pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

include_directories(lib/EigenRand-0.3.3)
link_directories(lib/EigenRand-0.3.3)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS "-mfpmath=sse")
set(CMAKE_CXX_FLAGS "-ftree-vectorize")
set(CMAKE_CXX_FLAGS "-O3")



add_library(Neural STATIC
    #Layers SOURCE
    src/layers/fc_layer.cpp
    src/layers/layer.cpp
    src/layers/activation_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/flatten_layer.cpp

    src/network.cpp
    src/kernel.cu
    src/core.cpp
    src/loader/mnist.cpp

    #Layers INCLUDE
    includes/layers/activation_layer.h
    includes/layers/fc_layer.h
    includes/layers/layer.h
    includes/layers/conv_layer.h
    includes/layers/flatten_layer.h

    includes/network.h
    includes/activation.h
    includes/kernel.h
    includes/core.h

    includes/loader/mnist.h

  )




#https://gist.github.com/erikzenker/713c4ff76949058d5d5d

target_link_libraries(Neural PUBLIC ${JSONCPP_LIBRARIES})
target_link_libraries(Neural PUBLIC sciplot::sciplot)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Neural PUBLIC OpenMP::OpenMP_CXX)
endif()



target_compile_features(Neural PUBLIC cxx_std_11)
#target_compile_options(Neural PUBLIC -Wall -Wextra -Wpedantic)

set_target_properties( Neural
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)
  add_executable(test_neural ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
  set_target_properties(test_neural  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_neural  PRIVATE Neural)
  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET test_neural  PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()