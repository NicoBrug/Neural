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


add_library(Neural STATIC
    src/network.cpp
    src/fc_layer.cpp
    src/layer.cpp
    src/activation_layer.cpp
    src/kernel.cu
    src/core.cpp

    src/loader/mnist.cpp

    includes/activation.h
    includes/activation_layer.h
    includes/fc_layer.h
    includes/layer.h
    includes/network.h
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