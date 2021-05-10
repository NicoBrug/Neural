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

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_PREFIX_PATH "/opt/Qt/5.15.2/gcc_64")
set(Qt5Core_DIR "/opt/Qt/5.15.2/gcc_64/lib/cmake/Qt5Core")
set(Qt5_DIR "/opt/Qt/5.15.2/gcc_64/lib/cmake/Qt5")
set(QT_QMAKE_EXECUTABLE "/opt/Qt/5.15.2/bin/qmake")
set(LD_LIBRARY_PATH "/opt/Qt/5.15.2/gcc_64/lib/")

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets 3DCore 3DExtras 3DRender 3DInput Charts)
#set_property(TARGET Qt5::Core APPEND PROPERTY INTERFACE_COMPILE_OPTIONS -fPIC)

set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")

# set up a mapping so that the Release configuration for the Qt imported target is
# used in the COVERAGE CMake configuration.
#set_target_properties(Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")

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
    neural/src/layers/fc_layer.cpp
    neural/src/layers/layer.cpp
    neural/src/layers/activation_layer.cpp
    neural/src/layers/conv_layer.cpp
    neural/src/layers/flatten_layer.cpp

    neural/src/network.cpp
    neural/src/kernel.cu
    neural/src/core.cpp
    neural/src/loader/mnist.cpp

    neural/src/plotter/plotter.cpp

    #Layers INCLUDE
    neural/includes/layers/activation_layer.h
    neural/includes/layers/fc_layer.h
    neural/includes/layers/layer.h
    neural/includes/layers/conv_layer.h
    neural/includes/layers/flatten_layer.h

    #network
    neural/includes/network.h
    neural/includes/activation.h
    neural/includes/kernel.h
    neural/includes/core.h
    neural/includes/loader/mnist.h

    neural/includes/plotter/plotter.h

  )


if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
	set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()


#https://gist.github.com/erikzenker/713c4ff76949058d5d5d

target_link_libraries(Neural PUBLIC ${JSONCPP_LIBRARIES})
target_link_libraries(Neural PUBLIC sciplot::sciplot)
target_link_libraries(Neural PUBLIC
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::3DCore
    Qt5::3DExtras
    Qt5::3DRender
    Qt5::3DInput
    Qt5::Charts
    )

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Neural PUBLIC OpenMP::OpenMP_CXX)
endif()


target_compile_features(Neural PUBLIC cxx_std_11)

set_target_properties( Neural
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)
  add_executable(test_neural ${CMAKE_CURRENT_SOURCE_DIR}/neural/main.cpp)
  set_target_properties(test_neural  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_neural  PRIVATE Neural)
  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET test_neural  PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()