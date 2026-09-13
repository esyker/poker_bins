cmake_minimum_required(VERSION 3.16)
project(poker_bins LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(faiss CONFIG REQUIRED)
set(hand_isomorphism_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/hand-isomorphism)
add_library(hand_isomorphism STATIC
    ${hand_isomorphism_SOURCE_DIR}/src/hand_index.c
    ${hand_isomorphism_SOURCE_DIR}/src/deck.c)
target_include_directories(hand_isomorphism PUBLIC ${hand_isomorphism_SOURCE_DIR}/src)
set_target_properties(hand_isomorphism PROPERTIES C_STANDARD 99)

set(ompeval_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/OMPEval)
add_library(ompeval STATIC ${ompeval_SOURCE_DIR}/omp/HandEvaluator.cpp)
target_include_directories(ompeval PUBLIC ${ompeval_SOURCE_DIR})

add_executable(build_offline src/bucketing/build_offline.cpp src/cli/build_offline.cpp)
target_include_directories(build_offline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(build_offline PRIVATE
    hand_isomorphism ompeval faiss OpenMP::OpenMP_CXX)
if(MSVC)
    target_compile_options(build_offline PRIVATE /O2 /DNDEBUG)
else()
    target_compile_options(build_offline PRIVATE -O3 -DNDEBUG -ffast-math -funroll-loops)
endif()
