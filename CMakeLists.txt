cmake_minimum_required(VERSION 3.20)
project(hexudon_bench LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost 1.81 REQUIRED COMPONENTS json)
find_package(Threads REQUIRED)

add_library(hexudon_core src/core.cpp)
target_include_directories(hexudon_core PUBLIC include)
target_link_libraries(hexudon_core PUBLIC Boost::json Threads::Threads)
target_compile_options(hexudon_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hexudon src/main.cpp)
target_link_libraries(hexudon PRIVATE hexudon_core)

enable_testing()
add_executable(hexudon_core_tests tests/core_tests.cpp)
target_link_libraries(hexudon_core_tests PRIVATE hexudon_core)
add_test(NAME hexudon_core_tests COMMAND hexudon_core_tests)
