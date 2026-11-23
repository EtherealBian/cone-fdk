cmake_minimum_required(VERSION 3.18)
project(cone_fdk LANGUAGES CXX)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "This release targets Windows x64")
endif()
add_executable(cone-fdk src/main.cpp)
target_link_libraries(cone-fdk PRIVATE shell32)
target_compile_features(cone-fdk PRIVATE cxx_std_14)
if(MINGW)
  target_link_options(cone-fdk PRIVATE -static -Wl,--no-insert-timestamp)
elseif(MSVC)
  target_link_options(cone-fdk PRIVATE /Brepro)
endif()
# CUDA compilation is independent of the host compiler: python scripts/ptx.py
# emits the driver-JIT PTX loaded by the --ptx argument at runtime.

