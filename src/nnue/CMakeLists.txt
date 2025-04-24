# CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(nnue_inference LANGUAGES CXX)

# ─── Force C++17 ───
set(CMAKE_CXX_STANDARD      17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS    OFF)

# ─── Find the CPU‑only LibTorch ───
find_package(Torch REQUIRED
             PATHS "C:/Users/daris/libtorch"
             NO_DEFAULT_PATH)

# ─── Build the shared library ───
add_library(nnue_inference SHARED
    nnue_inference.cpp
)

set_target_properties(nnue_inference
    PROPERTIES
      WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ─── Include & link Torch the old way ───
target_include_directories(nnue_inference PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(nnue_inference PRIVATE
    ${TORCH_LIBRARIES}
)

# ─── Drop the output DLL/.lib into the build dir ───
set_target_properties(nnue_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
