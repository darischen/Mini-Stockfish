# CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(nnue_inference LANGUAGES CXX)

# ─── Force C++17 ───
set(CMAKE_CXX_STANDARD      17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS    OFF)

# ─── Find the CPU‑only LibTorch ───
find_package(Torch REQUIRED
             PATHS "C:/Users/daris/anaconda3/envs/eews/lib/site-packages/torch/share/cmake"
             NO_DEFAULT_PATH)

# ─── Build the shared library ───
add_library(nnue_inference SHARED
    nnue_inference.cpp
)

target_compile_definitions(nnue_inference PRIVATE NNUE_INFERENCE_EXPORTS)

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

# ─── Chess move generation library (no Torch dependency) ───
add_library(chess_movegen SHARED
    chess_movegen.cpp
)
target_compile_definitions(chess_movegen PRIVATE CHESS_MOVEGEN_EXPORTS)
if(MSVC)
    target_compile_options(chess_movegen PRIVATE /O2)
else()
    target_compile_options(chess_movegen PRIVATE -O3)
endif()
set_target_properties(chess_movegen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    CXX_STANDARD 17
)
