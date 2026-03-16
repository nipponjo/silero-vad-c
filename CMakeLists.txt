cmake_minimum_required(VERSION 3.20)

project(silero_vad_c C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}")
endforeach()

find_program(Python3_EXECUTABLE NAMES python3 python REQUIRED)

set(SILERO_VAD_SAFETENSORS "${CMAKE_CURRENT_SOURCE_DIR}/src/silero_vad/data/silero_vad_16k.safetensors")
set(SILERO_VAD_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SILERO_VAD_GENERATED_C "${SILERO_VAD_GENERATED_DIR}/silero_vad_weights.c")
set(SILERO_VAD_GENERATED_H "${SILERO_VAD_GENERATED_DIR}/silero_vad_weights.h")

set(SILERO_VAD_WEIGHT_SOURCE "safetensors" CACHE STRING "Weight source for embedded export: safetensors or jit")
set_property(CACHE SILERO_VAD_WEIGHT_SOURCE PROPERTY STRINGS safetensors jit)
option(SILERO_VAD_ENABLE_SSE "Enable SSE optimizations in silero_vad.c" OFF)
option(SILERO_VAD_ENABLE_AVX2 "Enable AVX2/FMA optimizations in silero_vad.c" OFF)
option(SILERO_VAD_ENABLE_NEON "Enable ARM NEON optimizations in silero_vad.c" OFF)
option(SILERO_VAD_FAST_MATH "Enable faster, less strict floating-point optimizations" OFF)
option(SILERO_VAD_ENABLE_LTO "Enable link-time optimization" OFF)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SILERO_VAD_SYSTEM_PROCESSOR)
set(SILERO_VAD_TARGET_X86 OFF)
set(SILERO_VAD_TARGET_ARM OFF)
if(SILERO_VAD_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|x86|i[3-6]86)$")
  set(SILERO_VAD_TARGET_X86 ON)
elseif(SILERO_VAD_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|arm|armv[0-9].*)$")
  set(SILERO_VAD_TARGET_ARM ON)
endif()

if(SILERO_VAD_ENABLE_SSE AND SILERO_VAD_ENABLE_AVX2)
  message(FATAL_ERROR "SILERO_VAD_ENABLE_SSE and SILERO_VAD_ENABLE_AVX2 cannot both be ON.")
endif()

if(SILERO_VAD_ENABLE_SSE AND NOT SILERO_VAD_TARGET_X86)
  message(FATAL_ERROR "SILERO_VAD_ENABLE_SSE is only supported on x86/x86_64 targets.")
endif()

if(SILERO_VAD_ENABLE_AVX2 AND NOT SILERO_VAD_TARGET_X86)
  message(FATAL_ERROR "SILERO_VAD_ENABLE_AVX2 is only supported on x86/x86_64 targets.")
endif()

if(SILERO_VAD_ENABLE_NEON AND NOT SILERO_VAD_TARGET_ARM)
  message(FATAL_ERROR "SILERO_VAD_ENABLE_NEON is only supported on ARM targets.")
endif()

if(SILERO_VAD_WEIGHT_SOURCE STREQUAL "jit")
  set(SILERO_VAD_EXPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/export_silero_vad_jit_weights.py")
  set(SILERO_VAD_EXPORT_ARGS --output-dir "${SILERO_VAD_GENERATED_DIR}")
  set(SILERO_VAD_EXPORT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/export_silero_vad_jit_weights.py" "${CMAKE_CURRENT_SOURCE_DIR}/export_silero_vad_weights.py")
  set(SILERO_VAD_EXPORT_COMMENT "Generating embedded Silero VAD weights from torch.hub JIT model")
elseif(SILERO_VAD_WEIGHT_SOURCE STREQUAL "safetensors")
  set(SILERO_VAD_EXPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/export_silero_vad_weights.py")
  set(SILERO_VAD_EXPORT_ARGS --input "${SILERO_VAD_SAFETENSORS}" --output-dir "${SILERO_VAD_GENERATED_DIR}")
  set(SILERO_VAD_EXPORT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/export_silero_vad_weights.py" "${SILERO_VAD_SAFETENSORS}")
  set(SILERO_VAD_EXPORT_COMMENT "Generating embedded Silero VAD weights from safetensors")
else()
  message(FATAL_ERROR "Unsupported SILERO_VAD_WEIGHT_SOURCE='${SILERO_VAD_WEIGHT_SOURCE}'. Use 'safetensors' or 'jit'.")
endif()

add_custom_command(
  OUTPUT "${SILERO_VAD_GENERATED_C}" "${SILERO_VAD_GENERATED_H}"
  COMMAND "${Python3_EXECUTABLE}" "${SILERO_VAD_EXPORT_SCRIPT}" ${SILERO_VAD_EXPORT_ARGS}
  DEPENDS ${SILERO_VAD_EXPORT_DEPENDS}
  COMMENT "${SILERO_VAD_EXPORT_COMMENT}"
  VERBATIM
)

add_library(silero_vad SHARED
  silero_vad.c
  "${SILERO_VAD_GENERATED_C}"
)

target_compile_definitions(silero_vad PRIVATE SILERO_VAD_BUILD_DLL)
if(SILERO_VAD_ENABLE_SSE)
  target_compile_definitions(silero_vad PRIVATE SILERO_VAD_ENABLE_SSE=1)
endif()
if(SILERO_VAD_ENABLE_AVX2)
  target_compile_definitions(silero_vad PRIVATE SILERO_VAD_ENABLE_AVX2=1)
endif()
if(SILERO_VAD_ENABLE_NEON)
  target_compile_definitions(silero_vad PRIVATE SILERO_VAD_ENABLE_NEON=1)
endif()
set_target_properties(silero_vad PROPERTIES PREFIX "")
target_include_directories(silero_vad
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE "${SILERO_VAD_GENERATED_DIR}"
)

if(MSVC)
  target_compile_options(silero_vad PRIVATE /W4 /bigobj /O2 /GL)
  target_link_options(silero_vad PRIVATE /LTCG)
  if(SILERO_VAD_FAST_MATH)
    target_compile_options(silero_vad PRIVATE /fp:fast)
  endif()
  if(SILERO_VAD_ENABLE_AVX2)
    target_compile_options(silero_vad PRIVATE /arch:AVX2)
  elseif(SILERO_VAD_ENABLE_SSE AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(silero_vad PRIVATE /arch:SSE2)
  endif()
else()
  target_compile_options(silero_vad PRIVATE -Wall -Wextra -pedantic)
  target_link_libraries(silero_vad PRIVATE m)
  if(SILERO_VAD_FAST_MATH)
    target_compile_options(silero_vad PRIVATE -ffast-math)
  endif()
  if(SILERO_VAD_ENABLE_LTO)
    target_compile_options(silero_vad PRIVATE -flto)
    target_link_options(silero_vad PRIVATE -flto)
  endif()
  if(SILERO_VAD_ENABLE_AVX2)
    target_compile_options(silero_vad PRIVATE -mavx2 -mfma)
  elseif(SILERO_VAD_ENABLE_SSE AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(silero_vad PRIVATE -msse)
  endif()
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    # Keep Apple Silicon builds generic across M-series chips.
    target_compile_options(silero_vad PRIVATE -arch arm64)
    target_link_options(silero_vad PRIVATE -arch arm64)
  endif()
endif()
