cmake_minimum_required(VERSION 3.28)

set(PROJECT_NAME "EasyEinsum")
set(PROJECT_VERSION "0.1.0")

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Options ---
option(BUILD_EXE "Build example executables" OFF)
option(BUILD_PYTHON_LIB "Build Python library for EasyEinsum" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)

# --- 1.  Header-only INTERFACE) ---
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE 
    ${PROJECT_SOURCE_DIR}/src 
    ${PROJECT_SOURCE_DIR}/external/tblis-1.3.0/include/
)

# --- 2. dependencies ---
# OpenMP
find_package(OpenMP REQUIRED)
# TBlis 
set(TBLIS_LIBS ${PROJECT_SOURCE_DIR}/external/tblis-1.3.0/lib/libtblis.so)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEFAULT_COMPILE_FLAGS -O0 -g -march=native -fno-omit-frame-pointer)
    target_compile_definitions(${PROJECT_NAME} INTERFACE _DEBUG)
else()
    set(DEFAULT_COMPILE_FLAGS -O3 -march=native -funroll-loops)
endif()

# --- 3. CUDA ---
if(ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 89) # 4090D
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_compile_definitions(${PROJECT_NAME} INTERFACE _USE_CUDA_)

    find_package(CUDAToolkit REQUIRED)
    
    # MatX
    set(MATX_EN_CUTENSOR ON CACHE BOOL "Enable cuTENSOR support in MatX" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/MatX-1.0.0)
    # # cuTENSOR
    # find_path(CUTENSOR_INCLUDE_DIR
    #     NAMES cutensor.h
    #     HINTS $ENV{CUTENSOR_ROOT}/include
    #     NO_DEFAULT_PATH
    # )

    # find_library(CUTENSOR_LIB
    #     NAMES cutensor
    #     HINTS $ENV{CUTENSOR_ROOT}/lib/12
    #           $ENV{CUTENSOR_ROOT}/lib/11
    #           $ENV{CUTENSOR_ROOT}/lib
    #     NO_DEFAULT_PATH
    # )

    # if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIB)
    #     message(FATAL_ERROR "cuTENSOR not found in CUTENSOR_ROOT")
    # endif()

    # include_directories(BEFORE ${CUTENSOR_INCLUDE_DIR})
    # include_directories(${CUDAToolkit_INCLUDE_DIRS})

    # # CUDA dependencies
    # set(CUDA_DEPS 
    #     matx::matx 
    #     CUDA::cudart 
    #     CUDA::cublas 
    #     ${CUTENSOR_LIB}
    # )
    # target_include_directories(${PROJECT_NAME} INTERFACE ${CUTENSOR_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} INTERFACE matx::matx)
endif()

# --- 4. executables ---
if(BUILD_EXE)
    # --- CPU version ---
    add_executable(${PROJECT_NAME}_cpu_exe src/main.cpp)
    target_link_libraries(${PROJECT_NAME}_cpu_exe PRIVATE 
        ${PROJECT_NAME} 
        ${TBLIS_LIBS} 
        OpenMP::OpenMP_CXX
    )
    target_compile_options(${PROJECT_NAME}_cpu_exe PRIVATE ${DEFAULT_COMPILE_FLAGS})

    # --- CUDA version ---
    if(ENABLE_CUDA)
        add_executable(${PROJECT_NAME}_cuda_exe src/main.cu) 
        target_link_libraries(${PROJECT_NAME}_cuda_exe PRIVATE 
            ${PROJECT_NAME} 
            ${TBLIS_LIBS} 
            OpenMP::OpenMP_CXX
        )
        target_compile_options(${PROJECT_NAME}_cuda_exe PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:${DEFAULT_COMPILE_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
endif()

# --- 5. Python bindings ---
if(BUILD_PYTHON_LIB)
    # --- CPU version ---
    find_package(pybind11 REQUIRED)
    pybind11_add_module(easyeinsum_cpu ${CMAKE_CURRENT_SOURCE_DIR}/src/python/cpu_bindings.cpp)
    target_link_libraries(easyeinsum_cpu PRIVATE 
        ${PROJECT_NAME}         
        ${TBLIS_LIBS}           
        OpenMP::OpenMP_CXX      
    )
    target_compile_options(easyeinsum_cpu PRIVATE ${DEFAULT_COMPILE_FLAGS})
    # --- CUDA version ---
    if(ENABLE_CUDA)
        pybind11_add_module(easyeinsum_cuda ${CMAKE_CURRENT_SOURCE_DIR}/src/python/cuda_bindings.cu)
        target_link_libraries(easyeinsum_cuda PRIVATE 
            ${PROJECT_NAME}         
            ${TBLIS_LIBS}           
            OpenMP::OpenMP_CXX      
        )
        target_compile_options(easyeinsum_cuda PRIVATE ${DEFAULT_COMPILE_FLAGS})
    endif()
endif()
