cmake_minimum_required(VERSION 3.21)
project(fasteq_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ✅ GPU 架构设置（必须在 add_library 之前定义）
# 优先使用 setup.py 传入的 GPU_ARCHS，否则自动检测
if(NOT DEFINED GPU_ARCHS)
    execute_process(COMMAND rocm_agent_enumerator OUTPUT_VARIABLE DETECTED_ARCH)
    if(DETECTED_ARCH)
        string(STRIP "${DETECTED_ARCH}" GPU_ARCHS)
    else()
        set(GPU_ARCHS "gfx90a" CACHE STRING "Target AMD GPU architectures")
    endif()
endif()
message(STATUS "Building for HIP arch: ${GPU_ARCHS}")

# 验证架构格式（确保用分号分隔）
string(REPLACE " " ";" GPU_ARCHS "${GPU_ARCHS}")

set(Caffe2_USE_HIPBLAS OFF)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)

# ✅ 源文件：使用 .cu，让 PyTorch 自动 hipify
set(CU_SRCS
  src/cwtp_fwd.hip
  src/cwtp_bwd.hip
  src/mptp_fwd.hip
  src/mptp_bwd.hip
)

# 验证源文件存在
foreach(_f IN LISTS CU_SRCS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_f}")
        message(FATAL_ERROR "Missing source file: ${_f}")
    endif()
endforeach()

# ✅ 创建模块库（让 PyTorch 处理 hipify）
add_library(_hip MODULE ${CU_SRCS})

# ✅ 设置 GPU 架构（必须在 add_library 之后）
if(DEFINED GPU_ARCHS)
    foreach(arch IN LISTS GPU_ARCHS)
        target_compile_options(_hip PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${arch}>)
    endforeach()
endif()

# ✅ 编译选项
target_compile_options(_hip PRIVATE
    -DWITH_HIP
    -D__HIP_PLATFORM_AMD__=1
    -O3
    -std=c++17
    -fPIC
)

# 包含路径
target_include_directories(_hip PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    $<$<BOOL:${ROCM_PATH}>:${ROCM_PATH}/include>
)

# ✅ 链接（自动使用 TORCH_LIBRARIES 中的 HIP 库）
target_link_libraries(_hip PRIVATE 
    ${TORCH_LIBRARIES}
    hip::device
    amdhip64
)

# ✅ 输出为 Python 可导入的 .so
set_target_properties(_hip PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")