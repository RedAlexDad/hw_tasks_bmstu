cmake_minimum_required(VERSION 3.26)
project(CTDNN LANGUAGES CXX)

# C++ стандарт
set(CPP_VERSION 20)
set(CMAKE_CXX_STANDARD ${CPP_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Установка компиляторов
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# Включение подробного вывода
set(CMAKE_VERBOSE_MAKEFILE ON)

# Переменная для выбора устройства
set(DEVICE "gpu" CACHE STRING "Select device: gpu, cpu, any")

# Добавим флаги для указания конкретного устройства
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda")
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -O2")

# # Если необходимо, укажите отдельные переменные
set(SYCL_INCLUDE_DIR "/opt/intel/oneapi/compiler/2025.0/include")
set(SYCL_LIBRARY_DIR "/opt/intel/oneapi/compiler/2025.0/lib")

# Добавление пути к заголовкам и библиотекам SYCL
include_directories(${SYCL_INCLUDE_DIR})
link_directories(${SYCL_LIBRARY_DIR})

# Определение устройства через флаги
if(DEVICE STREQUAL "gpu")
    add_compile_definitions(ONEAPI_DEVICE_SELECTOR="cuda:*")
elseif(DEVICE STREQUAL "cpu")
    add_compile_definitions(ONEAPI_DEVICE_SELECTOR="cpu")
elseif(DEVICE STREQUAL "any")
    add_compile_definitions(ONEAPI_DEVICE_SELECTOR="*")
else()
    message(FATAL_ERROR "Invalid device specified. Use '-DEVICE=gpu', '-DEVICE=cpu', or '-DEVICE=any'.")
endif()

# IntelSYCL
find_package(IntelSYCL REQUIRED)

# Добавление исполняемых файлов
add_executable(main CTDNN/main.cpp)

# Вывод выбранного устройства во время сборки
message(STATUS "Building for device: ${DEVICE}")