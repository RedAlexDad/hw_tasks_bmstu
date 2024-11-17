cmake_minimum_required(VERSION 3.12)
project(TypePolynomialModel)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Найдем библиотеку Armadillo
find_package(Armadillo REQUIRED)

# Найдем библиотеку GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Каталог с исходниками
include_directories(src)

# Список всех исполняемых файлов
set(SOURCES
    PolynomialMemoryModel
    SparseDelayMemoryPolynomialModel
    OrthogonalMemoryPolynomialModel
    NonUniformMemoryPolynomialModel
    HybridMemoryPolynomialModel
    GeneralizedMemoryPolynomialModel
    ExponentiallyShapedMemoryDelayProfileModel
    EnvelopeMemoryPolynomialModel
    DDRVolterraModel
)

# Цикл для добавления и линковки исполняемых файлов
foreach(exec ${SOURCES})
    add_executable(${exec} src/${exec}.cpp)
    target_link_libraries(${exec} armadillo)
endforeach()