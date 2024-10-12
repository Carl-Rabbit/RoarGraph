set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
find_package(MKL REQUIRED HINTS ${MKL_ROOT})

add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE MKL::MKL)
target_link_libraries(${PROJECT_NAME} PUBLIC ${MKL_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} PUBLIC asan MKL::MKL)
# target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
# set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
# add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

#install()