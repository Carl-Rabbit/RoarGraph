set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
find_package(MKL REQUIRED HINTS ${MKL_ROOT})

add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${MKL_INCLUDE})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${MKL_LIBRARIES})  # dynamic link
target_link_libraries(${PROJECT_NAME} PUBLIC MKL::MKL)      # static link
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# debug
# target_link_libraries(${PROJECT_NAME} PUBLIC asan ${MKL_LIBRARIES})  # dynamic link
# # target_link_libraries(${PROJECT_NAME} PUBLIC asan MKL::MKL)      # static link
# set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)

# add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

#install()