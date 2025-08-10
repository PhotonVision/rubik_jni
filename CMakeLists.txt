cmake_minimum_required(VERSION 3.10)

project(rubik_jni)

set(CMAKE_CXX_STANDARD 17) # Changed from 14 to 17 for better C++ support
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
endif()

set(BUILD_SHARED_LIBS ON)

find_package(JNI)
if(JNI_FOUND)
    # Fixes odd AWT dependency
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
# also valid: windowsx86-64

# type can be "", "debug", "static", or "staticdebug"
set(OPENCV_TYPE "")

# Download opencv, and save the path
include(FetchContent)
fetchcontent_declare(
    opencv_lib
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
)
fetchcontent_makeavailable(opencv_lib)

# download OpenCV headers
fetchcontent_declare(
    opencv_header
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
)
fetchcontent_makeavailable(opencv_header)

# Add FlatBuffers with proper configuration
include(FetchContent)

fetchcontent_declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.25
)

# Configure FlatBuffers options before making available
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Build FlatBuffers tests" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "Install FlatBuffers" FORCE)
set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "Build FlatBuffers compiler" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH
    OFF
    CACHE BOOL
    "Build FlatBuffers hash tool"
    FORCE
)

fetchcontent_makeavailable(flatbuffers)

# This probably doesn't work great with shared libraries, but I don't care about those right now
file(
    GLOB_RECURSE OPENCV_LIB_PATH
    "${opencv_lib_SOURCE_DIR}/**/*.lib"
    "${opencv_lib_SOURCE_DIR}/**/*.so*"
)
set(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on opencv ${OPENCV_LIB_PATH}")

# Download TensorFlow Lite headers
include(FetchContent)
fetchcontent_declare(
    tflite_header
    URL
        https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.19.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
)
fetchcontent_makeavailable(tflite_header)

# Set TensorFlow Lite headers
set(TFLITE_INCLUDE_DIR
    "${tflite_header_SOURCE_DIR}"
    CACHE PATH
    "Path to TensorFlow Lite include directory"
)

# Set TensorFlow Lite library - use absolute path and verify it exists
set(LIB_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/lib/"
    CACHE FILEPATH
    "Path to TensorFlow Lite library"
)

# Check if the library path exists
if(NOT EXISTS ${LIB_PATH})
    message(FATAL_ERROR "Library directory not found at: ${LIB_PATH}")
else()
    message(STATUS "Found library directory: ${LIB_PATH}")
endif()

set(CMAKE_INSTALL_RPATH "lib")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} SHARED src/main/native/cpp/rubik_jni.cpp)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        Threads::Threads
        ${OPENCV_LIB_PATH}
        ${LIB_PATH}/libtensorflowlite.so # C++ API library
        ${LIB_PATH}/libexternal_delegate.so # External delegate
        flatbuffers # FlatBuffers library
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
        ${JNI_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_PATH}
        ${TFLITE_INCLUDE_DIR}
        ${flatbuffers_SOURCE_DIR}/include # Add FlatBuffers include directory
)

# Add some compiler flags for better error reporting
target_compile_options(
    ${PROJECT_NAME}
    PRIVATE -Wall -Wextra -Wno-unused-parameter
)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\\\$ORIGIN/")

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
install(
    FILES
        ${LIB_PATH}/libtensorflowlite.so # Only C++ library needed
        ${LIB_PATH}/libexternal_delegate.so
    TYPE LIB
)
