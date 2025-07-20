cmake_minimum_required(VERSION 3.10)
project(rubik_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(BUILD_SHARED_LIBS ON)

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

# Cross-compilation setup
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling for ${CMAKE_SYSTEM_PROCESSOR}")

    # Set the target architecture for OpenCV download
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(OPENCV_ARCH "linuxarm64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
        set(OPENCV_ARCH "linuxarm32")
    else()
        set(OPENCV_ARCH "linuxx86-64")
    endif()
else()
    # Native compilation - detect architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(OPENCV_ARCH "linuxarm64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
        set(OPENCV_ARCH "linuxarm32")
    else()
        set(OPENCV_ARCH "linuxx86-64")
    endif()
endif()

message(STATUS "Using OpenCV architecture: ${OPENCV_ARCH}")

# Find JNI - need to handle cross-compilation
find_package(JNI REQUIRED)
if(JNI_FOUND)
    # Fixes odd AWT dependency
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

# Skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined"
)

# OpenCV configuration
set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
set(OPENCV_TYPE "")

# Download opencv with correct architecture
include(FetchContent)
fetchcontent_declare(
    opencv_lib
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
)
fetchcontent_makeavailable(opencv_lib)

# Download OpenCV headers (architecture independent)
fetchcontent_declare(
    opencv_header
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
)
fetchcontent_makeavailable(opencv_header)

# TensorFlow - Note: This may not work well for cross-compilation
# You might need to provide pre-built TensorFlow libraries for ARM
if(NOT CMAKE_CROSSCOMPILING OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    fetchcontent_declare(
        tensorflow
        GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
        GIT_TAG v2.15.0
    )
    fetchcontent_makeavailable(tensorflow)

    set(TENSORFLOW_INCLUDE_PATH "${tensorflow_SOURCE_DIR}")
    file(GLOB_RECURSE TENSORFLOW_LIB_PATH)
    message("Using TensorFlow include path: ${TENSORFLOW_INCLUDE_PATH}")
    message("Found TensorFlow libraries: ${TENSORFLOW_LIB_PATH}")
else()
    # For cross-compilation, you'll need to provide pre-built TensorFlow libraries
    # or disable TensorFlow support
    message(
        WARNING
        "TensorFlow cross-compilation not fully supported. You may need to provide pre-built libraries."
    )
    set(TENSORFLOW_INCLUDE_PATH "")
    set(TENSORFLOW_LIB_PATH "")
endif()

# Find OpenCV libraries with correct architecture
file(
    GLOB_RECURSE OPENCV_LIB_PATH
    "${opencv_lib_SOURCE_DIR}/**/*.lib"
    "${opencv_lib_SOURCE_DIR}/**/*.so*"
)

set(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on opencv ${OPENCV_LIB_PATH}")

# Verify libraries are for correct architecture
if(CMAKE_CROSSCOMPILING)
    foreach(lib ${OPENCV_LIB_PATH})
        execute_process(
            COMMAND file ${lib}
            OUTPUT_VARIABLE FILE_INFO
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "Library info: ${lib} - ${FILE_INFO}")
    endforeach()
endif()

set(CMAKE_INSTALL_RPATH "lib")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Create the main library
add_library(${PROJECT_NAME} src/main/native/cpp/rubik_jni.cpp)

# Link libraries
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC Threads::Threads ${OPENCV_LIB_PATH} ${JNI_LIBRARIES}
)

# Add TensorFlow libraries if available
if(TENSORFLOW_LIB_PATH)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TENSORFLOW_LIB_PATH})
endif()

# Include directories
target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
        ${JNI_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_PATH}
)

# Add TensorFlow include if available
if(TENSORFLOW_INCLUDE_PATH)
    target_include_directories(
        ${PROJECT_NAME}
        PRIVATE ${TENSORFLOW_INCLUDE_PATH}
    )
endif()

# Set target properties
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\\\$ORIGIN/")

# Cross-compilation specific compiler flags
if(CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(${PROJECT_NAME} PRIVATE -march=armv8-a)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
        target_compile_options(
            ${PROJECT_NAME}
            PRIVATE -march=armv7-a -mfpu=neon
        )
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
