set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compilation tools
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_AR aarch64-linux-gnu-ar)
set(CMAKE_STRIP aarch64-linux-gnu-strip)
set(CMAKE_RANLIB aarch64-linux-gnu-ranlib)

# Compiler flags
set(CMAKE_C_FLAGS_INIT "-march=armv8-a")
set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a")

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Set the sysroot if you have one
# set(CMAKE_SYSROOT /path/to/arm64/sysroot)

# Java/JNI cross-compilation settings
# You may need to point to ARM64 JDK
# set(JAVA_HOME /path/to/arm64/jdk)
# set(JNI_INCLUDE_DIRS ${JAVA_HOME}/include ${JAVA_HOME}/include/linux)
