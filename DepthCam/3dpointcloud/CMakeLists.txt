cmake_minimum_required(VERSION 3.16)

project(3dpointcloud LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

# Find RealSense2 package
find_package(realsense2 REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

add_executable(3dpointcloud
  main.cpp
)
target_link_libraries(3dpointcloud Qt${QT_VERSION_MAJOR}::Core realsense2::realsense2 ${OpenCV_LIBS})

include(GNUInstallDirs)
install(TARGETS 3dpointcloud
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
