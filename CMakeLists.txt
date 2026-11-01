cmake_minimum_required(VERSION 3.21)
project(neurolens VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEUROLENS_BUILD_VIEWER "Build the Qt viewer" ON)

# core: pure C++, no Qt
add_subdirectory(core)

enable_testing()
add_subdirectory(tests)

if(NEUROLENS_BUILD_VIEWER)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    if(WIN32 AND NOT Qt6_DIR AND NOT CMAKE_PREFIX_PATH)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
           OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            set(_qt_prefix_candidates
                "C:/Qt/6.10.1/msvc2022_64"
                "C:/Qt/6.10.1/msvc2022_arm64"
            )
        else()
            set(_qt_prefix_candidates
                "C:/Qt/6.10.1/mingw_64"
            )
        endif()

        foreach(_qt_prefix IN LISTS _qt_prefix_candidates)
            if(EXISTS "${_qt_prefix}/lib/cmake/Qt6/Qt6Config.cmake")
                list(PREPEND CMAKE_PREFIX_PATH "${_qt_prefix}")
                break()
            endif()
        endforeach()
    endif()

    find_package(Qt6 REQUIRED COMPONENTS Widgets Quick Qml QuickWidgets)
    add_subdirectory(viewer)
endif()
