cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/home/rmslick/ObjectTracker/mask_rcnn")

find_package(realsense2 REQUIRED)
find_package(PCL 1.11 REQUIRED)
find_package(pybind11 REQUIRED)
find_package( OpenCV REQUIRED )


#add_executable(ObjectTracker main.cpp)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(realsense2 REQUIRED)
find_package(PCL 1.11 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED )


add_executable(ObjectTracker main.cpp ${SOURCES})


set_target_properties(ObjectTracker PROPERTIES LIBRARY_OUTPUT_DIRECTORY "/home/rmslick/ObjectTracker/Mask_RCNN/samples/trussjoint")


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(ObjectTracker realsense2 ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS} )