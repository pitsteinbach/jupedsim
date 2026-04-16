## CollisionFreeSpeedModelV2 sources
list(APPEND SIMULATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelV2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelV2.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelV2Builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelV2Builder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelV2Data.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelV2Update.hpp
)

set(SIMULATOR_SOURCES ${SIMULATOR_SOURCES} PARENT_SCOPE)