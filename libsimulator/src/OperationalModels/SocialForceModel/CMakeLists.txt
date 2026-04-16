## SocialForceModel sources
list(APPEND SIMULATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/SocialForceModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SocialForceModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SocialForceModelBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SocialForceModelBuilder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SocialForceModelData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SocialForceModelUpdate.hpp
)

set(SIMULATOR_SOURCES ${SIMULATOR_SOURCES} PARENT_SCOPE)