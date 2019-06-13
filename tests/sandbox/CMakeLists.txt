# Usually, you will not need to modify anything beyond this point...
#Method of manufactured solution 2D

# Set the name of the project and target:
SET(TARGET "sandbox")

INCLUDE_DIRECTORIES(
  lethe
  ${CMAKE_SOURCE_DIR}/src/
  )
ADD_EXECUTABLE(single_element single_element.cc)
DEAL_II_SETUP_TARGET(single_element)
TARGET_LINK_LIBRARIES(single_element lethe)
