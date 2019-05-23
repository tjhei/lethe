# Usually, you will not need to modify anything beyond this point...
#Method of manufactured solution 2D

# Set the name of the project and target:
SET(TARGET "ib_01")

INCLUDE_DIRECTORIES(
  lethe
  ${CMAKE_SOURCE_DIR}/src/
  )
ADD_EXECUTABLE(ib_01 ib_01.cc)
DEAL_II_SETUP_TARGET(ib_01)
TARGET_LINK_LIBRARIES(ib_01 lethe)
add_test( "IB" ib_01)

