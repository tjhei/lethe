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
add_test( "IB_01" ib_01)
ADD_EXECUTABLE(ib_02 ib_02.cc)
DEAL_II_SETUP_TARGET(ib_02)
TARGET_LINK_LIBRARIES(ib_02 lethe)
add_test( "IB_02" ib_02)
ADD_EXECUTABLE(ib_03 ib_03.cc)
DEAL_II_SETUP_TARGET(ib_03)
TARGET_LINK_LIBRARIES(ib_03 lethe)
add_test( "IB_03" ib_03)
ADD_EXECUTABLE(ib_04 ib_04.cc)
DEAL_II_SETUP_TARGET(ib_04)
TARGET_LINK_LIBRARIES(ib_04 lethe)
add_test( "IB_04" ib_04)
ADD_EXECUTABLE(ib_05 ib_05.cc)
DEAL_II_SETUP_TARGET(ib_05)
TARGET_LINK_LIBRARIES(ib_05 lethe)
add_test( "IB_05" ib_05)
