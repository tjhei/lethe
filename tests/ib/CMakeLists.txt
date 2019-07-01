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
ADD_EXECUTABLE(ib_06 ib_06.cc)
DEAL_II_SETUP_TARGET(ib_06)
TARGET_LINK_LIBRARIES(ib_06 lethe)
add_test( "IB_06" ib_06)
ADD_EXECUTABLE(ib_07 ib_07.cc)
DEAL_II_SETUP_TARGET(ib_07)
TARGET_LINK_LIBRARIES(ib_07 lethe)
add_test( "IB_07" ib_07)

ADD_EXECUTABLE(ib_08 ib_08.cc)
DEAL_II_SETUP_TARGET(ib_08)
TARGET_LINK_LIBRARIES(ib_08 lethe)
add_test( "IB_08" ib_08)
ADD_EXECUTABLE(ib_09 ib_09.cc)
DEAL_II_SETUP_TARGET(ib_09)
TARGET_LINK_LIBRARIES(ib_09 lethe)
add_test( "IB_09" ib_09)

ADD_EXECUTABLE(ib_B_01 ib_B_01.cc)
DEAL_II_SETUP_TARGET(ib_B_01)
TARGET_LINK_LIBRARIES(ib_B_01 lethe)
add_test( "IB_B_01" ib_B_01)

ADD_EXECUTABLE(ib_B_02 ib_B_02.cc)
DEAL_II_SETUP_TARGET(ib_B_02)
TARGET_LINK_LIBRARIES(ib_B_02 lethe)
add_test( "IB_B_02" ib_B_02)

ADD_EXECUTABLE(ib_B_03 ib_B_03.cc)
DEAL_II_SETUP_TARGET(ib_B_03)
TARGET_LINK_LIBRARIES(ib_B_03 lethe)
add_test( "IB_B_03" ib_B_03)

