PROJECT(ftkMixtureSegmentation)

cmake_minimum_required(VERSION 2.0)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET(MIXTURE_SEGMENTATION_SRCS
seg_helpers.cpp
)
SET(MIXTURE_SEGMENTATION_HDRS
seg_helpers.h
)

SET(EIGEN_INCLUDE_DIRS "/data/amine/libs/eig3-src/")
include_directories ( ${EIGEN_INCLUDE_DIRS} )



ADD_LIBRARY(ftkMixtureSegmentation ${MIXTURE_SEGMENTATION_SRCS} ${MIXTURE_SEGMENTATION_HDRS})

TARGET_LINK_LIBRARIES(ftkMixtureSegmentation ${ITK_USE_LIBRARIES} )

# kmeans clustering segmenation 
ADD_EXECUTABLE(mixture_segment main.cpp )
TARGET_LINK_LIBRARIES(mixture_segment ftkMixtureSegmentation  ${ITK_LIBRARIES} )
INSTALL(TARGETS mixture_segment RUNTIME DESTINATION ${INSTALL_BIN_DIR})

# removing small components
ADD_EXECUTABLE(clean_mixture_segmentation remove_small_components.cpp )
TARGET_LINK_LIBRARIES(clean_mixture_segmentation ftkMixtureSegmentation  ${ITK_LIBRARIES} )
INSTALL(TARGETS clean_mixture_segmentation RUNTIME DESTINATION ${INSTALL_BIN_DIR})

# find seeds
ADD_EXECUTABLE(detect_seeds detect_seeds.cpp )
TARGET_LINK_LIBRARIES(detect_seeds ftkMixtureSegmentation  ${ITK_LIBRARIES} )
INSTALL(TARGETS detect_seeds RUNTIME DESTINATION ${INSTALL_BIN_DIR})


# find number of cells and segment using spectral clustering
ADD_EXECUTABLE(spectral_segment spectral_segment.cpp )
TARGET_LINK_LIBRARIES(spectral_segment  ftkMixtureSegmentation  ${ITK_LIBRARIES} )
INSTALL(TARGETS spectral_segment RUNTIME DESTINATION ${INSTALL_BIN_DIR})



