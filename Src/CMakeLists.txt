include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable(pca
  PCA.h
  PCA.cu

  OPTIONS --cudart=shared  -arch sm_50 -use_fast_math
  DEBUG -Xptxas -v -g -G 
)
target_link_libraries( 
  pca
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CURAND_LIBRARY}
)

add_executable(pca_cpu
  PCA_cpu.h
  PCA_cpu.cpp
)
