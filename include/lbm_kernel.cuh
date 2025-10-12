#ifndef LBM_KERNEL_CUH
#define LBM_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

typedef struct {
    // Grid dimensions
    int Nx, Ny;

    // Physics parameters
    float tau;
    float nu;
    float u_lid;

    // Device pointers
    float* f_d;
    float* f_new_d;
    float* rho_d;
    float* ux_d;
    float* uy_d;

    // CUDA grid configuration
    dim3 blocks;
    dim3 threads;

    // Initialization flag
    bool initialized;

} LBM_State;

// Initialize CUDA-OpenGL interop texture
void initCudaTexture(cudaGraphicsResource** cudaResource, unsigned int glTexture);

// Fill grid with time-varying colors (for testing)
void fillGridWithColors(cudaGraphicsResource* cudaResource, int width, int height, float time);

// Cleanup CUDA resources
void cleanupCudaResources(cudaGraphicsResource* cudaResource);

void initializeLBM(LBM_State* state, int Nx, int Ny, float tau, float Re);

void stepLBM(cudaGraphicsResource* cudaResource, LBM_State* state);

void lbm_destroy(LBM_State* state);

#endif // LBM_KERNEL_CUH
