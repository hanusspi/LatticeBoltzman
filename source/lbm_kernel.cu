#include <glad/glad.h>
#include "lbm_kernel.cuh"
#include <cmath>

__constant__ int e_x[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
__constant__ int e_y[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
__constant__ float w[9]   = { 4.0f/9.0f,
                             1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
							 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };



__global__ void init_kernel(float* f, int Nx, int Ny, float u_lid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    int idx = y * Nx + x;
    // Initialize distributions to equilibrium with zero velocity and unit density
    float rho0 = 1.0f;
    float ux0 = 0.0f;
    float uy0 = 0.0f;

    if (y == Ny - 1) {
        ux0 = u_lid;
        uy0 = 0.0f;
    }
    float uu = ux0 * ux0 + uy0 * uy0;
    for (int i = 0; i < 9; i++) {
        float eu = e_x[i] * ux0 + e_y[i] * uy0;
        float feq = w[i] * rho0 * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * uu);
        f[i * Nx * Ny + y * Nx + x] = feq;
    }
}


__global__ void collision_kernel(float* f, int Nx, int Ny, float tau) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= Nx || y >= Ny) return;
    int idx = y * Nx + x;

	float rho = 0.0f;
	float rhou_x = 0.0f;
	float rhou_y = 0.0f;
#pragma unroll
    for(int i = 0; i < 9; i++) {
        float f_val = f[i * Nx * Ny + idx];
        rho += f_val;
        rhou_x += f_val * e_x[i];
        rhou_y += f_val * e_y[i];
	}
	float ux = rhou_x / rho;
	float uy = rhou_y / rho;

    float uu = ux * ux + uy * uy;
#pragma unroll
    for(int i = 0; i < 9; i++) {
        float eu = e_x[i] * ux + e_y[i] * uy;
        float f_eq = w[i] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * uu);
        f[i * Nx * Ny + idx] = f[i * Nx * Ny + idx] - (f[i * Nx * Ny + idx] - f_eq) / tau;
	}
}

__global__ void stream_kernel(
    float* f_new, float* f, int Nx, int Ny
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

#pragma unroll
    for (int i = 0; i < 9; i++) {
        int x_src = x - e_x[i];
		int y_src = y - e_y[i];
        if (x_src >= 0 && x_src < Nx && y_src >= 0 && y_src < Ny) {
            f_new[i * Nx * Ny + idx] = f[i * Nx * Ny + y_src * Nx + x_src];
        } else {
            // For boundary cells, keep the value (will be overwritten by boundary kernel)
            f_new[i * Nx * Ny + idx] = f[i * Nx * Ny + idx];
        }
    }
}

__global__ void boundary_kernel(float* f, int Nx, int Ny, float u_lid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    // Check if this is a boundary point
    bool is_left = (x == 0);
    bool is_right = (x == Nx - 1);
    bool is_bottom = (y == 0);
    bool is_top = (y == Ny - 1);

    // Only process boundary points
    if (!is_left && !is_right && !is_bottom && !is_top) return;

    // ===== CORNERS (handle first with priority) =====

    if (is_bottom && is_left) {
        // Bottom-left corner: bounce-back all incoming
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];  // right ← left
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // up ← down
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // up-right ← down-left
        return;  // Done with this point
    }

    if (is_bottom && is_right) {
        // Bottom-right corner
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // up ← down
        f[3 * Nx * Ny + idx] = f[1 * Nx * Ny + idx];  // left ← right
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // up-left ← down-right
        return;
    }

    if (is_top && is_left) {
        // Top-left corner (moving lid, but also wall)
        // Apply simplified bounce-back for corner
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];
        f[4 * Nx * Ny + idx] = f[2 * Nx * Ny + idx];
        f[8 * Nx * Ny + idx] = f[6 * Nx * Ny + idx];
        return;
    }

    if (is_top && is_right) {
        // Top-right corner
        f[3 * Nx * Ny + idx] = f[1 * Nx * Ny + idx];
        f[4 * Nx * Ny + idx] = f[2 * Nx * Ny + idx];
        f[7 * Nx * Ny + idx] = f[5 * Nx * Ny + idx];
        return;
    }

    // ===== EDGES (not corners) =====

    if (is_bottom) {
        // Bottom wall: bounce-back
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // up ← down
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // up-right ← down-left
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // up-left ← down-right
        return;
    }

    if (is_top) {
        // Top wall: moving lid (Zou-He)
        float ux = u_lid;

        // Read KNOWN distributions
        float f0 = f[0 * Nx * Ny + idx];
        float f1 = f[1 * Nx * Ny + idx];
        float f3 = f[3 * Nx * Ny + idx];
        float f4 = f[4 * Nx * Ny + idx];
        float f7 = f[7 * Nx * Ny + idx];
        float f8 = f[8 * Nx * Ny + idx];

        // Compute density (Zou-He formula)
        float rho = f0 + f1 + f3 + 2.0f * (f4 + f7 + f8);

        // Compute UNKNOWN distributions
        f[2 * Nx * Ny + idx] = f4;  // Bounce-back in normal direction

        f[5 * Nx * Ny + idx] = f7 - 0.5f * (f1 - f3) + 0.5f * rho * ux;

        f[6 * Nx * Ny + idx] = f8 + 0.5f * (f1 - f3) - 0.5f * rho * ux;  // ✓ MINUS

        return;
    }

    if (is_left) {
        // Left wall: bounce-back
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];  // right ← left
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // up-right ← down-left
        f[8 * Nx * Ny + idx] = f[6 * Nx * Ny + idx];  // down-right ← up-left
        return;
    }

    if (is_right) {
        // Right wall: bounce-back
        f[3 * Nx * Ny + idx] = f[1 * Nx * Ny + idx];  // left ← right
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // up-left ← down-right
        f[7 * Nx * Ny + idx] = f[5 * Nx * Ny + idx];  // down-left ← up-right
        return;
    }
}

__global__ void compute_macro_kernel (
	float* f, float* rho_d, float* ux_d, float* uy_d,
    int Nx, int Ny, float4* data
    ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    int idx = y * Nx + x;
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    float f_local;
#pragma unroll
	for (int i = 0; i < 9; i++) {
        f_local = f[i * Nx * Ny + y * Nx + x];
		rho += f_local;
		ux += f_local * e_x[i];
		uy += f_local * e_y[i];
	}

    rho_d[idx] = rho;
    float vel_x = ux / rho;
    float vel_y = uy / rho;
    ux_d[idx] = vel_x;
    uy_d[idx] = vel_y;

    // Compute velocity magnitude
    float vel_mag = sqrtf(vel_x * vel_x + vel_y * vel_y);

    // Compute direction angle [-pi, pi]
    float angle = atan2f(vel_y, vel_x);

    // Map angle to hue [0, 1]
    float hue = (angle + 3.14159265f) / (2.0f * 3.14159265f);

    // Convert HSV to RGB (H=direction, S=1, V=magnitude)
    float saturation = 1.0f;
    float value = fminf(vel_mag * 50.0f, 1.0f);  // Scale and clamp magnitude

    // HSV to RGB conversion
    float c = value * saturation;
    float hsv_x = c * (1.0f - fabsf(fmodf(hue * 6.0f, 2.0f) - 1.0f));
    float m = value - c;

    float r, g, b;
    if (hue < 1.0f/6.0f) {
        r = c; g = hsv_x; b = 0.0f;
    } else if (hue < 2.0f/6.0f) {
        r = hsv_x; g = c; b = 0.0f;
    } else if (hue < 3.0f/6.0f) {
        r = 0.0f; g = c; b = hsv_x;
    } else if (hue < 4.0f/6.0f) {
        r = 0.0f; g = hsv_x; b = c;
    } else if (hue < 5.0f/6.0f) {
        r = hsv_x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = hsv_x;
    }

    data[idx] = make_float4(r + m, g + m, b + m, 1.0f);
}



void initializeLBM(LBM_State* state, int Nx, int Ny, float tau, float Re) {
    state->Nx = Nx;
    state->Ny = Ny;
    state->tau = tau;
    state->nu = (tau - 0.5f) / 3.0f;
	state->u_lid = state->nu * Re / Ny;

    // Allocate device memory
    cudaMalloc(&state->f_d, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&state->f_new_d, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&state->rho_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->ux_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->uy_d, Nx * Ny * sizeof(float));

    // Setup grid configuration
    state->threads = dim3(16, 16);
    state->blocks = dim3((Nx + 15) / 16, (Ny + 15) / 16);

    // Initialize distributions
    init_kernel <<<state->blocks, state->threads >>> (state->f_d, Nx, Ny, state->u_lid);

    state->initialized = true;
}

void stepLBM(cudaGraphicsResource* cudaResource, LBM_State* state) {
    cudaGraphicsMapResources(1, &cudaResource, 0);

    // Get CUDA array from mapped resource
    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);

    // Allocate device memory for temporary buffer
    float4* d_data;
    cudaMalloc(&d_data, state->Nx * state->Ny * sizeof(float4));

    // Step 1: Collision (in-place on f_d)
    collision_kernel << <state->blocks, state->threads >> > (
        state->f_d, state->Nx, state->Ny, state->tau
        );

    cudaDeviceSynchronize();

    // Step 2: Stream from f_d to f_new_d
    stream_kernel << <state->blocks, state->threads >> > (
        state->f_new_d, state->f_d, state->Nx, state->Ny
        );

    cudaDeviceSynchronize();

    // Step 3: Apply boundary conditions to f_new_d
    boundary_kernel <<<state->blocks, state->threads >> > (
        state->f_new_d, state->Nx, state->Ny, state->u_lid
		);

    cudaDeviceSynchronize();

    // Step 4: Compute macroscopic quantities from f_new_d and prepare visualization
    compute_macro_kernel << <state->blocks, state->threads >> > (
        state->f_new_d, state->rho_d, state->ux_d, state->uy_d,
        state->Nx, state->Ny, d_data
        );

    cudaDeviceSynchronize();

    // Step 5: Swap buffers for next timestep
    float* temp = state->f_d;
    state->f_d = state->f_new_d;
    state->f_new_d = temp;

    // Copy visualization data to texture
    cudaMemcpy2DToArray(
        cudaArray,
        0, 0,
        d_data,
        state->Nx * sizeof(float4),
        state->Nx * sizeof(float4),
        state->Ny,
        cudaMemcpyDeviceToDevice
    );

    // Free temporary buffer
    cudaFree(d_data);

    // Unmap resource
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    // Synchronize
    cudaDeviceSynchronize();
}

void lbm_destroy(LBM_State* state) {
    if (state == NULL) return;

    cudaFree(state->f_d);
    cudaFree(state->f_new_d);
    cudaFree(state->rho_d);
    cudaFree(state->ux_d);
    cudaFree(state->uy_d);

    free(state);
}

// CUDA kernel to fill grid cells with time-varying colors
__global__ void fillColorKernel(float4* data, int width, int height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Normalized coordinates [0, 1]
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    // Create time-varying color pattern
    float r = 0.5f + 0.5f * sinf(u * 3.14159f + time);
    float g = 0.5f + 0.5f * sinf(v * 3.14159f + time * 1.3f);
    float b = 0.5f + 0.5f * sinf((u + v) * 3.14159f + time * 0.7f);

    // Write color to texture (RGBA format)
    data[idx] = make_float4(r, g, b, 1.0f);
}

void initCudaTexture(cudaGraphicsResource** cudaResource, unsigned int glTexture)
{
    // Register OpenGL texture with CUDA
    cudaGraphicsGLRegisterImage(
        cudaResource,
        glTexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
}

void fillGridWithColors(cudaGraphicsResource* cudaResource, int width, int height, float time)
{
    // Map OpenGL texture to CUDA
    cudaGraphicsMapResources(1, &cudaResource, 0);

    // Get CUDA array from mapped resource
    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);

    // Allocate device memory for temporary buffer
    float4* d_data;
    cudaMalloc(&d_data, width * height * sizeof(float4));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    fillColorKernel<<<gridSize, blockSize>>>(d_data, width, height, time);

    // Copy from device buffer to CUDA array (texture)
    cudaMemcpy2DToArray(
        cudaArray,
        0, 0,
        d_data,
        width * sizeof(float4),
        width * sizeof(float4),
        height,
        cudaMemcpyDeviceToDevice
    );

    // Free temporary buffer
    cudaFree(d_data);

    // Unmap resource
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    // Synchronize
    cudaDeviceSynchronize();
}

void cleanupCudaResources(cudaGraphicsResource* cudaResource)
{
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
    }
}
