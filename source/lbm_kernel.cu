#include <glad/glad.h>
#include "lbm_kernel.cuh"
#include <cmath>
#include <cstdio>


// Lattice velocities: e_x[i], e_y[i] for direction i
//   0: rest (0,0)
//   1: east (1,0)    2: north (0,1)    3: west (-1,0)   4: south (0,-1)
//   5: NE (1,1)      6: NW (-1,1)      7: SW (-1,-1)    8: SE (1,-1)
__constant__ int e_x[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
__constant__ int e_y[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };

// Lattice weights
__constant__ float w[9] = {
    4.0f / 9.0f,                                    // rest
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,  // cardinal
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f   // diagonal
};


// opposite[i] gives the direction opposite to i
__constant__ int opposite[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

// Small epsilon to prevent division by zero
__constant__ float eps = 1e-10f;

__device__ inline float compute_feq(int i, float rho, float ux, float uy) {
    float eu = e_x[i] * ux + e_y[i] * uy;
    float uu = ux * ux + uy * uy;
    return w[i] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * uu);
}

// Initialize distributions for lid-driven cavity
// Top row gets lid velocity, everywhere else is at rest
__global__ void init_kernel_lid_cavity(float* f, int Nx, int Ny, float u_lid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    float rho0 = 1.0f;
    float ux0 = 0.0f;
    float uy0 = 0.0f;

    // Top wall (y = Ny-1) moves with velocity u_lid
    if (y == Ny - 1) {
        ux0 = u_lid;
    }

    // Initialize to equilibrium
    for (int i = 0; i < 9; i++) {
        f[i * Nx * Ny + idx] = compute_feq(i, rho0, ux0, uy0);
    }
}

// Initialize distributions for flow over obstacle
// Uniform flow everywhere except inside obstacles
__global__ void init_kernel_flow_obstacle(float* f, int* boundary, int Nx, int Ny, float u_inlet) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    float rho0 = 1.0f;
    float ux0 = u_inlet;
    float uy0 = 0.0f;

    // Zero velocity inside obstacles
    if (boundary[idx] == NO_SLIP) {
        ux0 = 0.0f;
        uy0 = 0.0f;
    }

    // Initialize to equilibrium
    for (int i = 0; i < 9; i++) {
        f[i * Nx * Ny + idx] = compute_feq(i, rho0, ux0, uy0);
    }
}

// Initialize boundary types for flow over obstacle
__global__ void init_boundary_kernel(
    int* boundary, int Nx, int Ny,
    int obstacle_x, int obstacle_y, int obstacle_r
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    // Default: fluid interior
    boundary[idx] = FLUID;

    // Left boundary: inflow (including corners)
    if (x == 0) {
        boundary[idx] = INFLOW_LEFT;
    }
    // Right boundary: outflow (including corners)
    else if (x == Nx - 1) {
        boundary[idx] = OUTFLOW_RIGHT;
    }
    // Top boundary: free-slip (excluding corners, already handled)
    else if (y == Ny - 1) {
        boundary[idx] = FREE_SLIP_TOP;
    }
    // Bottom boundary: free-slip (excluding corners, already handled)
    else if (y == 0) {
        boundary[idx] = FREE_SLIP_BOTTOM;
    }

    // Circular obstacle (highest priority - overrides everything)
    int dx = x - obstacle_x;
    int dy = y - obstacle_y;
    if (dx * dx + dy * dy <= obstacle_r * obstacle_r) {
        boundary[idx] = NO_SLIP;
    }
}

__global__ void collision_kernel_simple(float* f, int Nx, int Ny, float tau) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    float rho = 0.0f;
    float rhou_x = 0.0f;
    float rhou_y = 0.0f;

#pragma unroll
    for (int i = 0; i < 9; i++) {
        float f_val = f[i * Nx * Ny + idx];
        rho += f_val;
        rhou_x += f_val * e_x[i];
        rhou_y += f_val * e_y[i];
    }

    float ux = rhou_x / (rho + eps);
    float uy = rhou_y / (rho + eps);

#pragma unroll
    for (int i = 0; i < 9; i++) {
        float f_eq = compute_feq(i, rho, ux, uy);
        f[i * Nx * Ny + idx] = f[i * Nx * Ny + idx] - (f[i * Nx * Ny + idx] - f_eq) / tau;
    }
}

__global__ void collision_kernel(float* f, int* boundary, int Nx, int Ny, float tau) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    if (boundary[idx] != FLUID) return;

    float rho = 0.0f;
    float rhou_x = 0.0f;
    float rhou_y = 0.0f;

#pragma unroll
    for (int i = 0; i < 9; i++) {
        float f_val = f[i * Nx * Ny + idx];
        rho += f_val;
        rhou_x += f_val * e_x[i];
        rhou_y += f_val * e_y[i];
    }

    float ux = rhou_x / (rho + eps);
    float uy = rhou_y / (rho + eps);

#pragma unroll
    for (int i = 0; i < 9; i++) {
        float f_eq = compute_feq(i, rho, ux, uy);
        f[i * Nx * Ny + idx] = f[i * Nx * Ny + idx] - (f[i * Nx * Ny + idx] - f_eq) / tau;
    }
}

__global__ void stream_kernel(float* f_new, float* f, int Nx, int Ny) {
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
            f_new[i * Nx * Ny + idx] = f[i * Nx * Ny + idx];
        }
    }
}


__global__ void boundary_kernel_lid_cavity(float* f, int Nx, int Ny, float u_lid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    bool is_left = (x == 0);
    bool is_right = (x == Nx - 1);
    bool is_bottom = (y == 0);
    bool is_top = (y == Ny - 1);

    if (!is_left && !is_right && !is_bottom && !is_top) return;


    if (is_bottom && is_left) {
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];  // E <- W
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // N <- S
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // NE <- SW
        return;
    }

    if (is_bottom && is_right) {
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // N <- S
        f[3 * Nx * Ny + idx] = f[1 * Nx * Ny + idx];  // W <- E
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // NW <- SE
        return;
    }

    if (is_top && is_left) {
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];  // E <- W
        f[4 * Nx * Ny + idx] = f[2 * Nx * Ny + idx];  // S <- N
        f[8 * Nx * Ny + idx] = f[6 * Nx * Ny + idx];  // SE <- NW
        return;
    }

    if (is_top && is_right) {
        f[3 * Nx * Ny + idx] = f[1 * Nx * Ny + idx];  // W <- E
        f[4 * Nx * Ny + idx] = f[2 * Nx * Ny + idx];  // S <- N
        f[7 * Nx * Ny + idx] = f[5 * Nx * Ny + idx];  // SW <- NE
        return;
    }

    if (is_bottom) {
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // N <- S
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // NE <- SW
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // NW <- SE
        return;
    }

    if (is_top) {
        float ux = u_lid;
        float f0 = f[0 * Nx * Ny + idx];
        float f1 = f[1 * Nx * Ny + idx];
        float f3 = f[3 * Nx * Ny + idx];
        float f4 = f[4 * Nx * Ny + idx];
        float f7 = f[7 * Nx * Ny + idx];
        float f8 = f[8 * Nx * Ny + idx];

        float rho = f0 + f1 + f3 + 2.0f * (f4 + f7 + f8);

        f[2 * Nx * Ny + idx] = f4;
        f[5 * Nx * Ny + idx] = f7 - 0.5f * (f1 - f3) + 0.5f * rho * ux;
        f[6 * Nx * Ny + idx] = f8 + 0.5f * (f1 - f3) - 0.5f * rho * ux;
        return;
    }

    if (is_left) {
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];  // E <- W
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // NE <- SW
        f[8 * Nx * Ny + idx] = f[6 * Nx * Ny + idx];  // SE <- NW
        return;
    }

    if (is_right) {
        f[3 * Nx * Ny + idx] = f[1 * Nx * Ny + idx];  // W <- E
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // NW <- SE
        f[7 * Nx * Ny + idx] = f[5 * Nx * Ny + idx];  // SW <- NE
        return;
    }
}


__global__ void boundary_kernel_flow_obstacle(float* f, int* boundary, int Nx, int Ny, float u_inlet) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;
    int btype = boundary[idx];

    if (btype == FLUID) return;


    if (btype == NO_SLIP) {
        float temp;

        temp = f[1 * Nx * Ny + idx];
        f[1 * Nx * Ny + idx] = f[3 * Nx * Ny + idx];
        f[3 * Nx * Ny + idx] = temp;

        temp = f[2 * Nx * Ny + idx];
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];
        f[4 * Nx * Ny + idx] = temp;

        temp = f[5 * Nx * Ny + idx];
        f[5 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];
        f[7 * Nx * Ny + idx] = temp;

        temp = f[6 * Nx * Ny + idx];
        f[6 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];
        f[8 * Nx * Ny + idx] = temp;

        return;
    }

    if (btype == INFLOW_LEFT) {
        float ux = u_inlet;
        float uy = 0.0f;
        float rho = 1.0f;  // Target density

        if (y == 0 || y == Ny - 1) {
            for (int i = 0; i < 9; i++) {
                f[i * Nx * Ny + idx] = compute_feq(i, rho, ux, uy);
            }
            return;
        }

        float f0 = f[0 * Nx * Ny + idx];
        float f2 = f[2 * Nx * Ny + idx];
        float f3 = f[3 * Nx * Ny + idx];
        float f4 = f[4 * Nx * Ny + idx];
        float f6 = f[6 * Nx * Ny + idx];
        float f7 = f[7 * Nx * Ny + idx];

        rho = (f0 + f2 + f4 + 2.0f * (f3 + f6 + f7)) / (1.0f - ux);

        rho = fmaxf(0.5f, fminf(rho, 2.0f));

        float rho_ux = rho * ux;
        float rho_uy = rho * uy;

        f[1 * Nx * Ny + idx] = f3 + (2.0f / 3.0f) * rho_ux;
        f[5 * Nx * Ny + idx] = f7 - 0.5f * (f2 - f4) + (1.0f / 6.0f) * rho_ux + 0.5f * rho_uy;
        f[8 * Nx * Ny + idx] = f6 + 0.5f * (f2 - f4) + (1.0f / 6.0f) * rho_ux - 0.5f * rho_uy;

        return;
    }

    if (btype == OUTFLOW_RIGHT) {
        int y_interior = y;
        if (y == 0) y_interior = 1;
        if (y == Ny - 1) y_interior = Ny - 2;

        int idx_interior = y_interior * Nx + (x - 1);

        for (int i = 0; i < 9; i++) {
            f[i * Nx * Ny + idx] = f[i * Nx * Ny + idx_interior];
        }

        return;
    }

    if (btype == OUTFLOW_TOP) {
        int idx_interior = (y - 1) * Nx + x;

        for (int i = 0; i < 9; i++) {
            f[i * Nx * Ny + idx] = f[i * Nx * Ny + idx_interior];
        }

        return;
    }

    if (btype == OUTFLOW_BOTTOM) {
        int idx_interior = (y + 1) * Nx + x;

        for (int i = 0; i < 9; i++) {
            f[i * Nx * Ny + idx] = f[i * Nx * Ny + idx_interior];
        }

        return;
    }

    if (btype == FREE_SLIP_TOP) {
        f[4 * Nx * Ny + idx] = f[2 * Nx * Ny + idx];  // S <- N
        f[8 * Nx * Ny + idx] = f[5 * Nx * Ny + idx];  // SE <- NE
        f[7 * Nx * Ny + idx] = f[6 * Nx * Ny + idx];  // SW <- NW

        return;
    }


    if (btype == FREE_SLIP_BOTTOM) {
        f[2 * Nx * Ny + idx] = f[4 * Nx * Ny + idx];  // N <- S
        f[5 * Nx * Ny + idx] = f[8 * Nx * Ny + idx];  // NE <- SE
        f[6 * Nx * Ny + idx] = f[7 * Nx * Ny + idx];  // NW <- SW

        return;
    }
}

__global__ void compute_macro_kernel_hsv(
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

#pragma unroll
    for (int i = 0; i < 9; i++) {
        float f_local = f[i * Nx * Ny + idx];
        rho += f_local;
        ux += f_local * e_x[i];
        uy += f_local * e_y[i];
    }

    rho_d[idx] = rho;
    float vel_x = ux / (rho + eps);
    float vel_y = uy / (rho + eps);
    ux_d[idx] = vel_x;
    uy_d[idx] = vel_y;

    float vel_mag = sqrtf(vel_x * vel_x + vel_y * vel_y);
    float angle = atan2f(vel_y, vel_x);
    float hue = (angle + 3.14159265f) / (2.0f * 3.14159265f);
    float saturation = 1.0f;
    float value = fminf(vel_mag * 50.0f, 1.0f);

    float c = value * saturation;
    float hsv_x = c * (1.0f - fabsf(fmodf(hue * 6.0f, 2.0f) - 1.0f));
    float m = value - c;

    float r, g, b;
    if (hue < 1.0f / 6.0f) {
        r = c; g = hsv_x; b = 0.0f;
    } else if (hue < 2.0f / 6.0f) {
        r = hsv_x; g = c; b = 0.0f;
    } else if (hue < 3.0f / 6.0f) {
        r = 0.0f; g = c; b = hsv_x;
    } else if (hue < 4.0f / 6.0f) {
        r = 0.0f; g = hsv_x; b = c;
    } else if (hue < 5.0f / 6.0f) {
        r = hsv_x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = hsv_x;
    }

    data[idx] = make_float4(r + m, g + m, b + m, 1.0f);
}

__global__ void compute_macro_kernel_heatmap(
    float* f, float* rho_d, float* ux_d, float* uy_d, int* boundary,
    int Nx, int Ny, float4* data, float vel_scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;

    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

#pragma unroll
    for (int i = 0; i < 9; i++) {
        float f_local = f[i * Nx * Ny + idx];
        rho += f_local;
        ux += f_local * e_x[i];
        uy += f_local * e_y[i];
    }

    rho_d[idx] = rho;
    float vel_x = ux / (rho + eps);
    float vel_y = uy / (rho + eps);
    ux_d[idx] = vel_x;
    uy_d[idx] = vel_y;
    float vel_mag = sqrtf(vel_x * vel_x + vel_y * vel_y);
    float scaled_vel = fminf(vel_mag * vel_scale, 1.0f);

    float r, g, b;
    if (scaled_vel < 0.25f) {
        float t = scaled_vel / 0.25f;
        r = 0.0f;
        g = t;
        b = 1.0f;
    } else if (scaled_vel < 0.5f) {
        float t = (scaled_vel - 0.25f) / 0.25f;
        r = 0.0f;
        g = 1.0f;
        b = 1.0f - t;
    } else if (scaled_vel < 0.75f) {
        float t = (scaled_vel - 0.5f) / 0.25f;
        r = t;
        g = 1.0f;
        b = 0.0f;
    } else {
        float t = (scaled_vel - 0.75f) / 0.25f;
        r = 1.0f;
        g = 1.0f - t;
        b = 0.0f;
    }

    if (boundary[idx] == NO_SLIP) {
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
    }

    data[idx] = make_float4(r, g, b, 1.0f);
}


void setupPhysicalSimulation(PhysicalParams* params, int D_lattice_target, float Ma_target) {
    params->D_lattice = D_lattice_target;

    params->Re = (params->U_phys * params->D_phys) / params->nu_phys;

    float c_s = 1.0f / sqrtf(3.0f);  // Lattice speed of sound
    params->u_lattice = Ma_target * c_s;

    params->dx = params->D_phys / params->D_lattice;

    params->dt = (params->u_lattice * params->dx) / params->U_phys;

    params->nu_lattice = (params->nu_phys * params->dt) / (params->dx * params->dx);

    params->tau = 3.0f * params->nu_lattice + 0.5f;

    params->t_phys = 0.0f;
    params->timestep = 0;

    printf("\n=== Physical to Lattice Conversion ===\n");
    printf("Physical Parameters:\n");
    printf("  Characteristic length: %.4f m\n", params->D_phys);
    printf("  Flow velocity: %.4f m/s\n", params->U_phys);
    printf("  Kinematic viscosity: %.2e m^2/s\n", params->nu_phys);
    printf("\nLattice Parameters:\n");
    printf("  Characteristic length: %d cells\n", params->D_lattice);
    printf("  Grid spacing dx: %.2e m/cell\n", params->dx);
    printf("  Time step dt: %.2e s/iteration\n", params->dt);
    printf("  Velocity u: %.4f lattice units\n", params->u_lattice);
    printf("  Viscosity nu: %.6f lattice units^2\n", params->nu_lattice);
    printf("  Tau: %.4f\n", params->tau);
    printf("  Mach number: %.4f\n", params->u_lattice / c_s);
    printf("  Reynolds number: %.1f\n", params->Re);
    printf("======================================\n\n");
}

void initializeLBM_LidCavity(LBM_State* state, int Nx, int Ny, float tau, float Re) {
    state->Nx = Nx;
    state->Ny = Ny;
    state->tau = tau;
    state->nu = (tau - 0.5f) / 3.0f;


    state->u_lid = state->nu * Re / Ny;

    state->viz_mode = VIZ_HSV_DIRECTION;
    state->timestep = 0;

    float Ma = state->u_lid / (1.0f / sqrtf(3.0f));
    printf("=== Lid-Driven Cavity Initialization ===\n");
    printf("Grid: %d x %d\n", Nx, Ny);
    printf("Reynolds number: %.1f\n", Re);
    printf("Tau: %.4f\n", tau);
    printf("Lid velocity: %.6f lattice units\n", state->u_lid);
    printf("Mach number: %.4f\n", Ma);

    printf("========================================\n\n");

    cudaMalloc(&state->f_d, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&state->f_new_d, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&state->rho_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->ux_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->uy_d, Nx * Ny * sizeof(float));
    state->boundary_d = nullptr;  // Not used for lid cavity

    state->threads = dim3(16, 16);
    state->blocks = dim3((Nx + 15) / 16, (Ny + 15) / 16);

    init_kernel_lid_cavity<<<state->blocks, state->threads>>>(
        state->f_d, Nx, Ny, state->u_lid
    );
    cudaDeviceSynchronize();

    state->initialized = true;
}

void stepLBM_LidCavity(cudaGraphicsResource* cudaResource, LBM_State* state) {
    cudaGraphicsMapResources(1, &cudaResource, 0);

    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);

    float4* d_data;
    cudaMalloc(&d_data, state->Nx * state->Ny * sizeof(float4));

    collision_kernel_simple<<<state->blocks, state->threads>>>(
        state->f_d, state->Nx, state->Ny, state->tau
    );
    cudaDeviceSynchronize();

    stream_kernel<<<state->blocks, state->threads>>>(
        state->f_new_d, state->f_d, state->Nx, state->Ny
    );
    cudaDeviceSynchronize();

    boundary_kernel_lid_cavity<<<state->blocks, state->threads>>>(
        state->f_new_d, state->Nx, state->Ny, state->u_lid
    );
    cudaDeviceSynchronize();

    compute_macro_kernel_hsv<<<state->blocks, state->threads>>>(
        state->f_new_d, state->rho_d, state->ux_d, state->uy_d,
        state->Nx, state->Ny, d_data
    );
    cudaDeviceSynchronize();

    float* temp = state->f_d;
    state->f_d = state->f_new_d;
    state->f_new_d = temp;

    state->timestep++;

    cudaMemcpy2DToArray(
        cudaArray, 0, 0,
        d_data,
        state->Nx * sizeof(float4),
        state->Nx * sizeof(float4),
        state->Ny,
        cudaMemcpyDeviceToDevice
    );

    cudaFree(d_data);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    cudaDeviceSynchronize();
}

void initializeLBM_FlowObstacle(
    LBM_State* state,
    int Nx, int Ny,
    float tau,
    float u_inlet,
    int obstacle_x, int obstacle_y, int obstacle_r
) {
    state->Nx = Nx;
    state->Ny = Ny;
    state->tau = tau;
    state->nu = (tau - 0.5f) / 3.0f;
    state->u_lid = u_inlet;  
    state->u_inlet = u_inlet;

    state->viz_mode = VIZ_HEATMAP;
    state->timestep = 0;

    int D = 2 * obstacle_r;
    float Re = u_inlet * D / state->nu;
    float Ma = u_inlet / (1.0f / sqrtf(3.0f));

    printf("=== Flow Over Obstacle Initialization ===\n");
    printf("Grid: %d x %d\n", Nx, Ny);
    printf("Obstacle: center (%d, %d), radius %d, diameter %d\n",
           obstacle_x, obstacle_y, obstacle_r, D);
    printf("Reynolds number: %.1f\n", Re);
    printf("Tau: %.4f\n", tau);
    printf("Inlet velocity: %.6f lattice units\n", u_inlet);
    printf("Mach number: %.4f\n", Ma);

    if (Ma > 0.15f) {
        printf("WARNING: Mach number %.4f is high!\n", Ma);
    }
    if (tau < 0.55f) {
        printf("WARNING: Tau %.4f is too low!\n", tau);
    }
    printf("=========================================\n\n");

    cudaMalloc(&state->f_d, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&state->f_new_d, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&state->rho_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->ux_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->uy_d, Nx * Ny * sizeof(float));
    cudaMalloc(&state->boundary_d, Nx * Ny * sizeof(int));

    state->threads = dim3(16, 16);
    state->blocks = dim3((Nx + 15) / 16, (Ny + 15) / 16);

    init_boundary_kernel<<<state->blocks, state->threads>>>(
        state->boundary_d, Nx, Ny, obstacle_x, obstacle_y, obstacle_r
    );
    cudaDeviceSynchronize();

    init_kernel_flow_obstacle<<<state->blocks, state->threads>>>(
        state->f_d, state->boundary_d, Nx, Ny, u_inlet
    );
    cudaDeviceSynchronize();

    state->initialized = true;
}

void stepLBM_FlowObstacle(cudaGraphicsResource* cudaResource, LBM_State* state) {
    cudaGraphicsMapResources(1, &cudaResource, 0);

    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);

    float4* d_data;
    cudaMalloc(&d_data, state->Nx * state->Ny * sizeof(float4));

    collision_kernel<<<state->blocks, state->threads>>>(
        state->f_d, state->boundary_d, state->Nx, state->Ny, state->tau
    );
    cudaDeviceSynchronize();

    stream_kernel<<<state->blocks, state->threads>>>(
        state->f_new_d, state->f_d, state->Nx, state->Ny
    );
    cudaDeviceSynchronize();

    boundary_kernel_flow_obstacle<<<state->blocks, state->threads>>>(
        state->f_new_d, state->boundary_d, state->Nx, state->Ny, state->u_inlet
    );
    cudaDeviceSynchronize();


    float vel_scale = 3.0f;
    compute_macro_kernel_heatmap<<<state->blocks, state->threads>>>(
        state->f_new_d, state->rho_d, state->ux_d, state->uy_d, state->boundary_d,
        state->Nx, state->Ny, d_data, vel_scale
    );
    cudaDeviceSynchronize();

    float* temp = state->f_d;
    state->f_d = state->f_new_d;
    state->f_new_d = temp;

    state->timestep++;

    cudaMemcpy2DToArray(
        cudaArray, 0, 0,
        d_data,
        state->Nx * sizeof(float4),
        state->Nx * sizeof(float4),
        state->Ny,
        cudaMemcpyDeviceToDevice
    );

    cudaFree(d_data);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    cudaDeviceSynchronize();
}

void lbm_destroy(LBM_State* state) {
    if (state == nullptr) return;

    cudaFree(state->f_d);
    cudaFree(state->f_new_d);
    cudaFree(state->rho_d);
    cudaFree(state->ux_d);
    cudaFree(state->uy_d);

    if (state->boundary_d != nullptr) {
        cudaFree(state->boundary_d);
    }

    delete state;
}

void initCudaTexture(cudaGraphicsResource** cudaResource, unsigned int glTexture) {
    cudaGraphicsGLRegisterImage(
        cudaResource,
        glTexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
}

void cleanupCudaResources(cudaGraphicsResource* cudaResource) {
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
    }
}

__global__ void fillColorKernel(float4* data, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    float r = 0.5f + 0.5f * sinf(u * 3.14159f + time);
    float g = 0.5f + 0.5f * sinf(v * 3.14159f + time * 1.3f);
    float b = 0.5f + 0.5f * sinf((u + v) * 3.14159f + time * 0.7f);

    data[idx] = make_float4(r, g, b, 1.0f);
}

void fillGridWithColors(cudaGraphicsResource* cudaResource, int width, int height, float time) {
    cudaGraphicsMapResources(1, &cudaResource, 0);

    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);

    float4* d_data;
    cudaMalloc(&d_data, width * height * sizeof(float4));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    fillColorKernel<<<gridSize, blockSize>>>(d_data, width, height, time);

    cudaMemcpy2DToArray(
        cudaArray, 0, 0,
        d_data,
        width * sizeof(float4),
        width * sizeof(float4),
        height,
        cudaMemcpyDeviceToDevice
    );

    cudaFree(d_data);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    cudaDeviceSynchronize();
}
