#ifndef LBM_KERNEL_CUH
#define LBM_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

enum CollisionType {
    FLUID = 0,           // Interior fluid cell
    NO_SLIP = 1,         // Bounce-back (obstacles, stationary walls)
    MOVING_LID = 2,      // Moving wall (lid-driven cavity top)
    INFLOW_LEFT = 3,     // Zou-He velocity inlet (left boundary)
    INFLOW_RIGHT = 4,    // Zou-He velocity inlet (right boundary)
    INFLOW_TOP = 5,      // Zou-He velocity inlet (top boundary)
    INFLOW_BOTTOM = 6,   // Zou-He velocity inlet (bottom boundary)
    OUTFLOW_LEFT = 7,    // Zero-gradient outlet (left boundary)
    OUTFLOW_RIGHT = 8,   // Zero-gradient outlet (right boundary)
    OUTFLOW_TOP = 9,     // Zero-gradient outlet (top boundary)
    OUTFLOW_BOTTOM = 10, // Zero-gradient outlet (bottom boundary)
    FREE_SLIP_TOP = 11,  // Slip wall (top)
    FREE_SLIP_BOTTOM = 12, // Slip wall (bottom)
    FREE_SLIP_LEFT = 13,   // Slip wall (left)
    FREE_SLIP_RIGHT = 14   // Slip wall (right)
};

enum VisualizationMode {
    VIZ_HSV_DIRECTION = 0,  // HSV color based on velocity direction (lid cavity)
    VIZ_HEATMAP = 1         // Heatmap based on velocity magnitude (flow obstacle)
};

typedef struct {
    int Nx, Ny;

    float tau;
    float nu;
    float u_lid;      // Lid velocity for cavity / inlet velocity for obstacle
    float u_inlet;    // Alias for inlet velocity

    float* f_d;
    float* f_new_d;
    float* rho_d;
    float* ux_d;
    float* uy_d;
    int* boundary_d;

    dim3 blocks;
    dim3 threads;

    int timestep;
    bool initialized;

    VisualizationMode viz_mode;

} LBM_State;

typedef struct {
    // Physical parameters (inputs)
    float D_phys;          // Characteristic length (meters) - e.g., cylinder diameter
    float U_phys;          // Flow velocity (m/s)
    float nu_phys;         // Kinematic viscosity (m²/s)

    // Lattice parameters (derived)
    int D_lattice;         // Characteristic length (lattice units)
    float dx;              // Grid spacing (meters per lattice unit)
    float dt;              // Time step (seconds per iteration)
    float u_lattice;       // Velocity (lattice units/timestep)
    float nu_lattice;      // Viscosity (lattice units²/timestep)
    float tau;             // Relaxation time
    float Re;              // Reynolds number

    // Time conversion
    float t_phys;          // Current physical time (seconds)
    int timestep;          // Current iteration number
} PhysicalParams;

// Initialize CUDA-OpenGL interop texture
void initCudaTexture(cudaGraphicsResource** cudaResource, unsigned int glTexture);

// Cleanup CUDA resources
void cleanupCudaResources(cudaGraphicsResource* cudaResource);


// Initialize LBM for lid-driven cavity
// - Top wall moves at velocity u_lid (computed from Re, tau, Ny)
// - Other walls are stationary bounce-back
void initializeLBM_LidCavity(LBM_State* state, int Nx, int Ny, float tau, float Re);

// Step the lid-driven cavity simulation
void stepLBM_LidCavity(cudaGraphicsResource* cudaResource, LBM_State* state);

// Convert physical parameters to lattice units
void setupPhysicalSimulation(PhysicalParams* params, int D_lattice_target, float Ma_target);

// Initialize LBM for flow over obstacle
// - Inflow on left with velocity u_inlet
// - Outflow on right (zero-gradient)
// - Free-slip on top/bottom
// - Circular obstacle at (obstacle_x, obstacle_y) with radius obstacle_r
void initializeLBM_FlowObstacle(
    LBM_State* state,
    int Nx, int Ny,
    float tau,
    float u_inlet,
    int obstacle_x, int obstacle_y, int obstacle_r
);

void stepLBM_FlowObstacle(cudaGraphicsResource* cudaResource, LBM_State* state);

void lbm_destroy(LBM_State* state);

#endif 
