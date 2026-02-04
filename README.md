# LBM Fluid Simulation

GPU-accelerated 2D fluid dynamics using the Lattice Boltzmann Method (LBM) with real-time OpenGL visualization.

## Overview

This project implements two classic CFD benchmarks:

- **LBM_LidCavity** - Lid-driven cavity flow (square domain, moving top wall)
- **LBM_FlowObstacle** - Flow over a circular cylinder (Karman vortex shedding)

## Algorithm

### Lattice Boltzmann Method

Alternative to FVM, LBM simulates fluid dynamics by tracking particle distribution functions on a discrete lattice. The method is inherently parallel and well-suited for GPU computation.

**D2Q9 Lattice**: 9 velocity directions in 2D
```
  6   2   5
    \ | /
  3 - 0 - 1
    / | \
  7   4   8
```

**Simulation Steps** (each timestep):
1. **Collision** - Relax distributions toward equilibrium (BGK operator)
2. **Streaming** - Propagate distributions to neighboring cells
3. **Boundary** - Apply boundary conditions

### Boundary Conditions

| Type | Description | Application |
|------|-------------|-------------|
| Bounce-back | No-slip wall | Obstacles, cavity walls |
| Zou-He | Velocity inlet/outlet | Inflow, moving lid |
| Zero-gradient | Extrapolation outlet | Outflow |
| Free-slip | Specular reflection | Channel walls |

### Key Parameters

- **tau** - Relaxation time (controls viscosity: nu = (tau - 0.5) / 3)
- **Re** - Reynolds number (ratio of inertial to viscous forces)
- **Ma** - Mach number (should be < 0.1 for incompressible flow, explodes at fast speeds)

## Requirements

- **CUDA Toolkit** 11.0+ (tested with 12.6)
- **CMake** 3.18+
- **C++17** compiler (MSVC, GCC, Clang)
- **OpenGL** 3.3+ capable GPU
- **NVIDIA GPU** with compute capability 7.5+ (RTX 20xx or newer recommended)

## Building

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"  # Windows
cmake --build . --config Release
```

## Configuration

Edit parameters in the respective `main_*.cpp` files:

**Lid Cavity** (`main_lid_cavity.cpp`):
```cpp
const int GRID_WIDTH = 256;
const int GRID_HEIGHT = 256;
const float TAU = 0.6f;
const float REYNOLDS = 400.0f;
```
As first test, the popular lid driven cavity test indicates the desired functionality of the solver
![Lid driven cavity](KarmanVerti.mp4)
**Flow Obstacle** (`main_flow_obstacle.cpp`):
```cpp
const int GRID_WIDTH = 1600;
const int GRID_HEIGHT = 400;
const int OBSTACLE_RADIUS = 40;
const float TAU = 1.1f;
const float INLET_VELOCITY = 0.04f;
```
The simulation shows the distinctive Kármán vortex street, indicating, that it works as desired
![Kármán vortex street](KarmanVerti.mp4)

## Visualization

- **Lid Cavity**: HSV color mapping (hue = flow direction, brightness = speed)
- **Flow Obstacle**: Heatmap (blue = slow, red = fast, black = obstacle)

## Controls

- **ESC** - Exit simulation

## References

- Mohamad, A.A. "Lattice Boltzmann Method: Fundamentals and Engineering Applications"
- Sukop & Thorne "Lattice Boltzmann Modeling"
- Zou & He (1997) "On pressure and velocity boundary conditions for the lattice Boltzmann BGK model"
