import numpy as np
import torch

def generate_gaussian_peaks(nx, ny, num_peaks=5, width=2.0):
    
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    
    X, Y = np.meshgrid(x, y)
    grid = np.zeros((ny, nx))
    
    peak_positions = np.random.randint(low=int(0.2*min(nx, ny)),
                                       high=int(0.8*min(nx, ny)), 
                                       size=(num_peaks, 2))
    
    for (i, j) in peak_positions:
        gauss = np.exp(-((X - i) ** 2 + (Y - j) ** 2) / (2 * width ** 2))
        grid += gauss
        
    return grid

def solve_wave_equation_2d(nx=50, ny=50, nt=100, c=0.5, dx=1.0, dy=1.0, dt=0.25):
    """
    Solves the 2D wave equation with reflective (Neumann) boundary conditions.
    """
    # Create grid and initialize arrays
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    t = np.linspace(0, (nt - 1) * dt, nt)
    
    # Initial displacement (u0)
    u0 = generate_gaussian_peaks(nx, ny)
    
    # Set up first time step using initial velocity
    u1 = u0 + 0.5 * dt * dt * c * c * (
        np.roll(u0, 1, axis=0) + np.roll(u0, -1, axis=0) + 
        np.roll(u0, 1, axis=1) + np.roll(u0, -1, axis=1) - 4*u0
    )
    
    # Coefficients for the wave equation
    r_x = (c * dt / dx) ** 2
    r_y = (c * dt / dy) ** 2
    
    # Solution storage
    solution = np.zeros((nt, nx, ny))
    solution[0] = u0
    solution[1] = u1
    
    # Time stepping using the wave equation formula
    for n in range(1, nt - 1):
        # Interior points
        u_next = 2*solution[n] - solution[n-1]
        
        # Spatial derivatives for interior points
        u_curr = solution[n]
        
        # Calculate Laplacian with special handling for boundaries
        # For x-direction
        d2x = np.zeros_like(u_curr)
        d2x[1:-1, :] = u_curr[2:, :] - 2*u_curr[1:-1, :] + u_curr[:-2, :]
        # Neumann conditions for x (copy gradient)
        d2x[0, :] = u_curr[1, :] - u_curr[0, :]
        d2x[-1, :] = u_curr[-2, :] - u_curr[-1, :]
        
        # For y-direction
        d2y = np.zeros_like(u_curr)
        d2y[:, 1:-1] = u_curr[:, 2:] - 2*u_curr[:, 1:-1] + u_curr[:, :-2]
        # Neumann conditions for y (copy gradient)
        d2y[:, 0] = u_curr[:, 1] - u_curr[:, 0]
        d2y[:, -1] = u_curr[:, -2] - u_curr[:, -1]
        
        # Combine terms
        u_next += r_x * d2x + r_y * d2y
        
        # Apply Neumann boundary conditions
        u_next[0, :] = u_next[1, :]    # Left boundary
        u_next[-1, :] = u_next[-2, :]  # Right boundary
        u_next[:, 0] = u_next[:, 1]    # Bottom boundary
        u_next[:, -1] = u_next[:, -2]  # Top boundary
        
        solution[n+1] = u_next
    
    return x, y, t, solution

def solve_heat_equation_2d(nx=100, ny=100, nt=100, dx=2, dy=2, alpha=1.0, dt=0.25):
    # Check stability criterion
    stability = alpha * dt * (1/(dx**2) + 1/(dy**2))
    if stability >= 0.5:
        print(f"Warning: Stability criterion not met. Current value: {stability}, should be < 0.5")
        print(f"Consider reducing dt or increasing dx/dy")

    # Create spatial grid
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    t = np.linspace(0, (nt - 1) * dt, nt)

    # Initialize solution array
    u = np.zeros((nt, nx, ny))

    # Initial displacement (u0) with random Gaussian peaks
    u[0] = generate_gaussian_peaks(nx, ny, num_peaks=5)

    # Finite difference coefficients
    cx = alpha * dt / (dx**2)
    cy = alpha * dt / (dy**2)

    # Time stepping loop
    for n in range(nt-1):
        # Update interior points using finite difference scheme
        u[n+1, 1:-1, 1:-1] = u[n, 1:-1, 1:-1] + cx * (u[n, 2:, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1]) + \
                              cy * (u[n, 1:-1, 2:] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2])

        # Apply Neumann boundary conditions (zero flux) - derivative at boundary is zero
        u[n+1, 0, :] = u[n+1, 1, :]    # Left boundary (x = 0)
        u[n+1, nx-1, :] = u[n+1, nx-2, :]  # Right boundary (x = nx-1)
        u[n+1, :, 0] = u[n+1, :, 1]    # Bottom boundary (y = 0)
        u[n+1, :, ny-1] = u[n+1, :, ny-2]  # Top boundary (y = ny-1)

    return x, y, t, u

def make_simulations(number, equation, equation_parameter, k=5, nx=100, ny=100, nt=100, dt=0.25, as_tensor=False):
    
    inputs = np.zeros((0, nx, ny))
    targets = np.zeros((0, nx, ny))

    for _ in range(number):

        if equation == 'wave':
            #print(f"Creating 2D wave equation simulation with parameter {equation_parameter}")
            _, _, _, solution = solve_wave_equation_2d(nx=nx, ny=ny, nt=nt, c=equation_parameter, dt=dt)
        elif equation == 'heat':
            #print(f"Creating 2D heat equation simulation with parameter {equation_parameter}")
            _, _, _, solution = solve_heat_equation_2d(nx=nx, ny=ny, nt=nt, alpha=equation_parameter, dt=dt)

        input_frames = solution[:-k]
        target_frames = solution[k:]

        inputs = np.vstack((inputs, input_frames))
        targets = np.vstack((targets, target_frames))

    if as_tensor:
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

    return inputs, targets


def print_model_information(architecture):
    num_pars = sum(p.numel() for p in architecture.parameters())
    output_dim = architecture(torch.randn(1, 1, 100, 100)).shape
    print(f"Num pars: {architecture.model_name}: --{num_pars}-- and output size {output_dim}")