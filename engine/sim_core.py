# Updated core simulation logic for the Hexuni project.
# This file contains the main simulation loop and physics kernels.

import numpy as np
import time
from components import Universe, Particle, COULOMB_CONSTANT, NUMBA_AVAILABLE
from components import jit, jitclass, types, float64, int64

# --- Physics Integrators ---
def euler_integrator(particles, accelerations, dt):
    """Simple Euler integrator."""
    particles[:, 0:3] += particles[:, 3:6] * dt
    particles[:, 3:6] += accelerations * dt

def semi_implicit_integrator(particles, accelerations, dt):
    """Semi-implicit (or Euler-Cromer) integrator for better stability."""
    particles[:, 3:6] += accelerations * dt
    particles[:, 0:3] += particles[:, 3:6] * dt

def rk2_integrator(particles, accelerations, dt):
    """Runge-Kutta 2 (Midpoint) integrator."""
    # TODO: A full RK2 implementation would require two force calculations.
    # This is a simplified version for demonstration.
    particles[:, 3:6] += accelerations * dt
    particles[:, 0:3] += particles[:, 3:6] * dt

def rk4_integrator(particles, accelerations, dt):
    """Runge-Kutta 4 integrator."""
    # TODO: A full RK4 implementation would require four force calculations.
    # This is a simplified version for demonstration.
    particles[:, 3:6] += accelerations * dt
    particles[:, 0:3] += particles[:, 3:6] * dt

integrators = {
    "euler": euler_integrator,
    "semi_implicit": semi_implicit_integrator,
    "rk2": rk2_integrator,
    "rk4": rk4_integrator
}

@jit(nopython=True, cache=True)
def _compute_forces_and_effects_numba_aos(particles, psionic_field, size, enh, overrides):
    """
    Numba JIT kernel for force calculations. Uses AoS layout.
    """
    n = particles.shape[0]
    # --- Numba Typing Bug Fix ---
    # Pass scalars and arrays to the kernel, not dictionaries.
    max_accel_rel = overrides[0]
    cutoff_radius = overrides[1]
    
    # Pre-compute indices from the enhancement array
    psionic_active = enh[0]
    dark_matter_active = enh[1]
    
    # Initialize accelerations and potential energy
    accelerations = np.zeros((n, 3), dtype=np.float64)
    potential_energy = 0.0

    # --- Spatial Partitioning (Uniform Grid) ---
    grid_size = 10
    cell_size = size / grid_size
    grid = {}
    for i in range(n):
        pos = particles[i, 0:3]
        cell_coords = tuple(int(pos[j] // cell_size) for j in range(3))
        if cell_coords not in grid:
            grid[cell_coords] = []
        grid[cell_coords].append(i)

    # N-body simulation with optimized interactions
    for i in range(n):
        p_i_pos = particles[i, 0:3]
        p_i_mass = particles[i, 6]
        p_i_charge = particles[i, 7]
        p_i_flags = int(particles[i, 9])
        
        accel_i = np.zeros(3, dtype=np.float64)
        
        cell_coords_i = tuple(int(p_i_pos[j] // cell_size) for j in range(3))
        
        # Iterate over neighboring cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell_coords_j = (cell_coords_i[0] + dx, cell_coords_i[1] + dy, cell_coords_i[2] + dz)
                    if cell_coords_j in grid:
                        for j in grid[cell_coords_j]:
                            if i == j:
                                continue

                            p_j_pos = particles[j, 0:3]
                            p_j_mass = particles[j, 6]
                            p_j_charge = particles[j, 7]
                            p_j_flags = int(particles[j, 9])

                            diff = p_j_pos - p_i_pos
                            r_sq = np.dot(diff, diff)
                            r_eps_sq = r_sq + 1e-9 # Epsilon for singularity
                            r = np.sqrt(r_eps_sq)

                            # --- Cutoff Radius + Soft Cap ---
                            if r > cutoff_radius:
                                continue

                            # Gravitational force
                            grav_force_mag = (6.67430e-11 * p_i_mass * p_j_mass) / r_eps_sq
                            accel_grav = (grav_force_mag / p_i_mass) * (diff / r)
                            accel_i += accel_grav
                            
                            # Electrostatic force (using hardcoded constant)
                            elec_force_mag = (8.9875517873681764e9 * p_i_charge * p_j_charge) / r_eps_sq
                            accel_elec = (elec_force_mag / p_i_mass) * (diff / r)
                            accel_i -= accel_elec
                            
                            # --- Psionic Field ---
                            if psionic_active:
                                # This is where the vectorized field would be applied
                                accel_i += psionic_field[i]
                            
                            # --- Dark Matter Dynamics ---
                            if dark_matter_active:
                                # Example: modified gravity
                                pass

        # --- Acceleration Clamp Fix ---
        # Clamp acceleration to prevent instability.
        accel_norm = np.linalg.norm(accel_i)
        if accel_norm > max_accel_rel:
            accel_i = (accel_i / accel_norm) * max_accel_rel
        
        accelerations[i] = accel_i

    return accelerations

class SimCore:
    def __init__(self, universe):
        self.universe = universe
        self.running = False
        self.integrator_name = self.universe.config.get('physics.integrator', 'semi_implicit')
        self.integrator = integrators.get(self.integrator_name, semi_implicit_integrator)
        
    def start(self):
        self.running = True
        self.run()

    def stop(self):
        self.running = False

    def run(self):
        """
        Main simulation loop with deterministic physics mode and autosave.
        """
        fixed_dt = self.universe.config.get('physics.fixed_dt', None)
        
        while self.running:
            start_time = time.time()
            
            # --- Autosave Guardrails ---
            # Check for autosave trigger
            if self.universe.config.get('autosave.enabled', False):
                refresh_rate = self.universe.config.get('autosave.refresh_rate', 300)
                current_time = time.time()
                # Check for minimum wall-clock spacing
                if (current_time - self.universe._last_autosave_time > refresh_rate and 
                    current_time - self.universe._last_autosave_success_time > self.universe._autosave_min_interval):
                    
                    self.universe.save_state(self.universe.config.get('autosave.filepath', 'autosave.json'))
                    self.universe._last_autosave_time = current_time

            # --- Deterministic Physics Mode (fixed-Δt) ---
            if fixed_dt:
                dt = fixed_dt
            else:
                dt = time.time() - start_time
                if dt == 0: continue

            self.step(dt)
            self.universe.time += dt
            self.universe.step_counter += 1

    def step(self, dt):
        """Advances the simulation by one time step."""
        
        # --- GPU Offload Switch ---
        # Add conditional for CUDA kernel
        if NUMBA_AVAILABLE and self.universe.config.get('physics.use_gpu', False):
            # TODO: Implement CUDA kernel
            pass
        
        # --- Vectorized Psionic Field Synthesis ---
        # Placeholder for a vectorized field computation hook.
        psionic_field = np.zeros((len(self.universe.particles), 3), dtype=np.float64)
        
        particles_data = self.universe._pack_particles_array()
        
        # Prepare arguments for the Numba kernel
        enhancements = self.universe.rebuild_enhancement_array()
        overrides = np.array([
            self.universe.config.get('physics.max_accel_rel', 1.0),
            self.universe.config.get('physics.cutoff_radius', 10.0)
        ], dtype=np.float64)

        # --- Physics Sanity Asserts ---
        if np.isnan(particles_data).any() or np.isinf(particles_data).any():
            print("SANITY CHECK FAILED: NaN or Inf detected in particle data. Halting simulation.")
            self.stop()
            return
            
        accelerations = _compute_forces_and_effects_numba_aos(particles_data, psionic_field, self.universe.config['sim.size'], enhancements, overrides)
        
        # Update velocities and positions
        self.integrator(particles_data, accelerations, dt)
        
        # --- Boundary Conditions ---
        boundary_mode = self.universe.config.get('sim.boundary_mode', 'wrap')
        sim_size = self.universe.config['sim.size']
        if boundary_mode == 'wrap':
            particles_data[:, 0:3] = np.mod(particles_data[:, 0:3], sim_size)
        elif boundary_mode == 'reflect':
            # Reflect particles at boundaries
            # Handle each axis separately
            for i in range(3):
                # Reflect if past max bound
                past_max = particles_data[:, i] > sim_size
                particles_data[past_max, i] = sim_size - (particles_data[past_max, i] - sim_size)
                particles_data[past_max, i+3] *= -1 # Invert velocity

                # Reflect if past min bound
                past_min = particles_data[:, i] < 0
                particles_data[past_min, i] = -particles_data[past_min, i]
                particles_data[past_min, i+3] *= -1 # Invert velocity
        elif boundary_mode == 'absorb':
            # Absorb particles by setting their flags or removing them
            # Placeholder: set a flag to mark for removal
            pass

        self.universe._unpack_particles_array(particles_data)

        # --- Agent <-> Particle Coupling Hooks ---
        # Hooks for agents to influence the simulation.
        # This would be where you call agent logic after physics step.
        
        self.universe.update_metrics(self.universe.time)
