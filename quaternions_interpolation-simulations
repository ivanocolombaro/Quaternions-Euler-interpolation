import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
import time
import tracemalloc

# Define start and end rotations using Euler angles (in degrees)
euler_start = [0, 0, 0]
euler_end = [90, 45, 30]

# Convert to quaternions
r_start = R.from_euler('xyz', euler_start, degrees=True)
r_end = R.from_euler('xyz', euler_end, degrees=True)
quat_start = r_start.as_quat()
quat_end = r_end.as_quat()

# Number of interpolation steps
steps = 100
times_euler = []
times_slerp = []
memory_euler = []
memory_slerp = []

# Interpolation using Euler angles
euler_path = []
tracemalloc.start()
start_time = time.time()
for t in np.linspace(0, 1, steps):
    interpolated = (1 - t) * np.array(euler_start) + t * np.array(euler_end)
    euler_path.append(interpolated)
times_euler.append(time.time() - start_time)
memory_euler.append(tracemalloc.get_traced_memory()[1])
tracemalloc.stop()

# Interpolation using SLERP
tracemalloc.start()
start_time = time.time()
key_times = [0, 1]
key_rots = R.from_quat([quat_start, quat_end])
slerp = Slerp(key_times, key_rots)
interp_rots = slerp(np.linspace(0, 1, steps))
slerp_path = interp_rots.as_euler('xyz', degrees=True)
times_slerp.append(time.time() - start_time)
memory_slerp.append(tracemalloc.get_traced_memory()[1])
tracemalloc.stop()

# Convert Euler path to numpy array for plotting
euler_path = np.array(euler_path)

# Plot orientation paths
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(euler_path[:, 0], label='Euler X')
axs[0].plot(euler_path[:, 1], label='Euler Y')
axs[0].plot(euler_path[:, 2], label='Euler Z')
axs[0].set_title('Euler Angle Interpolation')
axs[0].legend()

axs[1].plot(slerp_path[:, 0], label='SLERP X')
axs[1].plot(slerp_path[:, 1], label='SLERP Y')
axs[1].plot(slerp_path[:, 2], label='SLERP Z')
axs[1].set_title('Quaternion SLERP Interpolation')
axs[1].legend()

plt.tight_layout()
plt.savefig("interpolation_paths.png")
plt.show()
plt.close()

# Plot performance metrics
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].bar(['Euler', 'SLERP'], [times_euler[0], times_slerp[0]])
axs[0].set_title('Computation Time (s)')

axs[1].bar(['Euler', 'SLERP'], [memory_euler[0] / 1024, memory_slerp[0] / 1024])
axs[1].set_title('Peak Memory Usage (KB)')

plt.tight_layout()
#plt.savefig("performance_metrics.png")
plt.show()
plt.close()

print("Generated interpolation and performance comparison plots.")
