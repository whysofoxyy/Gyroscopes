import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as Rot


# ---------- Setup: settings ----------

PORT = "COM5"
BAUD_RATE = 115200

ACCEL_GAIN = 1.5
MAX_DT = 0.2

CALIBRATION_TIME = 10      # seconds used to measure the gyro bias

# Set True if Arduino sends gyro values in degrees/second.
# Set False if Arduino sends gyro values in radians/second.
GYRO_IN_DEG_PER_SEC = False

HISTORY_LEN = 300          # number of samples shown in the w graphs


# ---------- Setup: serial connection ----------

ser = serial.Serial(PORT, BAUD_RATE, timeout=0.05)
ser.reset_input_buffer()


# ---------- Data packet receival ----------
# Expected CSV line: time_ms, ax, ay, az, gx, gy, gz

def parse_serial_line(line):

    parts = line.decode(
        "utf-8", errors="ignore"
    ).strip().split(",")

    if len(parts) != 7:
        return None

    try:
        values = np.asarray(parts, dtype=float)
    except ValueError:
        return None

    if not np.all(np.isfinite(values)):
        return None

    return values


def get_gyro_rad(values):

    gyro = values[4:7].astype(float)


    return gyro


# ---------- Gyro bias calculation ----------

def calibrate_gyro_bias():

    print("\nKeep the MPU6050 stationary.")
    print(f"Calibrating gyroscope for {CALIBRATION_TIME} seconds...")

    gyro_samples = []
    start_time = time.perf_counter()

    # collect samples for a fixed time
    while time.perf_counter() - start_time < CALIBRATION_TIME:

        line = ser.readline()

        if not line:
            continue

        values = parse_serial_line(line)

        if values is None:
            continue

        gyro = get_gyro_rad(values)

        gyro_samples.append(gyro)

    if len(gyro_samples) == 0:
        raise RuntimeError("No valid data received during calibration.")

    gyro_samples = np.asarray(gyro_samples)

    # Mean gyroscope bias
    gyro_bias = np.mean(gyro_samples, axis=0)

    # Check calibration stability
    gyro_std = np.std(gyro_samples, axis=0)

    print("\nCalibration Complete!")
    print("Samples used:", len(gyro_samples))
    print("Gyroscope Bias (rad/s):", gyro_bias)
    print("Gyroscope Standard Deviation:", gyro_std)

    return gyro_bias


# Run calibration once at startup. After this, gyro_bias never changes.
gyro_bias = calibrate_gyro_bias()

# Discard old buffered samples
ser.reset_input_buffer()


# ---------- Simulation: cube geometry ----------

vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
], dtype=float)

faces = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 3, 7, 4]
]

face_colors = [
    "peachpuff",
    "pink",
    "yellow",
    "lightblue",
    "lightgreen",
    "darkgreen"
]


# ---------- Quaternion calculation ----------

# Initial quaternion: q = (1, 0, 0, 0)
# SciPy internally uses (x, y, z, w).
# Orientation maps body coordinates to world coordinates.

orientation = Rot.identity()


def update_orientation(accel, gyro, dt):

    global orientation

    # Stage 1: gyroscope integration

    # Angular displacement
    delta_theta = gyro * dt

    # Incremental rotation
    delta_rotation = Rot.from_rotvec(delta_theta)

    # Quaternion update: q_new = q * delta_q
    orientation = orientation * delta_rotation

    # Stage 2: accelerometer correction

    accel_magnitude = np.linalg.norm(accel)

    if not np.isfinite(accel_magnitude):
        return

    if accel_magnitude < 1e-8:
        return

    # Normalized measured gravity direction
    measured_gravity_body = accel / accel_magnitude

    # Predicted gravity direction in body coordinates
    predicted_gravity_body = orientation.inv().apply(
        np.array([0.0, 0.0, 1.0])
    )

    # Cross-product error
    error = np.cross(
        measured_gravity_body,
        predicted_gravity_body
    )

    # Correction angular velocity
    correction_vector = ACCEL_GAIN * error

    # Apply correction
    correction = Rot.from_rotvec(
        correction_vector * dt
    )

    orientation = orientation * correction


# ---------- Plots and simulation window: layout ----------

fig = plt.figure(figsize=(12, 9))

# Row 0: w graphs, Row 1: live simulation
axes = fig.subplots(
    2, 3,
    gridspec_kw=dict(
        top=0.80, bottom=0.07,
        hspace=0.35, wspace=0.30,
        height_ratios=[1, 2.5]
    )
)

# Index: 0,1,2 -> w graphs | 3,4,5 -> bottom row (3D plot)
axes = axes.flatten()

# The bottom row becomes ONE wide 3D plot:
# remove the three 2D axes there and add a 3D axis across that row
gs = axes[3].get_gridspec()

for a in axes[3:]:
    a.remove()

ax = fig.add_subplot(gs[1, :], projection="3d")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

ax.set_title(
    "MPU6050 - Quaternion Based Orientation Estimation"
)

ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=35)


# ---------- Simulation: cube and body axes ----------

cube = Poly3DCollection(
    [vertices[f] for f in faces],
    facecolors=face_colors,
    edgecolors="black",
    linewidths=1.5,
    alpha=0.95
)

ax.add_collection3d(cube)

# Body-fixed coordinate axes
axis_artists = []

for color in ["red", "green", "blue"]:

    line, = ax.plot(
        [], [], [],
        color=color,
        linewidth=3
    )

    axis_artists.append(line)


# ---------- Plots: gyroscope graphs (bias removed) ----------

axis_names = ["x", "y", "z"]
axis_colors = ["red", "green", "blue"]

time_hist = deque(maxlen=HISTORY_LEN)
w_hist = [deque(maxlen=HISTORY_LEN) for _ in range(3)]
t_elapsed = 0.0

w_axes = []
w_lines = []

for i in range(3):

    ax_w = axes[i]
    ax_w.set_title(f"w{axis_names[i]}")
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("rad/s")
    ax_w.grid(True)
    line_w, = ax_w.plot([], [], color=axis_colors[i], linewidth=1.5)
    w_axes.append(ax_w)
    w_lines.append(line_w)


# ---------- Miscellaneous: text displays ----------
# va="top" anchors each text block by its top edge, so it grows downward

angle_text = fig.text(
    0.02, 0.98, "",
    fontsize=11,
    family="monospace",
    va="top"
)

quat_text = fig.text(
    0.70, 0.98, "",
    fontsize=9,
    family="monospace",
    va="top"
)

# Constant gyro bias values (set once, never updated)
bias_text = fig.text(
    0.36, 0.98,
    "Gyro bias (rad/s)\n"
    f"wx bias: {gyro_bias[0]:+.5f}\n"
    f"wy bias: {gyro_bias[1]:+.5f}\n"
    f"wz bias: {gyro_bias[2]:+.5f}",
    fontsize=9,
    family="monospace",
    va="top"
)

status_text = fig.text(
    0.02, 0.025,
    "Waiting for sensor data...",
    fontsize=9
)


# ---------- Simulation update: called every frame ----------

last_timestamp_ms = None


def update(_):

    global last_timestamp_ms
    global t_elapsed

    received_data = False
    valid_samples = 0

    # Process available serial data
    while ser.in_waiting:

        line = ser.readline()

        values = parse_serial_line(line)

        if values is None:
            continue

        # Read sensor values
        timestamp_ms = int(values[0])

        ax_raw, ay_raw, az_raw = values[1:4]

        accel = np.array(
            [ax_raw, ay_raw, az_raw],
            dtype=float
        )

        gyro_measured = get_gyro_rad(values)

        # Remove the gyro bias
        gyro = gyro_measured - gyro_bias

        # Time step computation
        if last_timestamp_ms is None:

            last_timestamp_ms = timestamp_ms
            continue

        # Handles Arduino millis() rollover
        dt_ms = (
            timestamp_ms - last_timestamp_ms
        ) & 0xFFFFFFFF

        last_timestamp_ms = timestamp_ms

        dt = dt_ms / 1000.0

        if dt <= 0 or dt > MAX_DT:
            continue

        # Update the orientation quaternion
        update_orientation(
            accel,
            gyro,
            dt
        )

        # Store gyro data for the graphs
        t_elapsed += dt
        time_hist.append(t_elapsed)

        for i in range(3):
            w_hist[i].append(gyro[i])

        received_data = True
        valid_samples += 1

    # Update gyro graphs
    if len(time_hist) > 1:

        t_list = list(time_hist)
        x_min = t_list[0]
        x_max = max(t_list[-1], x_min + 1e-3)

        # One common y-range computed from all three axes
        all_values = np.concatenate([list(h) for h in w_hist])
        y_min = all_values.min()
        y_max = all_values.max()
        pad = max(0.1 * (y_max - y_min), 1e-3)

        for i in range(3):

            w_lines[i].set_data(t_list, list(w_hist[i]))

            w_axes[i].set_xlim(x_min, x_max)
            w_axes[i].set_ylim(y_min - pad, y_max + pad)   # same for all three

    # Rotate the cube
    R = orientation.as_matrix()

    rotated_vertices = vertices @ R.T

    cube.set_verts([
        rotated_vertices[f] for f in faces
    ])

    # Rotate the body coordinate axes
    axis_ends_body = np.eye(3) * 1.7

    rotated_axes = axis_ends_body @ R.T

    for i, line in enumerate(axis_artists):

        end = rotated_axes[i]

        line.set_data(
            [0, end[0]],
            [0, end[1]]
        )

        line.set_3d_properties(
            [0, end[2]]
        )

    # Euler angle back-calculation and display
    roll, pitch, yaw = orientation.as_euler(
        "xyz",
        degrees=True
    )

    angle_text.set_text(
        f"Roll:  {roll:7.1f} deg\n"
        f"Pitch: {pitch:7.1f} deg\n"
        f"Yaw:   {yaw:7.1f} deg"
    )

    # Quaternion display
    qx, qy, qz, qw = orientation.as_quat()

    quat_text.set_text(
        "Quaternion (x, y, z, w)\n"
        f"x: {qx:+.3f}\n"
        f"y: {qy:+.3f}\n"
        f"z: {qz:+.3f}\n"
        f"w: {qw:+.3f}"
    )

    # Status display
    if received_data:

        status_text.set_text(
            f"MPU6050 Connected | "
            f"Samples: {valid_samples}"
        )

    else:

        status_text.set_text(
            "Waiting for valid sensor data..."
        )

    return [
        cube,
        angle_text,
        quat_text,
        bias_text,
        status_text,
        *axis_artists,
        *w_lines
    ]


# ---------- Miscellaneous: run the animation ----------

ani = FuncAnimation(
    fig,
    update,
    interval=30,
    blit=False,
    cache_frame_data=False
)

try:
    plt.show()

finally:
    ser.close()