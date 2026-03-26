import numpy as np

'''
FUNCTIONS FOR VECTOR CALCULATIONS IN MODEL
'''

# GENERAL

def normalize(vector):
    return vector / np.linalg.norm(vector)

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def greatest_landmark_distance(landmarks1, landmarks2):
    greatest_distance = 0
    for index in range(len(landmarks1)):
        landmark_diff = landmarks2[index] - landmarks1[index]
        landmark_distance = np.linalg.norm(landmark_diff)
        if landmark_distance > greatest_distance:
            greatest_distance = landmark_distance

    return greatest_distance

def find_center_point_LWLC(object):                      # length weighted line centroid method

    top_sum = 0
    bottom_sum = 0
    endpoints = object.get_endpoints()

    for index in range(len(endpoints) - 1):

        # get geon length
        curr_geon_length = object.geons[index].length

        # get geon midpoint
        start = endpoints[index]
        end = endpoints[index + 1]
        curr_midpoint = (start + end) / 2

        top_sum += curr_geon_length * curr_midpoint
        bottom_sum += curr_geon_length

    return top_sum/bottom_sum

# ROTATION

def make_matrix_representation(v1, v2):
    
    u1 = normalize(v1)

    v2_ortho = v2 - np.dot(v2, u1) * u1
    u2 = normalize(v2_ortho)

    u3 = normalize(np.cross(u1, u2))
    
    return np.column_stack([u1, u2, u3])

def human_perceived_distance_shepard_metzler(
    dx, dy, dz,
    *,
    # axis gains (defaults suit line-drawing stimuli with weak depth cues)
    gx=1.00,          # lateral (x)
    gy=1.08,          # vertical (y) ~8% overestimation
    # depth gain is interpolated from cue strength below
    cue_strength=0.3, # 0=very weak depth cues (typical S–M drawings), 1=rich cues
    gz_weak=0.70,     # depth compression when cues are weak
    gz_rich=0.95,     # depth compression when cues are rich
    # psychophysical shape
    beta=1.0,         # Stevens exponent (≈1 for length)
    p=2.0             # Minkowski pooling order (2 = Euclidean)
):
    """
    Returns (per_axis, total):
      per_axis = np.array([Px, Py, Pz]) perceived magnitudes along x,y,z
      total    = pooled perceived 3D distance

    Notes:
      - Set cue_strength∈[0,1]. For classic Shepard–Metzler drawings, try 0–0.3.
      - Increase gy if verticals look even longer in your render; increase gz_weak
        (toward 1.0) if your stimuli carry stronger depth cues.
    """
    # interpolate depth gain based on cue strength
    gz = gz_weak + (gz_rich - gz_weak) * np.clip(cue_strength, 0.0, 1.0)

    # per-axis perceived magnitudes (near-linear)
    ax = gx * (abs(dx) ** beta)
    ay = gy * (abs(dy) ** beta)
    az = gz * (abs(dz) ** beta)

    if np.isinf(p):
        total = max(ax, ay, az)
    else:
        total = (ax**p + ay**p + az**p) ** (1.0 / p)

    return np.array([ax, ay, az]), total


def find_axis_of_rotation(original_object, target_object, center_coords=np.array([0, 0, 0]), step_size=10, prev_axis=None, prev_direction=1, prev_angle=None, total_angular_disparity=None):

    smoothness = 0.5

    # 1. Get object rotation points

    rotation_point_original = original_object.get_landmark_spatial_connection().get_start_point()           # point where geon vector and spatial connection vector intersect!
    rotation_point_target = target_object.get_landmark_spatial_connection().get_start_point()

    # 2. Align object rotation points

    # find distance from center point (prob origin) to rotation point
    original_shift = center_coords - rotation_point_original
    target_shift = center_coords - rotation_point_target

    # move objects so they are aligned (changed so rotation point is origin)
    original_object.update_start_coords(original_object.start_coords + original_shift)
    target_object.update_start_coords(target_object.start_coords + target_shift)

    # 3. Determine axis of rotation

    # ACTUAL DISTANCE

    # get direction vectors (from rotation point)
    original_geon_direction = normalize(-original_object.get_landmark_geon().get_vector())   # negated cause its pointing towards the rotation point, we want it to point away
    target_geon_direction = normalize(-target_object.get_landmark_geon().get_vector())

    original_spatcon_direction = normalize(original_object.get_landmark_spatial_connection().get_vector())
    target_spatcon_direction = normalize(target_object.get_landmark_spatial_connection().get_vector())

    # PERCIEVED DISTANCE 

    # add n error to x,y, and z axis for each vector
    # x,y axes will have + error, z axis will have - error

    # if prev_angle is not None and total_angular_disparity is not None:
    if prev_angle is not None and total_angular_disparity is not None and step_size < 30:
        added_x = prev_angle/total_angular_disparity
        added_y = prev_angle/total_angular_disparity
        added_z = prev_angle/total_angular_disparity 
    else:
        added_x = 1
        added_y = 1
        added_z = 1
    
    x_error = 1 + (0.2 * added_x)
    y_error = 1 + (0.1 * added_y)
    z_error = 1 + (-0.7 * added_z)

    original_geon_direction_w_error = [original_geon_direction[0] * x_error, original_geon_direction[1] * z_error, original_geon_direction[2] * y_error]
    original_spatcon_direction_w_error = [original_spatcon_direction[0] * x_error, original_spatcon_direction[1] * z_error, original_spatcon_direction[2] * y_error]

    # make matrix representations
    original_matrix = make_matrix_representation(original_geon_direction_w_error, original_spatcon_direction_w_error)
    # original_matrix = make_matrix_representation(original_geon_direction, original_spatcon_direction)
    target_matrix = make_matrix_representation(target_geon_direction, target_spatcon_direction)

    # calculate rotation matrix
    R = target_matrix @ original_matrix.T

    # calculation axis of rotation using eigenvector
    vals, vecs = np.linalg.eig(R)
    idx = np.argmin(np.abs(vals - 1.0))
    axis_vector = normalize(np.real(vecs[:, idx]))     

    print(axis_vector)

    # check if direction changed (and swap if it did!)
    if prev_axis is not None:
        if np.dot(axis_vector, prev_axis) < 0:
            axis_vector = -axis_vector

    # the axis of rotation should NOT change too much
    if prev_axis is not None:
        axis_vector = normalize(smoothness * prev_axis + (1-smoothness) * axis_vector)

    # # 4. Determine direction of rotation

    # Angle in [0, pi]
    tr = np.trace(R)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(c))

    angles = theta
    rotation = np.sign(theta)

    return axis_vector, rotation, angles

# SIMILARITY CHECK

def same_object(original_object, target_object, object_angle_threshold, total_angular_disparity, production_time, propositional_difficulty_time):
    run_time = 0

    # for each geon in object, check geon length and geon's spatial connections
    for i in range(len(original_object.geons)):

        if i != original_object.landmark_geon_index:                # skip landmark geon and spatial connection
            og_geon = original_object.geons[i]
            targ_geon = target_object.geons[i]

            # CHECK GEON LENGTH
            # if original and target geon lengths are different:
            run_time += production_time
            if  og_geon.length != targ_geon.length:
                return False, run_time

            # CHECK SPATIAL CONNECTIONS
            for j in range(len(og_geon.spatial_connections)):

                # check if spatial connections are NOT colinear
                run_time += production_time + (propositional_difficulty_time * total_angular_disparity)
                if cosine_similarity(og_geon.spatial_connections[j].direction, targ_geon.spatial_connections[j].direction) < object_angle_threshold: 
                    return False, run_time

    # if no issues found 
    return True, run_time
