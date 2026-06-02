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

# ROTATION

def make_matrix_representation(v1, v2):
    
    u1 = normalize(v1)

    v2_ortho = v2 - np.dot(v2, u1) * u1
    u2 = normalize(v2_ortho)

    u3 = normalize(np.cross(u1, u2))
    
    return np.column_stack([u1, u2, u3])

def align_rotation_points(original_object, target_object, center_coords):

    # 1. Get object rotation points

    rotation_point_original = original_object.get_landmark_spatial_connection().get_start_point()           # point where geon vector and spatial connection vector intersect!
    rotation_point_target = target_object.get_landmark_spatial_connection().get_start_point()

    # 2. Align object rotation points

    # find distance from center point to rotation point
    original_shift = center_coords - rotation_point_original
    target_shift = center_coords - rotation_point_target

    # move objects so they are aligned (changed so rotation point is origin)
    original_object.update_start_coords(original_object.start_coords + original_shift)
    target_object.update_start_coords(target_object.start_coords + target_shift)

def add_error(
    input_vector,
    # axis gains (defaults suit line-drawing stimuli with weak depth cues)
    gx=1.02,          # lateral (x)
    gy=1.01,          # vertical (y)
    gz=0.98
):
    """
    Returns (per_axis):
      per_axis = np.array([Px, Py, Pz]) perceived magnitudes along x,y,z

    Notes:
      - Set cue_strength∈[0,1]. For classic Shepard–Metzler drawings, try 0–0.3.
      - Increase gy if verticals look even longer in your render; increase gz_weak
        (toward 1.0) if your stimuli carry stronger depth cues.
    """
    dx = input_vector[0]
    dy = input_vector[1]            # y and z are switched in graphing
    dz = input_vector[2]

    # per-axis perceived magnitudes (near-linear)
    ax = gx * (dx)                # note w/ multiplication, cardinal axes are not affected!
    ay = gy * (dy)
    az = gz * (dz)

    return np.array([ax, ay, az])


def find_axis_of_rotation_geon_only(original_object, target_object, center_coords=np.array([0, 0, 0]), prev_axis=None, step_size=10, prev_direction=1, prev_angle=None, total_angular_disparity=None):

    smoothness = 0.5

    # 1. Align original and target object's rotation points

    align_rotation_points(original_object, target_object, center_coords)

    # 2. Determine axis of rotation needed to align geons

    # get geon vectors
    original_geon_direction = normalize(-original_object.get_landmark_geon().get_vector())   # negated cause its pointing towards the rotation point, we want it to point away
    target_geon_direction = normalize(-target_object.get_landmark_geon().get_vector())

    # add error on first axis calculation
    if prev_angle is None and total_angular_disparity is None:
        original_geon_direction = add_error(original_geon_direction)

    # calculate exact axis of rotation using cross product
    axis_vector = np.cross(original_geon_direction, target_geon_direction)

    # check if direction changed from previous step (and swap if it did!)
    if prev_axis is not None:
        if np.dot(axis_vector, prev_axis) < 0:
            axis_vector = -axis_vector

    # the axis of rotation should NOT change too much
    if prev_axis is not None:
        axis_vector = normalize(smoothness * prev_axis + (1-smoothness) * axis_vector)

    # 3. Determine angle and direction of rotation

    theta = np.arctan2(np.linalg.norm(axis_vector), np.dot(original_geon_direction, target_geon_direction))
    angle = theta
    direction = np.sign(theta)

    return axis_vector, direction, np.rad2deg(angle)

def find_axis_of_rotation_geon_and_spatcon(original_object, target_object, center_coords=np.array([0, 0, 0]), step_size=10, prev_axis=None, prev_direction=1, prev_angle=None, total_angular_disparity=None):

    smoothness = 0.5

    # 1. Align original and target object's rotation points
    
    align_rotation_points(original_object, target_object, center_coords)

    # 2. Determine axis of rotation

    # get geon vectors
    original_geon_direction = normalize(-original_object.get_landmark_geon().get_vector())   # negated cause its pointing towards the rotation point, we want it to point away
    target_geon_direction = normalize(-target_object.get_landmark_geon().get_vector())

    # get spatial connection vectors
    original_spatcon_direction = normalize(original_object.get_landmark_spatial_connection().get_vector())
    target_spatcon_direction = normalize(target_object.get_landmark_spatial_connection().get_vector())

    # add error on first axis calculation
    if prev_angle is None and total_angular_disparity is None:
        original_geon_direction = add_error(original_geon_direction)
        original_spatcon_direction = add_error(original_spatcon_direction)

    # make matrix representations
    original_matrix = make_matrix_representation(original_geon_direction, original_spatcon_direction)
    target_matrix = make_matrix_representation(target_geon_direction, target_spatcon_direction)

    # calculate rotation matrix
    R = target_matrix @ original_matrix.T

    # calculation axis of rotation using eigenvector
    vals, vecs = np.linalg.eig(R)
    idx = np.argmin(np.abs(vals - 1.0))
    axis_vector = normalize(np.real(vecs[:, idx]))     

    # check if direction changed (and swap if it did!)
    if prev_axis is not None:
        if np.dot(axis_vector, prev_axis) < 0:
            axis_vector = -axis_vector

    # the axis of rotation should NOT change too much
    if prev_axis is not None:
        axis_vector = normalize(smoothness * prev_axis + (1-smoothness) * axis_vector)

    # 4. Determine direction of rotation

    # Angle in [0, pi]
    tr = np.trace(R)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(c))

    angle = theta
    direction = np.sign(theta)

    # print("prev_angle: " + str(prev_angle))
    # print("theta: " + str(theta))
    # print(switch_direction)

    # if prev_angle != None and not switch_direction and prev_angle < theta:
    #     print("NOOOO")
    #     axis_vector = -axis_vector
    #     switch_direction = True

    return axis_vector, direction, np.rad2deg(angle)

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

    # add error on first axis calculation
    if prev_angle is None and total_angular_disparity is None:
        original_geon_direction = add_error(original_geon_direction)
        original_spatcon_direction = add_error(original_spatcon_direction)

    # make matrix representations
    original_matrix = make_matrix_representation(original_geon_direction, original_spatcon_direction)
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

    angle = theta
    rotation = np.sign(theta)

    return axis_vector, rotation, np.rad2deg(angle)

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

def set_closest_cardinal_axis(axis_vector):

    # only base axes ort simple combinations allowed too? maybe depends on person..
    # try only base ones first

    axis_vector_abs = np.abs(axis_vector)

    max_val_pos = np.argmax(axis_vector_abs)

    direction = np.sign(axis_vector)[max_val_pos]

    new_axis_vector = np.zeros(3)
    new_axis_vector[max_val_pos] = 1 * direction

    return new_axis_vector
