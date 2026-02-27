import numpy as np

class Geon:

    hrr = None

    def __init__(self, shape):                          # hrr (?)
        self.shape = shape
        self.relations = []

    def add_relation(self, relation):
        self.relations.append(relation)


class RectangularPrism(Geon):

    def __init__(self, length, direction):              # int, vector(3)
        super().__init__("rectangular_prism")
        self.length = length
        self.direction = direction
        self.normalize_direction()

        self.make_hrr()

    def make_hrr(self):
        # find length and direction in hrrs
        self.hrr = None

    def normalize_direction(self):
        self.direction = self.direction / np.linalg.norm(self.direction)

    def set_endpoints(self, start_coords):
        self.start_coords = start_coords.copy()
        self.end_coords = start_coords + (self.direction * self.length)

    def get_endpoints(self):
        return np.array([self.start_coords, self.end_coords])

class Relation:     # TODO: add direction vector!

    def __init__(self, geon_1, connection_g1, geon_2, connection_g2):           # Geon, Geon, int, int
        self.geon_1 = geon_1
        self.geon_2 = geon_2
        self.connection_g1 = connection_g1
        self.connection_g2 = connection_g2

        self.update_geon_relation(geon_1)
        self.update_geon_relation(geon_2)

    def update_geon_relation(self, geon):
        geon.add_relation(self)

class Object:

    hrrs = []

    def __init__(self, geons, relations, landmark_geon_index):                          # list, list, int
        self.geons = geons
        self.relations = relations
        self.landmark_geon_index = landmark_geon_index
        self.update_start_coords(np.array([0.,0.,0.]))

    def update_start_coords(self, new_coords):
        self.start_coords = new_coords
        self.set_endpoints()

    def set_endpoints(self):

        new_coords = self.start_coords.copy()
        for geon in self.geons:
            geon.set_endpoints(new_coords)

            new_coords = geon.end_coords.copy()

    def get_endpoints(self):

        count = 0
        for geon in self.geons:
            if count == 0:
                endpoints = np.array(geon.get_endpoints())
            else:
                endpoints = np.append(endpoints, np.expand_dims(geon.get_endpoints()[1], axis=0), axis=0)
            count += 1

        return endpoints

    def get_landmark_endpoints(self):
        landmark_geon_endpoints = self.geons[self.landmark_geon_index].get_endpoints()
        # relation_point = self.geons[self.landmark_geon_index+1].get_endpoints()[1]
        # landmark_geon_endpoints = np.append(landmark_geon_endpoints, np.expand_dims(relation_point, axis=0), axis=0)
        return landmark_geon_endpoints
    
    def get_landmark_geon_vector(self):
        landmark_geon = self.geons[self.landmark_geon_index]
        return landmark_geon.direction * landmark_geon.length
    
    def rotate(self, r):
        # update start coords
        self.update_start_coords(r.apply(self.start_coords))

        # update geons
        for geon in self.geons:
            geon.direction = r.apply(geon.direction)
            geon.normalize_direction()
    
def normalize(vector):
    return vector / np.linalg.norm(vector)

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

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


def find_center_point(object):                      # axis-aligned bounding box method

    x_endpoints = [object.start_coords[0]]
    y_endpoints = [object.start_coords[1]]
    z_endpoints = [object.start_coords[2]]

    direction_vector = object.start_coords.copy()
    for geon in object.geons:
        direction_vector = direction_vector + (geon.direction * geon.length)

        x_endpoints.append(direction_vector[0])
        y_endpoints.append(direction_vector[1])
        z_endpoints.append(direction_vector[2])

    x_val = (min(x_endpoints) + max(x_endpoints))/2
    y_val = (min(y_endpoints) + max(y_endpoints))/2
    z_val = (min(z_endpoints) + max(z_endpoints))/2

    return np.array([x_val, y_val, z_val])

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


def _nhat(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > eps else v

def _best_fit_axis_angle_weighted(A_dirs, B_dirs, weights, prev_axis=None, eps=1e-12):
    """
    A_dirs, B_dirs: (N,3) unit directions
    weights: (N,) nonnegative
    Returns: axis (unit), theta (signed, (-pi,pi])
    """
    A = np.asarray(A_dirs, float)  # (N,3)
    B = np.asarray(B_dirs, float)
    w = np.asarray(weights, float).reshape(-1, 1)  # (N,1)

    # Weighted Davenport/Kabsch: H = Σ w_i * b_i a_i^T
    H = (B * w).T @ A  # (3,N) * (N,3) -> (3,3)

    # SVD -> proper rotation
    U, S, Vt = np.linalg.svd(H)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt

    # Extract axis/angle robustly
    tr = np.trace(R)
    c = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))  # cos(theta)
    theta = np.arccos(c)

    if theta < 1e-12:
        axis = np.array([1., 0., 0.]) if prev_axis is None else _nhat(prev_axis)
        theta = 0.0
    elif np.pi - theta < 1e-6:
        eigvals, eigvecs = np.linalg.eig((R + np.eye(3)) / 2.0)
        axis = _nhat(np.real(eigvecs[:, np.argmax(np.real(eigvals))]))
    else:
        # omega = axis * sin(theta)
        omega = 0.5 * np.array([R[2,1] - R[1,2],
                                R[0,2] - R[2,0],
                                R[1,0] - R[0,1]])
        s = np.linalg.norm(omega)  # = sin(theta)
        axis = _nhat(omega / (s + eps))

    # Sign continuity
    if prev_axis is not None and np.dot(axis, prev_axis) < 0:
        axis = -axis
        theta = -theta

    return axis, theta

def _perceived_dir_total(v):
    # Expect: d ~ direction-like 3D, total ~ scalar (>=0).
    d, total = human_perceived_distance_shepard_metzler(v[0], v[1], v[2])
    d = _nhat(np.asarray(d, float))
    total = float(np.asarray(total))  # ensure scalar
    if not np.isfinite(total) or total < 0:
        total = 0.0
    return d, total

def i_angy(original_object, target_object,
           center_coords=np.array([0, 0, 0]),
           step_max_deg=5.0,
           angle_deadband_deg=0.25,
           prev_axis=None, prev_theta=None,
           axis_smooth=0.6):

    # 1) Align centers
    original_center = find_center_point_LWLC(original_object)
    target_center   = find_center_point_LWLC(target_object)
    original_object.update_start_coords(original_object.start_coords + (center_coords - original_center))
    target_object.update_start_coords(target_object.start_coords + (center_coords - target_center))

    # 2) Rays from center to landmarks
    orig_pts = original_object.get_landmark_endpoints()
    targ_pts = target_object.get_landmark_endpoints()
    A_rays = [np.asarray(p, float) - center_coords for p in orig_pts]
    B_rays = [np.asarray(q, float) - center_coords for q in targ_pts]

    # 3) Perceived direction + total for each ray
    A_dirs, A_totals = zip(*[_perceived_dir_total(v) for v in A_rays])
    B_dirs, B_totals = zip(*[_perceived_dir_total(v) for v in B_rays])
    A_dirs = np.vstack(A_dirs)
    B_dirs = np.vstack(B_dirs)

    # Use perceived totals to weight the pairs (avg of orig/target totals)
    weights = (np.asarray(A_totals) + np.asarray(B_totals)) * 0.5
    # Optional: temper to avoid one landmark dominating:
    # weights = np.sqrt(weights + 1e-12)

    # 4) Best-fit rotation (weighted)
    axis, theta_full = _best_fit_axis_angle_weighted(A_dirs, B_dirs, weights, prev_axis=prev_axis)

    # 5) Axis smoothing & renorm
    if prev_axis is not None:
        axis = _nhat(axis_smooth * prev_axis + (1 - axis_smooth) * axis)

    # 6) Deadband + capped step
    deadband = np.deg2rad(angle_deadband_deg)
    step_max = np.deg2rad(step_max_deg)
    if abs(theta_full) < deadband:
        theta_step = 0.0
    else:
        theta_step = np.sign(theta_full) * min(abs(theta_full), step_max)

    return axis, theta_step, theta_full

def find_axis_of_rotation(original_object, target_object, center_coords=np.array([0, 0, 0]), step_size=5, prev_axis=None, prev_direction=1, prev_angles=None):

    smoothness = 0.8
    axis_lock_tol = 1e-4             # NEW: lock axis when Σ||â×b̂|| is tiny
    angle_deadband = np.deg2rad(0.25) # NEW: snap tiny angles to 0 (~0.25°)
    angle_smooth = 0.4               # NEW: optional angle smoothing (0..1)
    eps = 1e-12

    # 1. Get object center coords, landmark coords, and origin

    original_center = find_center_point_LWLC(original_object)
    target_center = find_center_point_LWLC(target_object)

    # 2. Align object centers

    # find distance from center points to new center coords
    original_shift = center_coords - original_center
    target_shift = center_coords - target_center

    # move objects so they are aligned (changed so center point is origin)
    original_object.update_start_coords(original_object.start_coords + original_shift)
    target_object.update_start_coords(target_object.start_coords + target_shift)

    # update landmark values
    original_landmarks = original_object.get_landmark_endpoints()
    target_landmarks = target_object.get_landmark_endpoints()

    # 3. Determine axis of rotation

    # find distance from new center point to landmarks
    center_to_original = [landmark - center_coords for landmark in original_landmarks]
    center_to_target = [landmark - center_coords for landmark in target_landmarks]

    # PERCIEVED DISTANCE
    # # find percieved distance between objects using just one landmark:              # maybe do both and then average????? how is best way to use both distance info?
    # o1_direction, o1_total = human_perceived_distance_shepard_metzler(center_to_original[0][0], center_to_original[0][1], center_to_original[0][2])
    # t1_direction, t1_total = human_perceived_distance_shepard_metzler(center_to_target[0][0], center_to_target[0][1], center_to_target[0][2])
    # o2_direction, o2_total = human_perceived_distance_shepard_metzler(center_to_original[1][0], center_to_original[1][1], center_to_original[1][2])
    # t2_direction, t2_total = human_perceived_distance_shepard_metzler(center_to_target[1][0], center_to_target[1][1], center_to_target[1][2])

    # center_to_original = [(o1_direction) * o1_total, (o2_direction) * o2_total]
    # # o2_vec = 
    # center_to_target = [(t1_direction) * t1_total, (t2_direction) * t2_total]
    # # t2_vec = 

    # landmark_1_vec = t1_vec - o1_vec
    # landmark_2_vec = t2_vec - o2_vec
    # landmark_1_distance = np.linalg.norm(landmark_1_vec)
    # landmark_2_distance = np.linalg.norm(landmark_2_vec)

    # ACTUAL DISTANCE
    # find distance between original and target landmarks
    landmark_1_vec = center_to_target[0] - center_to_original[0]
    landmark_2_vec = center_to_target[1] - center_to_original[1]
    landmark_1_distance = np.linalg.norm(landmark_1_vec)
    landmark_2_distance = np.linalg.norm(landmark_2_vec)

    # normalize center to landmark vectors
    center_to_original = [normalize(landmark) for landmark in center_to_original]
    center_to_target = [normalize(landmark) for landmark in center_to_target]

    # calculate axis of rotation

    # FIND cross products of o1/t1, and o2/t2. Then add them together.
    raw_axis_vector = np.cross(normalize(center_to_original[0]), normalize(center_to_target[0])) + np.cross(normalize(center_to_original[1]), normalize(center_to_target[1]))       # based on weighted average of each landmark
    axis_vector_len = np.linalg.norm(raw_axis_vector)


    axis_vector = normalize(raw_axis_vector)

    print("DISTANCES: "+ str(landmark_1_distance), str(landmark_2_distance))
    print("AXIS VECTOR LEN: " + str(np.linalg.norm(raw_axis_vector)))

    # Handle degeneracies
    dots = [float(np.clip(np.dot(center_to_original[i], center_to_target[i]), -1.0, 1.0)) for i in (0,1)]
    anti_parallel = (dots[0] < -1 + 1e-6) and (dots[1] < -1 + 1e-6)

    if np.linalg.norm(raw_axis_vector) < eps:
        if anti_parallel:
            # 180° case: pick any axis ⟂ to a ray (or use prev_axis if available)
            seed = prev_axis if prev_axis is not None else center_to_original[0]
            k = np.array([1.,0.,0.]) if abs(seed[0]) < 0.9 else np.array([0.,1.,0.])
            raw_axis_vector = k - seed * np.dot(k, seed)
        else:
            # Already aligned or near; fall back to prev_axis if present
            if prev_axis is not None:
                raw_axis_vector = prev_axis
            else:
                # pick stable ⟂ to center_to_original[0]
                k = np.array([1.,0.,0.]) if abs(center_to_original[0][0]) < 0.9 else np.array([0.,1.,0.])
                raw_axis_vector = k - center_to_original[0]*np.dot(k, center_to_original[0])

    axis_vector = normalize(raw_axis_vector)

    # check if direction changed (and swap if it did!)
    if prev_axis is not None:
        if np.dot(axis_vector, prev_axis) < 0:
            axis_vector = -axis_vector

    # the axis of rotation should NOT change too much
    if prev_axis is not None:
        axis_vector = normalize(smoothness * prev_axis + (1-smoothness) * axis_vector)


    # 4. Determine direction of rotation

    # calculate angle for each landmark
    # theta_l1 = np.arctan2(np.dot(axis_vector, np.cross(o1_vec, t1_vec)), np.dot(o1_vec, t1_vec))
    # theta_l2 = np.arctan2(np.dot(axis_vector, np.cross(o2_vec, t2_vec)), np.dot(o2_vec, t2_vec))
    theta_l1 = np.arctan2(np.dot(axis_vector, np.cross(center_to_original[0], center_to_target[0])), np.dot(center_to_original[0], center_to_target[0]))
    theta_l2 = np.arctan2(np.dot(axis_vector, np.cross(center_to_original[1], center_to_target[1])), np.dot(center_to_original[1], center_to_target[1]))

    theta = np.average([theta_l1, theta_l2])
    angles = [theta_l1, theta_l2]

    # # 180° special case
    # if anti_parallel and axis_vector_len < axis_lock_tol:
    #     theta = np.pi

    # NEW: Deadband tiny angles (noise → zero command)
    if abs(theta) < angle_deadband:
        theta = 0.0

    # rotation = np.sign(theta)

    rotation = 0 if theta == 0.0 else (1 if theta > 0 else -1)

    # print("ANGLES: " + str(np.rad2deg(theta_l1)), str(np.rad2deg(theta_l2)))
    print("ANGLE: " + str(np.rad2deg(theta)))

    return axis_vector, rotation, angles















def find_axis_of_rotation_old(original_object, target_object, center_coords=np.array([0, 0, 0]), step_size=5, prev_axis=None, prev_direction=1, prev_angles=None):

    smoothness = 0.8

    # 1. Get object center coords, landmark coords, and origin

    original_center = find_center_point_LWLC(original_object)
    target_center = find_center_point_LWLC(target_object)

    # 2. Align object centers

    # find distance from center points to new center coords
    original_shift = center_coords - original_center
    target_shift = center_coords - target_center

    # move objects so they are aligned (changed so center point is origin)
    original_object.update_start_coords(original_object.start_coords + original_shift)
    target_object.update_start_coords(target_object.start_coords + target_shift)

    # update landmark values
    original_landmarks = original_object.get_landmark_endpoints()
    target_landmarks = target_object.get_landmark_endpoints()

    # 3. Determine axis of rotation

    # find distance from new center point to landmarks
    center_to_original = [landmark - center_coords for landmark in original_landmarks]
    center_to_target = [landmark - center_coords for landmark in target_landmarks]

    # # PERCIEVED DISTANCE
    # # find percieved distance between objects using just one landmark:              # maybe do both and then average????? how is best way to use both distance info?
    # o1_direction, o1_total = human_perceived_distance_shepard_metzler(center_to_original[0][0], center_to_original[0][1], center_to_original[0][2])
    # t1_direction, t1_total = human_perceived_distance_shepard_metzler(center_to_target[0][0], center_to_target[0][1], center_to_target[0][2])
    # o2_direction, o2_total = human_perceived_distance_shepard_metzler(center_to_original[1][0], center_to_original[1][1], center_to_original[1][2])
    # t2_direction, t2_total = human_perceived_distance_shepard_metzler(center_to_target[1][0], center_to_target[1][1], center_to_target[1][2])

    # o1_vec = (normalize(o1_direction) * o1_total) 
    # o2_vec = (normalize(o2_direction) * o2_total) 
    # t1_vec = (normalize(t1_direction) * t1_total) 
    # t2_vec = (normalize(t2_direction) * t2_total) 

    # landmark_1_vec = t1_vec - o1_vec
    # landmark_2_vec = t2_vec - o2_vec
    # landmark_1_distance = np.linalg.norm(landmark_1_vec)
    # landmark_2_distance = np.linalg.norm(landmark_2_vec)

    # ACTUAL DISTANCE
    # find distance between original and target landmarks
    landmark_1_vec = center_to_target[0] - center_to_original[0]
    landmark_2_vec = center_to_target[1] - center_to_original[1]
    landmark_1_distance = np.linalg.norm(landmark_1_vec)
    landmark_2_distance = np.linalg.norm(landmark_2_vec)

    # calculate axis of rotation
    # axis_vector = normalize(np.cross(landmark_1_distance, landmark_2_distance))

    # axis of rotation weighted by distance each landmark still has to travel
    # l1_axis_vector = normalize(np.cross(o1_vec, t1_vec))
    # l2_axis_vector = normalize(np.cross(o2_vec, t2_vec))

    raw_axis_vector = np.cross(landmark_1_vec, landmark_2_vec)
    axis_vector_len = np.linalg.norm(raw_axis_vector)
    axis_vector = normalize(raw_axis_vector)

    theta_weight_l1 = 0.5
    theta_weight_l2 = 0.5

    print("DISTANCES: "+ str(landmark_1_distance), str(landmark_2_distance))

    # if (landmark_1_distance < 0.1):
    #     theta_weight_l1 = 0
    #     l2_axis_vector = o1_vec
    #     smoothness = 0

    # if landmark_2_distance < 0.1:
    #     theta_weight_l2 = 0
    #     l1_axis_vector = o1_vec
    #     smoothness = 0

    # if prev_angles is not None:
    #     # angle_threshold = 3.0/3.4
    #     print("PREV ANGLES: "+ str(np.rad2deg(prev_angles[0])), str(np.rad2deg(prev_angles[1])))

    #     if abs(np.rad2deg(prev_angles[0])) <= step_size/2:
    #         theta_weight_l1 = 0
    #         l2_axis_vector = o1_vec
    #         smoothness = 0

    #     if abs(np.rad2deg(prev_angles[1])) <= step_size/2:
    #         theta_weight_l2 = 0
    #         l1_axis_vector = o1_vec
    #         smoothness = 0

        # if prev_angles[0]/prev_angles[1] < angle_threshold:
        #     # l1 angle smaller, weigh it more
        #     theta_weight_l1 = 0.9
        #     theta_weight_l2 = 0.1

        # if prev_angles[1]/prev_angles[0] < angle_threshold:
        #     # l2 angle smaller, weigh it more
        #     theta_weight_l1 = 0.1
        #     theta_weight_l2 = 0.9

    # ok i made exponent high because w/ the smoothing thing differences in distances get small and NEED to be considered
    # i think more intense smoothing == more intense exponent
 
    # weight_l1 = 0.5
    # weight_l2 = 0.5
    # axis_vector = normalize((weight_l1 * l1_axis_vector) + (weight_l2 * l2_axis_vector))
    # axis_vector = normalize((l1_axis_vector) + (l2_axis_vector))

    # check if direction changed (and swap if it did!)
    if prev_axis is not None:
        if np.dot(axis_vector, prev_axis) < 0:
            axis_vector = -axis_vector

    # the axis of rotation should NOT change too much
    if prev_axis is not None:
        axis_vector = normalize(smoothness * prev_axis + (1-smoothness) * axis_vector)

    # 4. Determine direction of rotation

    # original_landmark_location = o1_vec + o2_vec
    # target_landmark_location = t1_vec + t2_vec
    # # original_landmark_location = center_to_original[0] + center_to_original[1]
    # # target_landmark_location = center_to_target[0] + center_to_target[1]

    # # project vector representing landmark location onto plane perpendicular to axis of rotation
    # l1_perp = original_landmark_location - (np.dot(original_landmark_location, axis_vector)) * axis_vector
    # l2_perp = target_landmark_location - (np.dot(target_landmark_location, axis_vector)) * axis_vector


    # TRY DOING EACH LANDMARK SEPERATLYYYY
    # o1_perp = o1_vec - (np.dot(o1_vec, axis_vector)) * axis_vector
    # o2_perp = o2_vec - (np.dot(o2_vec, axis_vector)) * axis_vector
    # t1_perp = t1_vec - (np.dot(t1_vec, axis_vector)) * axis_vector
    # t2_perp = t2_vec - (np.dot(t2_vec, axis_vector)) * axis_vector

    # # theta = np.arctan2(np.dot(axis_vector, np.cross(l1_perp, l2_perp)), np.dot(l1_perp, l2_perp))
    # theta_l1 = np.arctan2(np.dot(axis_vector, np.cross(o1_perp, t1_perp)), np.dot(o1_perp, t1_perp))
    # theta_l2 = np.arctan2(np.dot(axis_vector, np.cross(o2_perp, t2_perp)), np.dot(o2_perp, t2_perp))

    t1_perp = landmark_1_vec - (np.dot(landmark_1_vec, axis_vector)) * axis_vector
    t2_perp = landmark_2_vec - (np.dot(landmark_2_vec, axis_vector)) * axis_vector

    landmark_1_vec = normalize(landmark_1_vec)
    landmark_2_vec = normalize(landmark_2_vec)

    # theta = np.arctan2(np.dot(axis_vector, np.cross(t1_perp, t2_perp)), np.dot(t1_perp, t2_perp))
    theta = np.arctan2(np.dot(axis_vector, np.cross(center_to_original[0], center_to_target[0])), np.dot(center_to_original[0], center_to_target[0]))
    # theta = np.arctan2(axis_vector_len, np.dot(landmark_1_vec, landmark_2_vec))

    angles = theta

    # print("NEW ANGLES: "+ str(np.rad2deg(theta_l1)), str(np.rad2deg(theta_l2)))


    # maybe do like. try to emphasize an axis that keeps the landmark 1 and landmark 2 angles even 
    # IF difference between prev and current angle is more intense for one angle than another --> emphasize axis of weaker one more 

    # if prev_angles is not None and abs(np.rad2deg(prev_angles[0])) <= step_size and abs(np.rad2deg(theta_l1)) <= step_size:     # if landmark 1 point has hit
    #     rotation = np.sign(theta_l2)

    # elif prev_angles is not None and abs(np.rad2deg(prev_angles[1])) <= step_size and abs(np.rad2deg(theta_l2)) <= step_size:     # if landmark 2 point has hit
    #     rotation = np.sign(theta_l1)

    # else:
    #     if abs(theta_l1+theta_l2) <= step_size:
    #         rotation = prev_direction
    #     else:
    #         rotation = np.sign(theta_l1 + theta_l2)

    rotation = np.sign(theta)

    print(abs(np.rad2deg(theta)))

    return axis_vector, rotation, angles





def find_axis_of_rotation_WORKS(original_object, target_object, center_coords=np.array([0, 0, 0]), prev_axis=[], prev_direction=1):

    # 1. Get object center coords, landmark coords, and origin

    original_center = find_center_point_LWLC(original_object)
    target_center = find_center_point_LWLC(target_object)

    # 2. Align centers

    # find distance from center points to new center coords
    original_center_to_origin = center_coords - original_center
    target_center_to_origin = center_coords - target_center

    # move objects so they are aligned (changed so center point is origin)
    original_object.update_start_coords(original_object.start_coords + original_center_to_origin)
    target_object.update_start_coords(target_object.start_coords + target_center_to_origin)

    # update landmark values
    original_landmarks = original_object.get_landmark_endpoints()
    target_landmarks = target_object.get_landmark_endpoints()

    # 3. Determine axis of rotation

    # PHASE ONE CALCULATION: axis of rotation is orthoganol to the vectors that go from centerpoint to midpoint of each object's landmark geon
    original_geon_midpoint = (original_landmarks[1] + original_landmarks[0])/2
    target_geon_midpoint = (target_landmarks[1] + target_landmarks[0])/2
    axis_vector = normalize(np.cross(original_geon_midpoint, target_geon_midpoint))     

    # PHASE TWO CALCULATION: if midpoint-centerpoint vectors in each object get close, axis is now combo of midpoint-centerpoint vectors!
    phase_two = False
    if len(prev_axis) > 0 and cosine_similarity(original_geon_midpoint, target_geon_midpoint) > 0.975:
        axis_vector = normalize(original_geon_midpoint + target_geon_midpoint)
        phase_two = True

    # 4. Determine direction of rotation

    # check if axis direction swapped in calculation (and negate it if it did!) - "forward" is whatever direction the last axis pointed
    if len(prev_axis) > 0:
        cosine_sim = cosine_similarity(prev_axis, axis_vector)
        if cosine_sim != 0:
            axis_vector = axis_vector * cosine_sim

    # smoothing change in axis
    # if len(prev_axis) > 0:
    #     axis_vector = normalize(0.8 * prev_axis + 0.2 * axis_vector)

    # find distance from center points to landmarks
    center_to_original = [landmark - original_center for landmark in original_landmarks]
    center_to_target = [landmark - target_center for landmark in target_landmarks]

    original_l1 = center_to_original[0]
    target_l1 = center_to_target[0]
    original_l2 = center_to_original[1]
    target_l2 = center_to_target[1]

    # calculate direction of rotation
    if not phase_two:

        # project landmark midpoint vectors onto plane perpendicular to axis of rotation
        original_loc = original_l1 + original_l2
        target_loc = target_l1 + target_l2

        original_perp = original_loc - (np.dot(original_loc, axis_vector)) * axis_vector
        target_perp = target_loc - (np.dot(target_loc, axis_vector)) * axis_vector

        # determine shortest angle between perpendicular landmark midpoint vectors
        theta = np.arctan2(np.dot(axis_vector, np.cross(original_perp, target_perp)), np.dot(original_perp, target_perp))

        # sign should determine direction
        rotation = np.sign(theta)
        if rotation == 0:
            rotation = prev_direction

    else:

        # project each landmark vector onto plane perpendicular to axis of rotation
        og_l1_perp = original_l1 - (np.dot(original_l1, axis_vector)) * axis_vector
        targ_l1_perp = target_l1 - (np.dot(target_l1, axis_vector)) * axis_vector

        og_l2_perp = original_l2 - (np.dot(original_l2, axis_vector)) * axis_vector
        targ_l2_perp = target_l2 - (np.dot(target_l2, axis_vector)) * axis_vector

        # calculate shortest angle for both landmarks
        theta_l1 = np.arctan2(np.dot(axis_vector, np.cross(og_l1_perp, targ_l1_perp)), np.dot(og_l1_perp, targ_l1_perp))
        theta_l2 = np.arctan2(np.dot(axis_vector, np.cross(og_l2_perp, targ_l2_perp)), np.dot(og_l2_perp, targ_l2_perp))

        # add angles, sign should determine direction
        rotation = np.sign(theta_l1 + theta_l2)
        if rotation == 0:
            rotation = prev_direction

    return axis_vector, rotation