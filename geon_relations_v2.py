import numpy as np

class Geon:

    def __init__(self, shape):
        self.shape = shape
        self.spatial_connections = []

    def add_spatial_connection(self, spatial_connection):
        self.spatial_connections.append(spatial_connection)

    def reset_spatial_connections(self):
        self.spatial_connections = []


class SpatialConnection:

    def __init__(self, next_geon, start_point, direction):           # Geon, Geon, vector(3), vector(3)
        self.next_geon = next_geon
        self.start_point = start_point
        self.direction = direction

    def normalize_direction(self):
        self.direction = self.direction / np.linalg.norm(self.direction)

    def get_vector(self):
        return self.direction
    
    def get_start_point(self):
        return self.start_point


class RectangularPrism(Geon):

    def __init__(self, length, direction):              # int, vector(3)
        super().__init__("rectangular_prism")
        self.length = length
        self.direction = direction
        self.normalize_direction()

    def normalize_direction(self):
        self.direction = self.direction / np.linalg.norm(self.direction)

    def set_endpoints(self, start_coords):
        self.start_coords = start_coords.copy()
        self.end_coords = start_coords + (self.direction * self.length)

    def get_endpoints(self):
        return np.array([self.start_coords, self.end_coords])
    
    def get_vector(self):
        # return self.start_coords + (self.direction * self.length)
        return self.direction * self.length

# class Relation:

#     def __init__(self, geon_1, connection_g1, geon_2, connection_g2, direction):           # Geon, Geon, int, int, vector
#         self.geon_1 = geon_1
#         self.geon_2 = geon_2
#         self.connection_g1 = connection_g1
#         self.connection_g2 = connection_g2
#         self.direction = direction

#         self.update_geon_relation(geon_1)
#         self.update_geon_relation(geon_2)
#         self.normalize_direction()

#     def update_geon_relation(self, geon):
#         geon.add_relation(self)

#     def normalize_direction(self):
#         self.direction = self.direction / np.linalg.norm(self.direction)

class Object:

    def __init__(self, geons, landmark_geon_index):                          # list, int
        self.geons = geons
        self.landmark_geon_index = landmark_geon_index
        self.update_start_coords(np.array([0.,0.,0.]))

    def update_start_coords(self, new_coords):
        self.start_coords = new_coords
        self.set_endpoints()

    def set_endpoints(self):

        # set geon endpoints
        new_coords = self.start_coords.copy()
        for geon in self.geons:
            geon.set_endpoints(new_coords)
            new_coords = geon.end_coords.copy()

        # set geon s[atial connections]
        # clear existing spatial connections
        for geon in self.geons:
            geon.reset_spatial_connections()

        # set new spatial connections
        for i in range(len(self.geons) - 1):

            # set forwards spatial connections
            curr_geon = self.geons[i]
            next_geon = self.geons[i+1]

            curr_geon.add_spatial_connection(SpatialConnection(next_geon, curr_geon.end_coords, next_geon.direction))

            # set backwards spatial connections
            backwards_index = len(self.geons) - 1 - i
            curr_geon = self.geons[backwards_index]
            next_geon = self.geons[backwards_index - 1]
            curr_geon.add_spatial_connection(SpatialConnection(next_geon, curr_geon.end_coords, next_geon.direction))


    def rotate(self, r):

        # update geons
        for geon in self.geons:
            geon.direction = r.apply(geon.direction)
            geon.normalize_direction()

        # update start coords
        self.update_start_coords(r.apply(self.start_coords))


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
        landmark_spatial_connection = self.geons[self.landmark_geon_index].spatial_connections[0]
        spatial_connection_point = landmark_spatial_connection.get_vector()
        return np.append(landmark_geon_endpoints, [spatial_connection_point], axis=0)
    
    def get_landmark_geon(self):
        return self.geons[self.landmark_geon_index]
    
    def get_landmark_spatial_connection(self):
        return self.geons[self.landmark_geon_index].spatial_connections[0]
    
def normalize(vector):
    return vector / np.linalg.norm(vector)

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

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


def find_axis_of_rotation(original_object, target_object, center_coords=np.array([0, 0, 0]), step_size=5, prev_axis=None, prev_direction=1, prev_angles=None):

    smoothness = 0.8

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

    # # PERCEIVED DISTANCE
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

    # get direction vectors (from rotation point)
    original_geon_direction = normalize(-original_object.get_landmark_geon().get_vector())   # negated cause its pointing towards the rotation point, we want it to point away
    target_geon_direction = normalize(-target_object.get_landmark_geon().get_vector())

    original_spatcon_direction = normalize(original_object.get_landmark_spatial_connection().get_vector())
    target_spatcon_direction = normalize(target_object.get_landmark_spatial_connection().get_vector())

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

    angles = theta
    rotation = np.sign(theta)

    return axis_vector, rotation, angles



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
