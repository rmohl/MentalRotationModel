import numpy as np

'''
CLASSES FOR OBJECT REPRESENTATION IN MODEL
'''

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

'''
OTHER REPRESENTATION FUNCTIONS
'''

# x: right is positive, y: further away is positive, z: up is positive
vector_dict = {
    "U": np.array([0, 0, 1]),
    "D": np.array([0, 0, -1]),
    "L": np.array([-1, 0, 0]),
    "R": np.array([1, 0, 0]),
    "B": np.array([0, 1, 0]),
    "F": np.array([0, -1, 0])
}

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

def center_to_origin(object):

    origin = np.array([0, 0, 0])
    center_point = find_center_point_LWLC(object)

    # find distance from center point to origin
    shift = origin - center_point

    # move object so center point is at origin)
    object.update_start_coords(object.start_coords + shift)

def load_objects(filename):
    objects = []
    with open(filename, "r") as file:

        for line in file:                           # each line is an object!
            geon_vectors = []
            geons = line.strip().split(",")

            for geon in geons:
                geon_stats = geon.strip().split(" ")
                direction = vector_dict[geon_stats[0]]
                length = int(geon_stats[1])
                geon_vectors.append(RectangularPrism(length, direction))

            # create original object and relations
            objects.append(Object(
                geons = geon_vectors,
                landmark_geon_index = 0                    # always make landmark index the first geon
            ))

            # make center of the object the origin (useful for rotation purposes)
            center_to_origin(objects[-1])

    return objects