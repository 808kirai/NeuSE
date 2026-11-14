"""
works out where on the ground a sighting is, from where the unit is and how its camera points
without this a sighting can only be placed at the unit itself, which can be a few hundred metres off

assumes flat ground and a camera with no lens distortion, so the result is an estimate
it is as good as the height, heading, tilt and view angles it is given
"""
import math

# metres in one degree of latitude
metres_per_degree = 111320


def ground_offset(box, frame_size, altitude, tilt, fov):
    # how far ahead and to the right of the unit the centre of a box is on the ground, in metres
    # box is x, y, width and height in pixels and frame_size is the frame's width and height
    # altitude is the camera's height above the ground in metres
    # tilt is how far the camera points below level in degrees, 90 is straight down
    # fov is how wide the camera sees across and down the frame in degrees
    # returns None when the box is at or above the horizon, so it never reaches the ground
    width, height = frame_size
    across, down = fov
    focal_x = (width / 2) / math.tan(math.radians(across) / 2)
    focal_y = (height / 2) / math.tan(math.radians(down) / 2)

    # the ray through the box centre, right and down in the picture with the lens axis as 1
    ray_right = (box[0] + box[2] / 2 - width / 2) / focal_x
    ray_down = (box[1] + box[3] / 2 - height / 2) / focal_y

    # tip the ray by the camera tilt to get its forward and downward parts in the world
    angle = math.radians(tilt)
    forward = math.cos(angle) - ray_down * math.sin(angle)
    downward = math.sin(angle) + ray_down * math.cos(angle)
    if downward <= 1e-6:
        return None

    # stretch the ray until it has dropped the camera's full height
    reach = altitude / downward
    return forward * reach, ray_right * reach


def ground_position(position, pose, box, frame_size, fov, max_range):
    # latitude and longitude of the ground under a box, or None when it cannot be worked out
    # position is the unit's latitude and longitude
    # pose holds altitude in metres above the ground, heading in degrees clockwise from north
    # and tilt in degrees below level, where 90 is straight down
    try:
        altitude, heading, tilt = float(pose["altitude"]), float(pose["heading"]), float(pose["tilt"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (altitude > 0 and 0 < tilt <= 90 and math.isfinite(heading)):
        return None

    offset = ground_offset(box, frame_size, altitude, tilt, fov)
    if offset is None:
        return None
    ahead, right = offset
    # a ray that only just dips below the horizon lands very far away and cannot be trusted
    if math.hypot(ahead, right) > max_range:
        return None

    # turn ahead and right into north and east using the heading
    turn = math.radians(heading)
    north = ahead * math.cos(turn) - right * math.sin(turn)
    east = ahead * math.sin(turn) + right * math.cos(turn)
    lat = position[0] + north / metres_per_degree
    lon = position[1] + east / (metres_per_degree * math.cos(math.radians(position[0])))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


# size of a coverage square in degrees, about 22 m at the equator
cell_degrees = 0.0002


def footprint(position, pose, frame_size, fov, max_range):
    # the four ground corners the frame covers, as latitude and longitude, or None when they cannot be worked out
    # corners go top left, top right, bottom right, bottom left of the picture
    # a frame whose top edge looks at or past the horizon has no closed footprint, so it gives None
    width, height = frame_size
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    points = [ground_position(position, pose, [x, y, 0, 0], frame_size, fov, max_range) for x, y in corners]
    return None if any(p is None for p in points) else points


def cell_of(lat, lon):
    # the coverage square a point falls in, as whole numbers
    return math.floor(lat / cell_degrees), math.floor(lon / cell_degrees)


def cell_area(row):
    # area of one coverage square in square metres, squares get narrower away from the equator
    lat = (row + 0.5) * cell_degrees
    side = cell_degrees * metres_per_degree
    return side * side * math.cos(math.radians(lat))


def inside(point, polygon):
    # true when a point is inside a polygon, by counting edge crossings
    lat, lon = point
    hit = False
    for (lat1, lon1), (lat2, lon2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (lat1 > lat) != (lat2 > lat):
            crossing = lon1 + (lat - lat1) * (lon2 - lon1) / (lat2 - lat1)
            if lon < crossing:
                hit = not hit
    return hit


def cells_in(polygon, most=20000):
    # every coverage square whose centre lies inside the footprint
    # most stops a wildly large footprint from filling the memory
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    low_row, low_col = cell_of(min(lats), min(lons))
    high_row, high_col = cell_of(max(lats), max(lons))
    if (high_row - low_row + 1) * (high_col - low_col + 1) > most:
        return set()
    found = set()
    for row in range(low_row, high_row + 1):
        for col in range(low_col, high_col + 1):
            if inside(((row + 0.5) * cell_degrees, (col + 0.5) * cell_degrees), polygon):
                found.add((row, col))
    return found
