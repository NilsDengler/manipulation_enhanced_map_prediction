import pdb
import math
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import heapq

from skimage.morphology import h_maxima

from shapely import Polygon, MultiPolygon
from scipy.ndimage import distance_transform_edt
from skimage.draw import line
from scipy.ndimage import label as connected_components
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import LineString, Polygon, Point

from scipy.ndimage import distance_transform_edt, label, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
def get_average_probability(prob_map, point, radius):
    """
    Get the average probability within a circular region around a point.
    Uses a bounding box to reduce unnecessary computation.
    """
    x_center, y_center = point
    radius = int(np.ceil(radius))

    # Define bounding box, clipped to map boundaries
    x_min = max(0, x_center - radius)
    x_max = min(prob_map.shape[0], x_center + radius + 1)
    y_min = max(0, y_center - radius)
    y_max = min(prob_map.shape[1], y_center + radius + 1)

    # Extract local region
    local_region = prob_map[x_min:x_max, y_min:y_max]

    # Compute distances inside local region
    x, y = np.ogrid[x_min:x_max, y_min:y_max]
    dist_sq = (x - x_center)**2 + (y - y_center)**2
    mask = dist_sq <= radius**2

    values = local_region[mask]
    return np.mean(values) if values.size > 0 else 0


def is_far_enough(new_point, selected_points, threshold):
    if not selected_points:
        return True

    selected_array = np.array(selected_points)
    distances = np.linalg.norm(selected_array - new_point, axis=1)
    return np.all(distances >= threshold)


# Helper function to calculate uncertainty-based distance (using Dijkstra)
def compute_uncertainty_distance_map(uncertainty_map):
    # Dimensions of the map
    rows, cols = uncertainty_map.shape

    # Initialize distance map with infinity
    distance_map = np.full_like(uncertainty_map, np.inf, dtype=float)

    # Min-heap priority queue for Dijkstra's algorithm (stores tuples of (distance, row, col))
    pq = []

    # Push all cells with value 0 into the priority queue (multi-source)
    for r in range(rows):
        for c in range(cols):
            if uncertainty_map[r, c] == 0:
                distance_map[r, c] = 0
                heapq.heappush(pq, (0, r, c))  # (distance, row, col)

    # Directions for 4-connected grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Dijkstra's algorithm (priority queue based)
    while pq:
        current_dist, r, c = heapq.heappop(pq)

        # If we already found a smaller distance for this cell, continue
        if current_dist > distance_map[r, c]:
            continue

        # Check all 4 neighboring cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Calculate the distance to the neighbor through this cell
                new_dist = current_dist + uncertainty_map[nr, nc]

                # If we found a smaller distance, update and push to the queue
                if new_dist < distance_map[nr, nc]:
                    distance_map[nr, nc] = new_dist
                    heapq.heappush(pq, (new_dist, nr, nc))

    return distance_map

def generate_points(map, threshold=0.5, uncertain_distance=False):
    # Thresholding
    if uncertain_distance:
        distance_map = compute_uncertainty_distance_map(map)
        binary = np.where(distance_map > 0, 1, 0)
    else:
        _, binary = cv2.threshold(map, threshold, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        # Compute distance transform
        distance_map = distance_transform_edt(binary)

    # Get coordinates of ones and their distances
    coordinates_of_ones = np.argwhere(binary == 1)
    distances = distance_map[binary == 1]

    # Sort coordinates by distance descending
    sorted_indices = np.argsort(-distances)
    sorted_coordinates = coordinates_of_ones[sorted_indices]
    sorted_distances = distances[sorted_indices]

    # Select well-separated top points
    selected_points = []
    min_distance = 20

    for i, point in enumerate(sorted_coordinates):
        if all(np.linalg.norm(point - np.array(p)) >= min_distance for p in selected_points):
            selected_points.append(point)
            if len(selected_points) == 4:
                break

    # Calculate average probabilities (optional loop depending on get_average_probability impl)
    #average_probabilities = [
    #    get_average_probability(map, point, sorted_distances[i])
    #    for i, point in enumerate(sorted_coordinates[:len(selected_points)])
    #]

    return distance_map, binary, selected_points


def inverse_raycast(start_point, end_point, semantics, occupancy, conf):
    ray_segments = []

    rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])
    labels = semantics[rr, cc]
    occ_vals = occupancy[rr, cc]
    conf_vals = conf[rr, cc]

    current_label = labels[0]
    segment_length = 1
    conf_sum = occ_vals[0] if current_label == 0 else conf_vals[0]
    count = 1

    for i in range(1, len(labels)):
        label = labels[i]
        confidence = occ_vals[i] if label == 0 else conf_vals[i]

        if label == current_label:
            segment_length += 1
            conf_sum += confidence
            count += 1
        else:
            avg_conf = conf_sum / count if count > 0 else 0
            ray_segments.append((current_label, segment_length, avg_conf))

            current_label = label
            segment_length = 1
            conf_sum = confidence
            count = 1

    # Add the last segment
    avg_conf = conf_sum / count if count > 0 else 0
    ray_segments.append((current_label, segment_length, avg_conf))

    return ray_segments

def compute_group_metrics(labels, width, length_sum, conf_sum, occ_length_sum, start, end):
    avg_length = length_sum / width
    avg_occupancy = conf_sum / occ_length_sum if occ_length_sum > 0 else 0
    avg_occupancy_length = occ_length_sum / width if width > 0 else 0
    return (labels, avg_length, width, start, end, avg_occupancy, avg_occupancy_length)

def build_corridors(ray_list, target_list, max_width=200):
    grouped_segments = []


    current_labels = remove_and_collapse([seg[0] for seg in ray_list[0]])
    current_width = 1
    start = target_list[0]
    current_length = sum(seg[1] for seg in ray_list[0])
    total_conf = sum(seg[2] * seg[1] for seg in ray_list[0] if seg[0] == 0)
    total_occ_length = sum(seg[1] for seg in ray_list[0] if seg[0] == 0)

    for i in range(1, len(ray_list)):
        segments = ray_list[i]
        labels_only = [seg[0] for seg in segments]
        collapsed = remove_and_collapse(labels_only)

        if collapsed == current_labels and current_width < max_width:
            current_width += 1
            current_length += sum(seg[1] for seg in segments)
            total_conf += sum(seg[2] * seg[1] for seg in segments if seg[0] == 0)
            total_occ_length += sum(seg[1] for seg in segments if seg[0] == 0)
        else:
            end = target_list[i - 1]
            grouped_segments.append(compute_group_metrics(
                current_labels, current_width, current_length, total_conf, total_occ_length, start, end
            ))

            # Start new group
            current_labels = collapsed
            current_width = 1
            start = target_list[i]
            current_length = sum(seg[1] for seg in segments)
            total_conf = sum(seg[2] * seg[1] for seg in segments if seg[0] == 0)
            total_occ_length = sum(seg[1] for seg in segments if seg[0] == 0)

    # Final group
    end = target_list[-1]
    grouped_segments.append(compute_group_metrics(
        current_labels, current_width, current_length, total_conf, total_occ_length, start, end
    ))

    return grouped_segments


def find_push_corridor(point, semantics, conf, occupancy):
    height, width = semantics.shape

    # Vectorized candidate targets (scan across row 74)
    y_coords = np.arange(width)
    target_list = np.column_stack((np.full_like(y_coords, 74), y_coords))

    # Run inverse raycasting
    ray_list = [
        inverse_raycast(point, tuple(target), semantics, occupancy, conf)
        for target in target_list
    ]

    # Group and score
    grouped_segments = build_corridors(ray_list, target_list)
    best, best_score = best_segments(grouped_segments, length=True)[0]

    return best, best_score


def remove_and_collapse(labels):
    # First, filter out 14 and 15
    filtered_labels = [label for label in labels if label != 0]
    if not filtered_labels:
        return []

    collapsed_labels = [filtered_labels[0]]  # Start with the first label

    for label in filtered_labels[1:]:
        if label != collapsed_labels[-1]:  # Only add if it's not the same as the last one
            collapsed_labels.append(label)

    return collapsed_labels


def draw_group(map,point,segment,score):
    _,_,width, start, end, _,_ = segment
    rr, cc = line(point[0], point[1], start[0],start[1])  # Ray from POI to front
    rr_2, cc_2 = line(point[0], point[1], end[0],end[1])  # Ray from POI to front

    plt.imshow(map, cmap="jet", interpolation="nearest")
    plt.title(f"Score {score}")
    plt.plot(cc, rr, color="red", linewidth=1)  # Draw the ray
    plt.plot(cc_2, rr_2, color="red", linewidth=1)  # Draw the ray

    plt.show()

def score_ray_group(group, legth = True):
    labels, length, width, start,end, avg_occupancy, avg_occupancy_length = group

    # Normalize lengths
    norm_length = length / 200
    norm_unknown = avg_occupancy_length / length

    # Weights (tuned to balance the scale)
    w_obj = 5.0
    if legth:
        w_length = 5.0
    else:
        w_length = -10.0
    w_conf = 2.0
    w_width = 5.0
    w_width = 5.0


    num_objects = 0
    for label in labels:
        if label !=0:
            num_objects += 1

    score = (
        w_obj * num_objects +
        w_length * norm_length +
        w_conf * avg_occupancy * norm_unknown -
        w_width * math.log1p(width)
    )

    return score


def best_segments(groups, length=True, top_n=1):
    # Compute scores for all groups
    scored_groups = [(group, score_ray_group(group, length)) for group in groups]

    # Sort by score in ascending order (lower is better)
    scored_groups.sort(key=lambda x: x[1])

    # Return the top N
    return scored_groups[:top_n]



def first_object_hit(semantic_map, start, end):
    # Get the line's points using Bresenham's algorithm or a similar approach
    rr, cc = line(start[0], start[1], end[0], end[1])

    # Efficiently loop through the ray and return on first hit
    for r, c in zip(rr, cc):
        if semantic_map[r, c] != 0:  # assuming 0 is background
            return semantic_map[r, c], (r, c)

    # No object hit
    return None, None

def extract_object(semantic_map, class_label, hit_point):
    # Create binary mask of just the class label
    class_mask = semantic_map == class_label

    # Connected component labeling on the class mask
    labeled_mask, num_objects = connected_components(class_mask)

    # Get the instance ID at the hit point
    instance_id = labeled_mask[hit_point[0], hit_point[1]]

    if instance_id == 0:
        return None  # No object found at the hit point (or invalid hit point)

    # Extract the object directly without re-running connected components
    object_mask = labeled_mask == instance_id
    return object_mask


def object_to_push(semantic_map, start, point):
    class_label, hit_point = first_object_hit(semantic_map, start, point)
    if hit_point:
        # Directly extract the object mask without plotting, unless needed
        object_mask = extract_object(semantic_map, class_label, hit_point)
        return object_mask
    return None


def compute_center(mask):
    coords = np.argwhere(mask)
    return np.mean(coords, axis=0).astype(int)  # returns (row, col)


def generate_directions(center, radius, num_directions=144):
    directions = []
    cx, cy = center

    # Precompute angle range and angular step
    start_angle = math.radians(60)  # 60 degrees
    end_angle = math.radians(300)  # 300 degrees
    angle_range = end_angle - start_angle
    angle_step = angle_range / (num_directions - 1)

    # Generate all directions
    for i in range(num_directions):
        angle = start_angle + angle_step * i
        dx = int(round(radius * math.cos(angle)))
        dy = int(round(radius * math.sin(angle)))
        tx = min(max(0, cx + dx), 74)
        ty = min(max(0, cy + dy), 149)

        directions.append((tx, ty))

    return directions


def find_best_push_direction(center, directions, sem, conf, occupancy):
    ray_list = []
    for target in directions:
        ray_list.append(inverse_raycast(center, target, sem, occupancy, conf))

    # Build corridors once, avoid intermediate steps
    segments = build_corridors(ray_list, directions, max_width=15)
    corridors = best_segments(segments, length=False, top_n=5)
    #draw_group(sem,center,best,best_score)
    return corridors


def get_connected_components(m):
        # get connected_components
        binary = m.copy() > 0
        binary[:2, :] = 0

        erode_kernel = np.ones((3, 3), np.uint8)
        eroded_binary = cv2.erode(binary.astype(np.uint8), erode_kernel, iterations=3)
        cc_output = cv2.connectedComponentsWithStats(eroded_binary.astype(np.uint8) * 255, 4, cv2.CV_32S)
        connected_maps = np.zeros((cc_output[0], cc_output[1].shape[0], cc_output[1].shape[1]), dtype=np.uint8)
        connected_mask = cc_output[1]
        a, b = np.where(eroded_binary)
        c = np.stack((a, b), axis=1)
        interp = NearestNDInterpolator(c.tolist(), connected_mask[connected_mask != 0])
        a, b = np.where(1 - eroded_binary)
        filled_data = interp(a, b)
        flood_fill = np.zeros_like(connected_mask)
        flood_fill[a, b] = filled_data
        to_fill = (binary - eroded_binary).astype(bool)
        connected_mask[to_fill] = flood_fill[to_fill]
        for i in range(cc_output[0]):
            connected_maps[i] = connected_mask == i
        #return connected_maps
        components_idices = [np.vstack(np.where(connected_mask == i)).T for i in range(cc_output[0])]
        return components_idices, connected_maps

def get_first_intersection(line, polygon):
    inter = line.intersection(polygon)
    if inter.is_empty:
        return None
    elif isinstance(inter, Point):
        return np.array(inter.coords[0])
    elif hasattr(inter, 'geoms'):
        return np.array(inter.geoms[0].coords[0])
    else:
        return np.array(inter.coords[0])

def extend_line(p1, p2, length=1000):
    """Extend a line beyond p2 in direction from p1 to p2"""
    v = np.array(p2) - np.array(p1)
    v = v / np.linalg.norm(v)
    return LineString([p2 + v * length,p2])

def sample_along_object_boundary(contour, sec_int1, sec_int2, v1, v2, num_samples=3):
    # Find indices on the contour closest to sec_int1 and sec_int2
    dists1 = np.linalg.norm(contour - sec_int1, axis=1)
    dists2 = np.linalg.norm(contour - sec_int2, axis=1)
    idx1 = np.argmin(dists1)
    idx2 = np.argmin(dists2)

    # Get contour section (handle wrapping)
    if idx1 <= idx2:
        path = contour[idx1:idx2+1]
    else:
        path = np.concatenate((contour[idx1:], contour[:idx2+1]), axis=0)

    # Uniformly sample num_samples points along this path
    path = np.array(path)
    cumdist = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    cumdist = np.insert(cumdist, 0, 0)

    total_length = cumdist[-1]
    sample_distances = np.linspace(0, total_length, num_samples)

    sampled_points = []
    for d in sample_distances:
        idx = np.searchsorted(cumdist, d)
        if idx == 0:
            pt = path[0]
            alpha = 0.0
        else:
            t = (d - cumdist[idx - 1]) / (cumdist[idx] - cumdist[idx - 1])
            pt = (1 - t) * path[idx - 1] + t * path[idx]
            alpha = d / total_length

        vec = alpha * v1 + (1 - alpha) * v2
        vec = vec / np.linalg.norm(vec)
        scale = np.random.uniform(10, 50)
        vec_scaled = vec * scale
        end_pt = pt + vec_scaled

        sampled_points.append((pt, end_pt))


    return np.array(sampled_points).astype(int)

def push_sample(mask, center, end_a, end_b, num_samples = 20):
    # Find intersection of corridor vectors with object mask
    center = np.array((center[1], center[0]))
    end_a = np.array((end_a[1], end_a[0]))
    end_b = np.array((end_b[1], end_b[0]))

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obj_poly = Polygon(contours[0][:, 0, :])
    contour = contours[0][:, 0, :]  # (N, 2) shape

    # Convert to shapely lines
    line_a = LineString([end_a, center])
    line_b = LineString([end_b, center])

    int_a = get_first_intersection(line_a, obj_poly)
    int_b = get_first_intersection(line_b, obj_poly)

    #plt.imshow(mask, cmap='gray')
    #plt.plot([center[0], end_a[0]], [center[1], end_a[1]], 'r-', label='Line A')
    #plt.plot([center[0], end_b[0]], [center[1], end_b[1]], 'g-', label='Line B')
    #plt.scatter(*zip(center), c='yellow', label='Center')
    #plt.legend()
    #plt.title("Check Push Corridor Geometry")
    #plt.show()
    if int_a is None or int_b is None:
        int_a = center
        int_b = center
        return None
        #raise ValueError("Push corridor lines didn't intersect object.")

        # Cross lines: a -> int_b, b -> int_a
    ext_line1 = extend_line(end_a, int_b)
    ext_line2 = extend_line(end_b, int_a)
    v2 = np.array(end_a) - np.array(int_b)
    v1 = np.array(end_b) - np.array(int_a)

    # Intersect extended lines with object polygon (again) for the second side
    sec_int1 = get_first_intersection(ext_line1, obj_poly)
    sec_int2 = get_first_intersection(ext_line2, obj_poly)

    if sec_int1 is None:
        sec_int1 = int_b
    if sec_int2 is None:
        sec_int2 = int_a


        #plt.imshow(mask, cmap='gray')
    #plt.plot([center[0], end_a[0]], [center[1], end_a[1]], 'r-', label='Line A')
    #plt.plot([center[0], end_b[0]], [center[1], end_b[1]], 'g-', label='Line B')
    #plt.scatter(*zip(sec_int1), c='blue', label='int1')
    #plt.scatter(*zip(sec_int2), c='orange', label='int2')

    #plt.legend()
    #plt.title("Check Push Corridor Geometry")
    #plt.show()
    return sample_along_object_boundary(contour, sec_int1, sec_int2, v1, v2, num_samples)

def to_instance_map(semantic_map, debug = False):
    free_space_mask = np.logical_not(np.isin(semantic_map, [14, 15]).astype(np.uint8))
    #smoothed = gaussian_filter(free_space_mask.astype(float), sigma=0.5)
    distance = distance_transform_edt(free_space_mask)

    smoothed_distance = gaussian_filter(distance, sigma=1.0)
    #markers = label(local_maxi)[0]
    h_max = h_maxima(smoothed_distance, h=1.0)
    markers = label(h_max)[0]


    labels_ws = watershed(-distance, markers, mask=free_space_mask)
    if debug:
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))

        axs[0].imshow(free_space_mask, cmap='gray')
        axs[0].set_title("Free Space Mask (14 & 15)")

        axs[1].imshow(distance, cmap='magma')
        axs[1].set_title("Distance Transform")

        axs[2].imshow(h_max, cmap='gray')
        axs[2].set_title("H-Maxima Peaks")

        axs[3].imshow(labels_ws, cmap='nipy_spectral')
        axs[3].set_title("Watershed Instances")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    return labels_ws


def get_samples(corridor_vis, point, label,sem_conf,occupacy_uncertainty,num_samples = 10,debug = False):
    start = corridor_vis[3]
    mask = object_to_push(label, start, point)

    if mask is not None:

        center = compute_center(mask)
        directions = generate_directions(center, radius=40, num_directions=144)
        best_corridor = find_best_push_direction(
            center, directions, label, sem_conf, occupacy_uncertainty
        )
        #directions_time = time.time()
        #print('Time taken for direction: {} seconds'.format(directions_time - scoring))

        all_samples = []
        #for corridor, score in best_corridor[]:
        corridor, score = best_corridor[0]
        _, _, width, start, end, _, _ = corridor
        samples = push_sample(mask, center, start, end, num_samples)
        all_samples.append((corridor, samples, score))
        #end_push = time.time()
        #print('Time taken for puchvector: {} seconds'.format(end_push - directions_time))

        #print('Time taken for push push: {} seconds'.format(end_push - start_push))
        #plt.imshow(mask, cmap='gray')
        #for pt, end in samples:
        #    plt.scatter(end[0], end[1], c='blue')
        #    plt.scatter(pt[0], pt[1], c='red')
        #plt.legend()
        #plt.title("Points Sampled Along Object Border")
        #plt.show()
        if debug :
            debug_vis(occupacy_uncertainty,label ,samples, center,corridor, corridor_vis, point, mask)
        if samples is not None:
            shift = np.array([21, 15])
            samples = samples + shift
        return samples

    #plt.figure(2)
    #plt.imshow(occupancy_dist)
    #plt.title('Distance Occupancy at time {}'.format(i))
    #plt.figure(3)
    #plt.imshow(occupancy_binary)
    #plt.title('Binary Occupancy at time {}'.format(i))

    #plt.show()
    #end_push = time.time()
    #print('Time taken for push push: {} seconds'.format(end_push - start_push))
    return None

def generate_push_samples(p_map, u_map, label, sem_conf, num_samples = 20, uncertain_distance = False ,debug = True):

    mapped_arr = p_map
    u_map = u_map[::-1, ::-1][15:90, 21:-21]
    #plt.imshow(mapped_arr, cmap='gray')
    #plt.show()
    label = label[15:90, 21:-21]
    label = to_instance_map(label)
    # plt.imshow(self.get_semantic_rgb_image(previous_semantic_map)[0][::-1, ::-1, :])
    sem_conf = sem_conf[15:90, 21:-21]
    # np.save("semantic_map",self.get_semantic_rgb_image(previous_semantic_map))
    # plt.title('Semantic Map at time {}'.format(i))
    #init_time = time.time()
    #print('Time taken for init: {} seconds'.format(init_time - start_push))
    occupancy_dist, occupancy_binary, selected_points = generate_points(mapped_arr, 0.2, uncertain_distance)
    #point_gener_time = time.time()
    #print('Time taken for point generation: {} seconds'.format(point_gener_time - init_time))
    #plt.imshow(occupancy_dist)
    #for point in selected_points:
    #    plt.scatter(point[1], point[0],color="blue")
    #plt.show()
    results = [
        (find_push_corridor(point, label, sem_conf, u_map), point)
        for point in selected_points
    ]

    (best, best_score), best_point = min(results, key=lambda x: x[0][1])
    sorted_results = sorted(results, key=lambda x: x[0][1])

    #scoring = time.time()
    #print('Time taken for scroeing: {} seconds'.format(scoring - point_gener_time))
    sample_list = []
    for result in sorted_results:
        samples = get_samples(result[0][0],result[1],label,sem_conf,u_map,num_samples,debug)
        if samples is not None:
            sample_list.append(samples)



    return sample_list

    #plt.figure(2)


    #plt.figure(3)
    #plt.imshow(occupancy_binary)
    #plt.title('Binary Occupancy at time {}'.format(i))

    #plt.show()
    #end_push = time.time()
    #print('Time taken for push push: {} seconds'.format(end_push - start_push))



def debug_vis(occupacy, semantic, samples, center, push_corridor, view_corridor, target, mask):
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    axs[0].imshow(occupacy, cmap='gray')
    axs[0].set_title("Occupacy")
    _, _, width, start, end, _, _ = view_corridor
    rr, cc = line(target[0], target[1], start[0], start[1])  # Ray from POI to front
    rr_2, cc_2 = line(target[0], target[1], end[0], end[1])
    axs[0].plot(cc, rr, color="green", linewidth=1)  # Draw the ray
    axs[0].plot(cc_2, rr_2, color="green", linewidth=1)
    _, _, width, start, end, _, _ = push_corridor
    rr, cc = line(center[0], center[1], start[0], start[1])  # Ray from POI to front
    rr_2, cc_2 = line(center[0], center[1], end[0], end[1])
    axs[0].plot(cc, rr, color="red", linewidth=1)  # Draw the ray
    axs[0].plot(cc_2, rr_2, color="red", linewidth=1)
    for pt, end in samples:
        axs[0].scatter(end[0], end[1], c='blue')
        axs[0].scatter(pt[0], pt[1], c='red')
    axs[1].imshow(semantic, cmap='jet')
    axs[1].set_title("Semantic")
    _, _, width, start, end, _, _ = view_corridor
    rr, cc = line(target[0], target[1], start[0], start[1])
    rr_2, cc_2 = line(target[0], target[1], end[0], end[1])
    axs[1].plot(cc, rr, color="green", linewidth=1)
    axs[1].plot(cc_2, rr_2, color="green", linewidth=1)
    _, _, width, start, end, _, _ = push_corridor
    rr, cc = line(center[0], center[1], start[0], start[1])
    rr_2, cc_2 = line(center[0], center[1], end[0], end[1])
    axs[1].plot(cc, rr, color="red", linewidth=1)
    axs[1].plot(cc_2, rr_2, color="red", linewidth=1)
    for pt, end in samples:
        axs[1].scatter(end[0], end[1], c='blue')
        axs[1].scatter(pt[0], pt[1], c='red')
    plt.tight_layout()
    plt.show()

    #plt.imshow(semantic, cmap='gray')
    #plt.show()
    #plt.imshow(semantic, cmap='gray')
    #_, _, width, start, end, _, _ = view_corridor
    #rr, cc = line(target[0], target[1], start[0], start[1])  # Ray from POI to front
    #rr_2, cc_2 = line(target[0], target[1], end[0], end[1])
    #plt.plot(cc, rr, color="green", linewidth=1)  # Draw the ray
    #plt.plot(cc_2, rr_2, color="green", linewidth=1)#

    #mask_rgb = np.zeros((*mask.shape, 4))  # RGBA image
    #mask_rgb[mask == 1] = [1, 0, 0, 0.5]  # Red color with 50% opacity
    #plt.imshow(mask_rgb)  # Overlay on top
    #plt.show()

    #plt.imshow(semantic, cmap='gray')
    #_, _, width, start, end, _, _ = push_corridor
    #rr, cc = line(center[0], center[1], start[0], start[1])  # Ray from POI to front
    #rr_2, cc_2 = line(center[0], center[1], end[0], end[1])
    #plt.plot(cc, rr, color="red", linewidth=2)  # Draw the ray
    #plt.plot(cc_2, rr_2, color="red", linewidth=2)
    #for pt, end in samples:
    #    plt.scatter(end[0], end[1], c='blue')
    #    plt.scatter(pt[0], pt[1], c='red')
    #plt.show()
    #print("test")





