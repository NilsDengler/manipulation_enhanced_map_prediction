import skgeom as sg
from matplotlib import pyplot as plt
import numpy as np
from skgeom import minkowski
from shapely.geometry import Point, Polygon, box, MultiPoint
from shapely.geometry.polygon import orient
from shapely.affinity import rotate,translate
import shapely
import seaborn as sns


def create_box(h,w,cx,cy,theta):
    """
    Theta is in degrees
    """
    theta = theta/180*np.pi
    vertices = []
    for addx,addy in zip([-w/2,w/2,w/2,-w/2],[-h/2,-h/2,h/2,h/2]):
        vertices.append(np.array([addx,addy]))
    vertices = np.array(vertices)
    rotmatrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    rot_vertices = (rotmatrix@vertices.transpose()).transpose()
    c = np.array([cx,cy])
    final_vertices = c+rot_vertices
    tmp = []
    for vertex in final_vertices:
        tmp.append(sg.Point2(*vertex))
    final_poly = sg.Polygon(tmp)

    return final_poly


def get_skbox_from_shapely(tmp):
    coords = np.asarray(tmp.exterior.coords)[:-1]
    vertices = []
    for i in coords:
        vertices.append(sg.Point2(*i))
    return sg.Polygon(vertices)
        

def get_circle_radius(tmp):
    coords = np.asarray(tmp.boundary.coords)
    centroid = np.asarray(tmp.centroid.xy).reshape(-1,2)
    radius = np.linalg.norm(coords-centroid,axis = 1).mean()
    return radius


def create_circle(cx,cy,radius,num_vertices = 20):
    vertices = []

    for i in range(num_vertices):
        vertices.append(sg.Point2(cx+radius*np.cos(2*np.pi*i/num_vertices),cy+radius*np.sin(2*np.pi*i/num_vertices)))

    circle = sg.Polygon(vertices)
    return circle


def get_mk_sum(a,b):
    
    if((np.array(a.boundary.coords).shape[0]> 10)):
        radius = get_circle_radius(a)
        mk_sum = b.buffer(radius)
    elif((np.array(b.boundary.coords).shape[0]> 10)):
        radius = get_circle_radius(b)
        mk_sum = a.buffer(radius)
    else:
        box1 = get_skbox_from_shapely(a)
        box2 = get_skbox_from_shapely(b)
        mk_sum = minkowski.minkowski_sum(box1,box2)
        coords = np.asarray(mk_sum.outer_boundary().coords)
        mk_sum = Polygon(coords.astype(list))
    return mk_sum


def get_original_valid_placing_area(ws,b):
    if((np.array(b.boundary.coords).shape[0]> 10)):
        radius = get_circle_radius(b)
        valid_placing_area = ws.buffer(-radius)
    else:
        xmin,ymin,xmax,ymax = b.bounds
        delta_x = (xmax-xmin)/2
        delta_y = (ymax-ymin)/2
        ws_coords = np.asarray(ws.exterior.coords)[:-1]
        # print('before',ws_coords)

        xs = get_shrunken_coords(ws_coords,0,delta_x)
        ys = get_shrunken_coords(ws_coords,1,delta_y)
        verts = np.concatenate([xs.reshape(-1,1),ys.reshape(-1,1)],axis = 1)
        # print('after',verts)
        valid_placing_area = orient(Polygon(verts.tolist()))
    return valid_placing_area


def get_shrunken_coords(coords,axis,delta):
    vals = coords[:,axis]
    valmin = vals.min()
    valmax = vals.max()
    vals[np.isclose(vals,valmin)] += delta
    vals[np.isclose(vals,valmax)] -= delta
    return vals


def sample_point_with_alignment(odds_map,all_valid_points,workspace,resolution,placed_objects,scale = 5,buffer = 1.5):
    log_scale = np.log(scale)
    coords = [p.xy for p in all_valid_points.geoms]
    coords = np.array(coords).reshape(-1,2)
    indices = grid_to_index(coords,workspace,resolution)
    x_index = indices[:,0]
    y_index = indices[:,1]
    valid_odds_map = odds_map[y_index,x_index].copy()
    xs = coords[:,0]
    all_valid_points.geoms
    for placed_object in placed_objects:
        cx,radius = placed_object.get_row_alignment_shape()
    #     alignment = placed_object.alignment
        alignment = placed_object.alignment
        # line.distance(candidate_points)
        dist = np.abs(xs-cx)
        max_dist = buffer*radius
        within_radius = dist <= max_dist
        relevant_dists = dist[within_radius]
        odds_adjustment = np.exp(log_scale*(1-2*relevant_dists/max_dist))
        odds_adjustment = np.clip(odds_adjustment,0.1,scale)
        valid_odds_map[within_radius] = alignment*valid_odds_map[within_radius]*odds_adjustment + (1-alignment)*valid_odds_map[within_radius]
    valid_probs_map = valid_odds_map/valid_odds_map.sum()
    chosen_index = np.random.choice(np.arange(0,len(all_valid_points.geoms)),1,p = valid_probs_map)[0]
    chosen_point = all_valid_points.geoms[chosen_index]
    odds_map[y_index,x_index] = valid_odds_map
    probs_map = odds_map/odds_map.sum()
    return chosen_point,probs_map


def grid_to_index(coords,ws,resolution):
    xmin,ymin,xmax,ymax = ws.bounds
    indices = np.round((coords-np.array([xmin,ymin]))/resolution).astype(int)
    return indices


class PlacedObject:
    def __init__(self, shape, radius_of_influence, conditional_prob, object_class, alignment, angle, name):
        """This class is used to define all the necessary details of what a shape placed in the shelf looks like.
           It also defines a few useful utility functions for their placement

        Args:
            shape (shapely.geometry): the geometry of the placed object as a shapely geometry (either a circle or a polygon)
            radius_of_influence (float): The distance, in meters, that defines the vicinity in which this placed object still influences the class of its neighboring objects
            conditional_prob ([floats] - n-dimensional categorical distribution): The conditional probability of how this object influences the placement of new neighbors around it
            object_class (int): integer that describes the class to which this object belongs.
            alignment (float): How strongly this object "forces" future objects to be placed in an organized row in the shelves (i.e., how organized the shelf is around this object)
            angle (float): Angle of placement of the object's instance in the world, in degrees - gets converted to radians for world placement
        """
        self.shape = shape
        self.radius_of_influence = radius_of_influence
        self.conditional_prob = np.array(conditional_prob)
        self.object_class = object_class
        self.alignment = np.clip(alignment, 0.00, 0.99)
        self.bounding_radius = shapely.minimum_bounding_radius(shape)
        self.adjust_conditional_prob_to_alignment()
        self.angle = np.deg2rad(angle)
        self.angle = angle
        self.name = name


    def get_area(self):
        return self.shape.area


    def get_radius_of_influence_shape(self):
        return Point(self.shape.centroid).buffer(self.radius_of_influence)


    def get_row_alignment_shape(self):
        cx,cy = self.shape.centroid.xy
        cx = cx[0]
        return cx,self.bounding_radius


    def adjust_conditional_prob_to_alignment(self):
        tmp = np.zeros_like(self.conditional_prob)
        tmp[self.object_class] = 1
        self.conditional_prob = self.conditional_prob*(1-self.alignment) + self.alignment*tmp


class CircularObjects:
    """This class defines a valid placeable circular item"""
    def __init__(self,radius):
        self.original_radius = radius
        self.radius = radius

    def update_dimensions(self, dimensions):
        self.radius = dimensions[0]

    def reset_dimensions(self):
        self.radius = self.original_radius

    def get_generic_shape(self,angle):
        return orient(Point(0,0).buffer(self.radius))

    def place(self,x,y,angle):
        return orient(Point(x,y).buffer(self.radius))


class BoxObjects:
    """This class defines a valid placeable box-like objects"""
    def __init__(self,height,width):
        self.height = height
        self.width = width
        self.original_height = height
        self.original_width = width

    def update_dimensions(self, dimensions):
        self.height = dimensions[0]
        self.width = dimensions[1]

    def reset_dimensions(self):
        self.height = self.original_height
        self.width = self.original_width

    def get_generic_shape(self,angle):
        """angle is in degrees"""
        return orient(rotate(box(0,0,self.width,self.height,ccw = True),angle))

    def place(self,x,y,angle):
        return translate(orient(rotate(box(-self.width/2,-self.height/2,self.width/2,self.height/2),angle)),x,y)


def create_placed_object(shape,obj_property_dict,object_class,n_classes,angle, name):
    radius_of_influence = obj_property_dict['radius_of_influence']
    conditional_prob = obj_property_dict.get('conditional_prob',None)
    if(conditional_prob is None):
        affinities = obj_property_dict.get('affinities',None)
        self_affinity = obj_property_dict.get('self_affinity',None)
        # here we define 
        if(affinities is None):
            raise KeyError('Missing affinity key in problem definition of class {}'.format(object_class))

        elif(self_affinity is None):
            raise KeyError('Missing self_affinity in problem definition of class {}'.format(object_class))
        conditional_probs = np.ones(n_classes)
        #double probability of affinities
        conditional_probs[affinities]*=2
        self_prob = np.zeros(n_classes)
        self_prob[object_class] = 1
        conditional_probs = conditional_probs/conditional_probs.sum()
        #enforce some self affinity
        conditional_prob = (1-self_affinity)*conditional_probs + self_affinity*self_prob
    
        
    alignment = obj_property_dict.get('alignment',0)
    object_class = object_class
    return PlacedObject(shape, radius_of_influence, conditional_prob, object_class, alignment, angle, name)


def get_placement_grid(ws,resolution):
    workspace_coords = np.asarray(ws.exterior.coords)
    coords_max = workspace_coords.max(axis = 0)
    coords_min = workspace_coords.min(axis = 0)
    x_span = np.arange(coords_min[0],coords_max[0],resolution)
    y_span = np.arange(coords_min[1],coords_max[1],resolution)
    mg = np.meshgrid(x_span,y_span)
    new_mg = np.concatenate((mg[0].reshape(-1, 1), mg[1].reshape(-1, 1)), axis=1)
    candidate_points = MultiPoint(new_mg)
    return candidate_points, x_span, y_span, mg


def get_starting_probabilities(x_span,y_span,n_classes):
    probs = np.zeros((x_span.shape[0],y_span.shape[0],n_classes))
    probs[:,:,:] = 1/n_classes
    return probs


def get_starting_occupancies(x_span,y_span):
    probs = np.ones((x_span.shape[0],y_span.shape[0]))
    return probs


def display_arrangements(placed_objects,n_classes):
    colors = sns.color_palette("husl", n_classes)
    if(len(placed_objects) >1):
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        for placed_object in placed_objects:    
            xs, ys = placed_object.shape.exterior.xy  
            color = colors[placed_object.object_class]
            axs.fill(xs, ys, alpha=0.5, fc=color, ec='none')
        plt.xlim(-0.4, 0.4)
        plt.ylim(0.6, 1.1)
        plt.show()