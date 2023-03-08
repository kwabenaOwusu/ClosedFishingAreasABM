#!/usr/bin/env python
# coding: utf-8

# ### importing necessary libraries 

import shutil, os
import numpy as np
import pandas as pd
import math
from math import atan2
import random
import statistics
import csv
from scipy.optimize import minimize

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib import rcParams, cycler
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,  zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, box, mapping
from shapely.ops import cascaded_union, unary_union, nearest_points
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
#import shapely.speedups

import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio import Affine # or from affine import Affine
from rasterio.plot import plotting_extent
from glob import glob 

import osmnx as ox
ox.settings.log_console=True
ox.__version__
import networkx as nx

#---------------------------------------------------------------------------------------------------------------------------------#

# ### read files (eez, eez_12nm, eez_24nm) and initialize parameters
def Param_Initialize():
    global eez, eez_12nm, eez_24nm, IFA, IFA_poly, Type_MPA,  Frac_MPA , Dist_MPA, Time_MPA
    global num_clusters, init_fish, point_deviation, rad_repulsion, rad_orientation, rad_attraction
    global move_fish, growth_prob, K, num_pirogue, catchability, time, time1, frac_coop, coop_effort, noncoop_effort
    global TOTAL_CATCH, TOTAL_FISH, CURRENT_CATCH, exploration_area, exploration_prob, num_days
    
    # reading shapefiles
    eez  = gpd.read_file(os.path.join(os.getcwd(), 'eez/eez.shp')) # Exclusive Economic Zone (200 nautical miles)
    eez_12nm  = gpd.read_file(os.path.join(os.getcwd(), 'eez_12nm/eez_12nm.shp')) # 12 nautical miles zones (territorial seas)
    eez_24nm  = gpd.read_file(os.path.join(os.getcwd(), 'eez_24nm/eez_24nm.shp')) # 24 nautical miles zones (contiguous zones)
    # eez  = gpd.read_file("/Users/kwabx/MPA_code/eez/eez.shp") # Exclusive Economic Zone (200 nautical miles)
    # eez_12nm  = gpd.read_file("/Users/kwabx/MPA_code/eez_12nm/eez_12nm.shp") # 12 nautical miles zones (territorial seas)
    # eez_24nm  = gpd.read_file("/Users/kwabx/MPA_code/eez_24nm/eez_24nm.shp") # 24 nautical miles zones (contiguous zones)
    
    # reproject crs 
    eez  = eez.to_crs(epsg=25000) 
    eez_12nm  = eez_12nm.to_crs(epsg=25000) 
    eez_24nm  = eez_24nm.to_crs(epsg=25000) 

    # inshore fishing area 
    IFA = gpd.overlay(eez_12nm, eez_24nm, how='union',keep_geom_type=True) # IFA = (territorial seas) + (contiguous zones) 
    IFA_poly=unary_union([eez_12nm.at[0,'geometry'],eez_24nm.at[0,'geometry']]) # the multipolygon equivalent of IFA
    
    # parameters
    Frac_MPA = 0.08   # fraction of the fishing ground to set as closed fishing area 
    # Type_MPA = 'individual_closure' # type of simulation (no_closure, full_closure, individual_closure, large_closure)
    # Dist_MPA = 0.000001     # distance between cosed fishing area expressed as a fraction of IFA area
    # Time_MPA = 15       # time to terminate closed fishing area

    # fish characteristics
    num_clusters = 5 # initialise number of fish clusters
    init_fish = 1000 # initial number of fishes
    point_deviation = 30000 # std deviation from cluster of fish
    rad_repulsion = 10000 # radius of repulsion zone (metres)
    rad_orientation = 20000 # radius of orientation zone (metres) 
    rad_attraction =  30000 # radius of attraction zone (metres)  
    move_fish = 63000 # speed of fish
    growth_prob =  0.3 # maximum intrinsic growth rate
    K = 100000 # carrying capacity of fishing ground
    
    # fisher characteristics
    num_pirogue = 100 # number of pirogues
    catchability = 0.8 # catchability of pirogues
    frac_coop = 1 #fraction of cooperators
    coop_effort = 1 # effort of cooperators (max fraction of fish to potentially harvest)
    noncoop_effort = 1 # effort of noncooperators (max fraction of fish to potentially harvest)
    exploration_area =  49 * ((0.001 * IFA.area.sum())**2)   # max area to explore randomly
    exploration_prob = 0.5 # probability to explore randomly
    
    time = 0 # time for updating
    time1 = [time]
    num_days = 5 # number of days to run simulation
    
    TOTAL_CATCH = [0]
    CURRENT_CATCH =[0]
    TOTAL_FISH = [init_fish]
    
Param_Initialize() 

#---------------------------------------------------------------------------------------------------------------------------------#

def Regular_Cell():
    global cell, cells, cells_eez_12nm, cells_eez_24nm
    
    xmin, ymin, xmax, ymax = IFA_poly.bounds # get the bounds of IFA

    #cell size 
    min_cell_area = 0.001 * IFA.area.sum() # in square meters / 1000000 
    cell_size = math.sqrt(min_cell_area) # length of square cell

    # create the cells in a loop
    polygons = []
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):

            # bounds of polygon
            x1 = x0+cell_size 
            y1 = y0+cell_size

            polygons.append(box(x0, y0, x1, y1))
        
    # form geodataframe 
    cell = gpd.GeoDataFrame(polygons,geometry='geometry',columns=['geometry'],crs=IFA.crs) 
    cells = gpd.overlay(IFA, cell, how='intersection') # only cells within the IFA
    cells['area'] = cells.apply(lambda row: row['geometry'].area, axis=1)
    
    cells_eez_12nm = gpd.overlay(eez_12nm, cell, how='intersection') # only cells within the ees_12nm
    cells_eez_12nm['area'] = cells_eez_12nm.apply(lambda row: row['geometry'].area, axis=1)
    cells_eez_24nm = gpd.overlay(eez_24nm, cell, how='intersection') # only cells within the ees_24nm
    cells_eez_24nm['area'] = cells_eez_24nm.apply(lambda row: row['geometry'].area, axis=1)
    

Regular_Cell() 

#---------------------------------------------------------------------------------------------------------------------------------#

### spawning hotspots within eez_12nm 
def MPA_Characteristics():
    global MPA, Area_MPA, mpa_area, neighbor_cells, MPA_cell_shapefile, MPA_cellx, MPA_ConvexHull, MPA_union, num_mpa 

    
    def mpa_characteristics():
        global MPA, Area_MPA, mpa_area, neighbor_cells, MPA_cell_shapefile, MPA_cellx,  MPA_ConvexHull,  MPA_union, cells_eez_12nm, num_mpa 
        
        MPA_start_cell_index=random.randrange(len(cells_eez_12nm)) # index of closed fishing area start cell
        MPA_cell =cells_eez_12nm.at[MPA_start_cell_index,'geometry'] # geometry of closed fishing area start cell
        MPA_cellx = cells_eez_12nm.at[MPA_start_cell_index,'geometry'] # just for computing nearest distance to it
        MPA_cell_shapefile = cells_eez_12nm.loc[MPA_start_cell_index : MPA_start_cell_index] # shapefile of closed fishing area start cell
        mpa_area = 0 # initialize closed fishing area
        ALL_neighbor_cells = []
        # MPA_cell_shapefile = cells.loc[cells['geometry'].contains(IFA_poly.centroid)]
        # MPA_cell = MPA_cell_shapefile.at[MPA_cell_shapefile.index.values[0],'geometry']
        # Area_MPA = Frac_MPA * (cells['area'].sum()) # total area of MPA as a fraction of IFA area

        while True:
            # print(len(cells)) 
            neighbor_cells = cells_eez_12nm.loc[ ((cells_eez_12nm['geometry']).distance(MPA_cell) < 5000) & (cells_eez_12nm['geometry'] != MPA_cell)                                        # & ((cells_eez_12nm.intersection(MPA_cell)).geom_type == "LineString") 
                                      ] # cells in neighborhood start mpa cell 
            neighbor_cells =  gpd.GeoDataFrame(pd.concat([MPA_cell_shapefile,neighbor_cells]),geometry='geometry', crs=IFA.crs)
            neighbor_cells['dist-start-cell'] = neighbor_cells.apply(lambda row: row['geometry'].distance(MPA_cellx), axis=1)
            neighbor_cells.sort_values(by = 'dist-start-cell')
           
            for idx, row in neighbor_cells.iterrows():
                if mpa_area  < Area_MPA :
                    mpa_area += row['area'] 
                    cells_eez_12nm.drop([idx], inplace=True) 
                else:
                    neighbor_cells.drop([idx], inplace=True)

            list_neighbor_cells = list(neighbor_cells['geometry']) # convert current neighbor cells to list
            ALL_neighbor_cells += list_neighbor_cells # contains all neighbor cells over time
            MPA_cell = unary_union(ALL_neighbor_cells) # unionize all neighbor cells over time
            MPA_cell_shapefile = gpd.GeoDataFrame(geometry='geometry',columns=['geometry'],crs=IFA.crs) # empty geodataframe

            if mpa_area >= Area_MPA:
                break

        MPA = gpd.GeoDataFrame(ALL_neighbor_cells,geometry='geometry',columns=['geometry'], crs=IFA.crs) 
        return MPA
    
    # mpa_characteristics()
    
    def MPA_Specifics():
        global MPA, Area_MPA, mpa_area, neighbor_cells, MPA_cell_shapefile, MPA_cellx,  MPA_ConvexHull,  MPA_union, cells_eez_12nm, num_mpa 

  
        num_mpa = 5 # nr. of hotspots
        Area_MPA = (Frac_MPA * (cells['area'].sum())) / num_mpa # area of each hostspot
        for i in range(num_mpa): 
            globals()[f'MPA{i+1}']  = mpa_characteristics() # dynamically name the variables
        MPA =  gpd.GeoDataFrame(pd.concat([MPA1,MPA2,MPA3,MPA4,MPA5]),geometry='geometry', crs=IFA.crs)
        ConvexHull = unary_union(MPA['geometry']).convex_hull
        MPA_union = gpd.GeoDataFrame([ConvexHull],geometry='geometry',columns=['geometry'],crs=IFA.crs)
        MPA_ConvexHull = gpd.overlay(eez_12nm, MPA_union, how='intersection',keep_geom_type=False) # only convex cells within the IFA

    MPA_Specifics()

    Regular_Cell() # reset cells 

MPA_Characteristics() 

#---------------------------------------------------------------------------------------------------------------------------------#

#### initialise fish (in clusters)
def Fish_Initialize() :
    global fish_geodata
    
    center =[] # set the centers of clustering
    xmin, ymin, xmax, ymax = IFA_poly.bounds # get the bounds of IFA
    fish_geodata = gpd.GeoDataFrame(geometry='geometry',columns=['geometry'],crs=IFA.crs) # empty geodataframe

    for cluster in range(num_clusters): # create center for clusters
        while True:   
            cluster_x = random.uniform(xmin, xmax)
            cluster_y = random.uniform(ymin, ymax)
            pnt = Point(cluster_x, cluster_y)
            if IFA_poly.contains(pnt):
                center.append((cluster_x,cluster_y))
                break


    for fish in range(init_fish): # create fishes around clusters
        if (fish < 0.2 * init_fish) :
            while True:   
                fish_x = random.gauss(center[0][0],point_deviation) # gaussian distribution around center
                fish_y = random.gauss(center[0][1],point_deviation)
                pnt = Point(fish_x, fish_y)
                if IFA_poly.contains(pnt):
                    fish_geodata.at[fish,'geometry'] = pnt
                    fish_geodata.at[fish,'x'] = fish_x
                    fish_geodata.at[fish,'y'] = fish_y
                    fish_geodata.at[fish,'agent type'] = 'fish'
                    break

        elif (0.20 * init_fish) <= fish < (0.4 * init_fish) :
             while True:   
                fish_x = random.gauss(center[1][0],point_deviation)
                fish_y = random.gauss(center[1][1],point_deviation)
                pnt = Point(fish_x, fish_y)
                if IFA_poly.contains(pnt):
                    fish_geodata.at[fish,'geometry'] = pnt
                    fish_geodata.at[fish,'x'] = fish_x
                    fish_geodata.at[fish,'y'] = fish_y
                    fish_geodata.at[fish,'agent type'] = 'fish'
                    break

        elif (0.4 * init_fish) <= fish < (0.6 * init_fish) :
            while True:   
                fish_x = random.gauss(center[2][0],point_deviation)
                fish_y = random.gauss(center[2][1],point_deviation)
                pnt = Point(fish_x, fish_y)
                if IFA_poly.contains(pnt):
                    fish_geodata.at[fish,'geometry'] = pnt
                    fish_geodata.at[fish,'x'] = fish_x
                    fish_geodata.at[fish,'y'] = fish_y
                    fish_geodata.at[fish,'agent type'] = 'fish'
                    break

        elif (0.6 * init_fish) <= fish < (0.8 * init_fish) :
            while True:   
                fish_x = random.gauss(center[3][0],point_deviation)
                fish_y = random.gauss(center[3][1],point_deviation)
                pnt = Point(fish_x, fish_y)
                if IFA_poly.contains(pnt):
                    fish_geodata.at[fish,'geometry'] = pnt
                    fish_geodata.at[fish,'x'] = fish_x
                    fish_geodata.at[fish,'y'] = fish_y
                    fish_geodata.at[fish,'agent type'] = 'fish'
                    break

        else :
            while True:   
                fish_x = random.gauss(center[4][0],point_deviation)
                fish_y = random.gauss(center[4][1],point_deviation)
                pnt = Point(fish_x, fish_y)
                if IFA_poly.contains(pnt):
                    fish_geodata.at[fish,'geometry'] = pnt
                    fish_geodata.at[fish,'x'] = fish_x
                    fish_geodata.at[fish,'y'] = fish_y
                    fish_geodata.at[fish,'agent type'] = 'fish'
                    break

Fish_Initialize() 

#---------------------------------------------------------------------------------------------------------------------------------#

# ### fish updating (movements and reproduction)
def Fish_Update() :
    global fish_geodata
    
    # randomly sample a geometry from the geodataframe
    focal_fish_index=random.randrange(len(fish_geodata)) # index of focal fish
    focal_fish=fish_geodata.at[focal_fish_index,'geometry'] 
    
    # nearest point from exterior to the focal fish
    ext=np.array(IFA_poly.geoms[1].exterior.coords) # coordinate of exterior
    ext_multipoint = MultiPoint(ext) 
    ext_nearest_geoms = nearest_points(focal_fish, ext_multipoint)
    ext_coords_x = ext_nearest_geoms[1].x  
    ext_coords_y = ext_nearest_geoms[1].y 
    
    # computing zones around fish
    repulsion_zone=focal_fish.buffer(rad_repulsion, cap_style = 1) # circle buffer the point
    orientation_zone=focal_fish.buffer(rad_orientation, cap_style = 1) # circle buffer the point
    attraction_zone=focal_fish.buffer(rad_attraction, cap_style = 1) # circle buffer the point

    # locate fish with the zones of focal fish
    repulsion = fish_geodata.loc[((fish_geodata['geometry']).within(repulsion_zone)) & ((fish_geodata['geometry']) != focal_fish)]
    orientation = fish_geodata.loc[((fish_geodata['geometry']).within(orientation_zone)) & ((fish_geodata['geometry']) != focal_fish)]
    attraction = fish_geodata.loc[((fish_geodata['geometry']).within(attraction_zone)) & ((fish_geodata['geometry']) != focal_fish)]
   
    # if fishes within repulsion zone, move away from the spot that would be the center of mass (midpoint) of all  fish within repulsion zone
    if len(repulsion) > 0: 
        repulsion_x = repulsion['x'].mean()
        repulsion_y = repulsion['y'].mean()
        theta = (math.atan2((repulsion_y - focal_fish.y), (repulsion_x - focal_fish.x)) + math.pi ) % (2 * math.pi) # if greater than  (2 * math.pi) then compute with a minus
        fish_geodata.at[focal_fish_index,'x'] +=  move_fish*math.cos(theta) 
        fish_geodata.at[focal_fish_index,'y'] +=  move_fish*math.sin(theta) 
        fish_geodata.at[focal_fish_index,'x'] = ext_coords_x if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'x']  # ( When fish-agent Outside a border of the IFA)
        fish_geodata.at[focal_fish_index,'y'] = ext_coords_y if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'y']  # ( When fish-agent Outside a border of the IFA)
        fish_geodata.at[focal_fish_index,'geometry'] =  Point(fish_geodata.at[focal_fish_index,'x'], fish_geodata.at[focal_fish_index,'y']) 

    # if fishes within parallel-orientation zone, change direction to match the average direction of all the other fish  within parallel-orientation zone     
    elif all([len(repulsion) == 0, len(orientation) > 0]):  
        theta = (orientation.apply(lambda row: math.atan2((focal_fish.y - row['y']),(focal_fish.x - row['x'])) , axis=1)).mean()
        fish_geodata.at[focal_fish_index,'x'] = math.cos(theta) 
        fish_geodata.at[focal_fish_index,'y'] = math.sin(theta) 
        fish_geodata.at[focal_fish_index,'x'] = ext_coords_x if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'x']  # ( When fish-agent Outside a border of the IFA)
        fish_geodata.at[focal_fish_index,'y'] = ext_coords_y if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'y']  #  ( When fish-agent Outside a border of the IFA)
        fish_geodata.at[focal_fish_index,'geometry'] =  Point(fish_geodata.at[focal_fish_index,'x'], fish_geodata.at[focal_fish_index,'y']) 

    # if fishes within only the attraction zone, head towards the middle (midpoint) of the fishes in zone of attraction.   
    elif all([len(repulsion) == 0, len(orientation) == 0, len(attraction) > 0]): 
        attraction_x = attraction['x'].mean()
        attraction_y = attraction['y'].mean()
        theta = (math.atan2((attraction_y - focal_fish.y), (attraction_x - focal_fish.x)) + math.pi ) % (2 * math.pi) # if greater than  (2 * math.pi) then compute with a minus
        fish_geodata.at[focal_fish_index,'x'] +=  move_fish*math.cos(theta) 
        fish_geodata.at[focal_fish_index,'y'] +=  move_fish*math.sin(theta) 
        fish_geodata.at[focal_fish_index,'x'] = ext_coords_x if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'x']  # ( When fish-agent Outside a border of the IFA) 
        fish_geodata.at[focal_fish_index,'y'] = ext_coords_y if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'y']  # ( When fish-agent Outside a border of the IFA)
        fish_geodata.at[focal_fish_index,'geometry'] =  Point(fish_geodata.at[focal_fish_index,'x'], fish_geodata.at[focal_fish_index,'y']) 

    # if no fishes in all the zone, move in a random direction 
    elif all([len(repulsion) == 0, len(orientation) == 0, len(attraction) == 0]):  
        theta = 2*math.pi*random.random()  
        fish_geodata.at[focal_fish_index,'x'] +=  move_fish*math.cos(theta) 
        fish_geodata.at[focal_fish_index,'y'] +=  move_fish*math.sin(theta)
        fish_geodata.at[focal_fish_index,'x'] = ext_coords_x if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'x']  # ( When fish-agent Outside a border of the IFA) 
        fish_geodata.at[focal_fish_index,'y'] = ext_coords_y if IFA_poly.contains(Point(fish_geodata.at[focal_fish_index,'x'],fish_geodata.at[focal_fish_index,'y'])) == False else fish_geodata.at[focal_fish_index,'y']  # ( When fish-agent Outside a border of the IFA)
        fish_geodata.at[focal_fish_index,'geometry'] =  Point(fish_geodata.at[focal_fish_index,'x'], fish_geodata.at[focal_fish_index,'y']) 

    # logistid reproduction
    if random.random() <   growth_prob * (1 - len(fish_geodata) / K):
        fish_geodata.loc[len(fish_geodata)] = [Point(focal_fish.x , focal_fish.y), focal_fish.x , focal_fish.y, 'fish' ]
        
            
Fish_Update() 

#---------------------------------------------------------------------------------------------------------------------------------#

#### pirogue tasks
#### spatial query function to find points within polygons
def intersect_using_spatial_index(source_gdf, intersecting_gdf):
    """
    Conduct spatial intersection using spatial index for candidates GeoDataFrame to make queries faster.
    Note, with this function, you can have multiple Polygons in the 'intersecting_gdf' and it will return all the points 
    intersect with ANY of those geometries.
    """
    source_sindex = source_gdf.sindex
    possible_matches_index = []
    
    # 'itertuples()' function is a faster version of 'iterrows()'
    for other in intersecting_gdf.itertuples():
        bounds = other.geometry.bounds
        c = list(source_sindex.intersection(bounds))
        possible_matches_index += c
    
    # Get unique candidates
    unique_candidate_matches = list(set(possible_matches_index))
    possible_matches = source_gdf.iloc[unique_candidate_matches]

    # Conduct the actual intersect
    result = possible_matches.loc[possible_matches.intersects(intersecting_gdf.unary_union)]
    return result

#---------------------------------------------------------------------------------------------------------------------------------#

#### Initialize pirogues
def Pirogue_Initialize() :
    global pirogue_geodata, cells_pirogue_movement

    xmin, ymin, xmax, ymax = IFA_poly.bounds # get the bounds of IFA
    pirogue_geodata = gpd.GeoDataFrame(geometry='geometry',columns=['geometry'],crs=IFA.crs) # empty geodataframe
    cells_pirogue_movement = gpd.overlay(cells, MPA_ConvexHull, how='difference') # cells a pirogue can move to - LARGE_CLOSURE
    # cells_pirogue_movement = gpd.overlay(cells, MPA, how='difference') # cells a pirogue can move to - INDIVIDUAL_CLOSURE
    # cells_pirogue_movement = gpd.overlay(cells, cells, how='intersection',keep_geom_type=True) # cells a pirogue can move to - NO_CLOSURE
    # cells_pirogue_movement = gpd.GeoDataFrame(geometry='geometry',columns=['geometry'],crs=IFA.crs) # cells a pirogue can move to - FULL_CLOSURE
   

    # setting the charactersitics of pirogues
    for pirogue in range(num_pirogue): 
        while True:   
            pirogue_x = random.uniform(xmin, xmax)
            pirogue_y = random.uniform(ymin, ymax)
            pnt = Point(pirogue_x,pirogue_y)
            if len(cells_pirogue_movement.loc[cells_pirogue_movement['geometry'].contains(pnt)]) > 0: 
                pirogue_geodata.at[pirogue,'geometry'] = pnt
                pirogue_geodata.at[pirogue,'x'] = pirogue_x
                pirogue_geodata.at[pirogue,'y'] = pirogue_y
                pirogue_geodata.at[pirogue,'agent type'] = 'pirogue'
                pirogue_geodata.at[pirogue,'catch'] = 0
                pirogue_geodata.at[pirogue,'current catch'] = 0
                pirogue_geodata.at[pirogue,'cooperative-trait'] = 'coop' if  pirogue <  int(frac_coop * num_pirogue) else  'non-coop'          # set their cooperative-trait
                pirogue_geodata.at[pirogue,'effort'] = coop_effort if  pirogue <  int(frac_coop * num_pirogue) else noncoop_effort          # set their cooperative-trait
                pirogue_geodata.at[pirogue,'time0'] = 0
                break
                
Pirogue_Initialize()

#---------------------------------------------------------------------------------------------------------------------------------#

# #### pirogue updating (movements and harvest)
def Pirogue_Update() :
    global pirogue_geodata, repulsion, focal_pirogue_cell ,focal_pirogue_cell_fishes, focal_pirogue_neighbor_cells
    global focal_pirogue_cell, focal_pirogue, exploration_poly_shapefile, union_neighborhood, main_exploration_poly_shapefile

    xmin, ymin, xmax, ymax = IFA_poly.bounds # get the bounds of IFA

    #randomly sample a geometry from the geodataframe
    focal_pirogue_index=random.randrange(len(pirogue_geodata)) # index of focal prirogue
    focal_pirogue=pirogue_geodata.loc[focal_pirogue_index:focal_pirogue_index]  # sample focal piorgue using the index
    
    # # # cell containing the focal_pirogue
    focal_pirogue_cell = cells_pirogue_movement.loc[cells_pirogue_movement['geometry'].contains(focal_pirogue.at[focal_pirogue_index,'geometry'])]
    
    # cells containing neighbors (moore neighborhood) of focal pirogue  
    focal_pirogue_neighbor_cells = cells_pirogue_movement.loc[ ((cells_pirogue_movement['geometry']).distance(focal_pirogue_cell.at[focal_pirogue_cell.index[0],'geometry']) == 0)  &  (cells_pirogue_movement['geometry'] != focal_pirogue_cell.at[focal_pirogue_cell.index[0],'geometry']) ]
   
    if len(focal_pirogue_neighbor_cells) > 0 : # if focal cell has neighbor cells
        # find all pirogues in the focal_pirogue_neighbor_cells
        focal_pirogue_neighbors=intersect_using_spatial_index(source_gdf=pirogue_geodata, intersecting_gdf=focal_pirogue_neighbor_cells)
    else:
        focal_pirogue_neighbors = gpd.GeoDataFrame(geometry='geometry',columns=['geometry'],crs=IFA.crs)
        
    
    # exploration area
    exploration_poly =   ((focal_pirogue_cell.at[focal_pirogue_cell.index[0],'geometry']).centroid).buffer(math.sqrt(exploration_area)/2 , cap_style = 3) # square polygon of area to explore (buffered at focal pirogue)
    exploration_poly_shapefile =  gpd.GeoDataFrame([exploration_poly],geometry='geometry',columns=['geometry'], crs=IFA.crs) # square polygon to shapefile
    exploration_poly_shapefile=gpd.overlay(cells_pirogue_movement, exploration_poly_shapefile, how='intersection',keep_geom_type=True) # take only the exploration area within pirogue movement area
    if len(focal_pirogue_neighbor_cells ) > 0:
        union_neighborhood = gpd.overlay(focal_pirogue_neighbor_cells, focal_pirogue_cell, how='union',keep_geom_type=True) # union of moore neighborhood (i.e. including focal pirogue cell) ,keep_geom_type=True   
    else:
        union_neighborhood = focal_pirogue_cell
    main_exploration_poly_shapefile = gpd.overlay(exploration_poly_shapefile, union_neighborhood ,how='difference',keep_geom_type=True) # take only exploration area excluding moore neighborhood
     
        
    # EEI algorithm    
    Xmin, Ymin, Xmax, Ymax = main_exploration_poly_shapefile.total_bounds
    if random.random() <  exploration_prob: # explore
        while True:   
            exploration_x = random.uniform(Xmin, Xmax)
            exploration_y = random.uniform(Ymin, Ymax)
            pnt = Point(exploration_x, exploration_y)
            pnt_shapefile =  gpd.GeoDataFrame([pnt],geometry='geometry',columns=['geometry'], crs=IFA.crs) 
            pnt_within_explore=intersect_using_spatial_index(source_gdf=pnt_shapefile, intersecting_gdf=main_exploration_poly_shapefile) # check if explore point is within main_exploration_poly_shapefile
            if len(pnt_within_explore) > 0 :
                pirogue_geodata.at[focal_pirogue_index,'geometry'] = pnt
                pirogue_geodata.at[focal_pirogue_index,'x'] = exploration_x 
                pirogue_geodata.at[focal_pirogue_index,'y'] = exploration_y
                break

    else : # imitate / exploit
        if len(focal_pirogue_neighbors) > 0 :
            idx_maxcatch_focal_pirogue_neighbor=focal_pirogue_neighbors['current catch'].idxmax() # index of focal_pirogue_neighbor with max current catch
            if (pirogue_geodata.at[idx_maxcatch_focal_pirogue_neighbor,'current catch']) > (pirogue_geodata.at[focal_pirogue_index,'current catch']):
                pirogue_geodata.at[focal_pirogue_index,'geometry'] = pirogue_geodata.at[idx_maxcatch_focal_pirogue_neighbor,'geometry']
                pirogue_geodata.at[focal_pirogue_index,'x'] =  pirogue_geodata.at[idx_maxcatch_focal_pirogue_neighbor,'x']
                pirogue_geodata.at[focal_pirogue_index,'y'] =   pirogue_geodata.at[idx_maxcatch_focal_pirogue_neighbor,'y']


    # harvest fishes from focal pirogue cell
    # focal_pirogue_cell_fishes=intersect_using_spatial_index(source_gdf=fish_geodata, intersecting_gdf=focal_pirogue_cell) # fishes in focal pirogue cell
    focal_pirogue_cell_fishes=intersect_using_spatial_index(source_gdf=fish_geodata, intersecting_gdf= union_neighborhood) # fishes in focal pirogue cell + neigboring cell
    focal_pirogue_catch = int(catchability * pirogue_geodata.at[focal_pirogue_index,'effort'] * len(focal_pirogue_cell_fishes)) # number of fishes to harvest
    pirogue_geodata.at[focal_pirogue_index,'current catch'] += focal_pirogue_catch # add harvest to current catch 
    pirogue_geodata.at[focal_pirogue_index,'catch'] += focal_pirogue_catch  # add harvest to total catch 
    fish_index_remove=list(focal_pirogue_cell_fishes.sample(n=focal_pirogue_catch).index.values) # index of fishes to remove
    fish_geodata.drop(fish_index_remove, inplace=True) # remove the fish
    fish_geodata.reset_index(drop=True, inplace=True)  # reset the index of geodataframe
        
    
Pirogue_Update() 

#---------------------------------------------------------------------------------------------------------------------------------#

### observing fishing ground
def Observe():
    
    fig, ax = plt.subplots(figsize=(16,8))
    minx1, miny1, maxx1, maxy1 = IFA['geometry'].total_bounds# ezz bounds
     
    # create a legend: we'll plot empty lists with the desired color, label, symbol
    for facecol, label, edgecol, symb, alph in [('white','Inshore Fishing Area','black','s', 1), 
                                      ('white','Spawning Hotspot','mediumblue','s', 1),
                                      ('red','Closed to Fishing ','red','s', 0.3),
                                      ('black','Fish Agent','black','.', 1) ,
                                      ('white','Fisher Agent','black','o', 1) ]:
        ax.scatter([], [], facecolor=facecol, s=100, label=label, alpha=alph, edgecolors=edgecol, marker=symb,linewidths=1.5 )
        ax.legend(facecolor="white", edgecolor="none",prop={"size":12}, loc=(0.7,0.02),ncol=1 ) #loc=(0.5,0.02)


    # plot IFA, cells, MPA, fish, pirogue
    IFA.plot(ax=ax,facecolor='white', edgecolor='none', lw=1,zorder=1,alpha=0.5)
    cells.plot(ax=ax,facecolor="none", edgecolor='black', lw=0.4,zorder=1, alpha=0.5)
    if len(MPA) > 0 :
            MPA.plot(ax=ax,facecolor='none', edgecolor='blue', lw=1.4,zorder=2, alpha=0.5) # Plot MPA
            # convex hull around mpa
            MPA_ConvexHull.plot(ax=ax,facecolor='red', edgecolor='none',lw=1.4, zorder=3,alpha=0.17)
    fish_geodata.plot(ax=ax,color='black', edgecolor='black' ,marker = '.',markersize=4,zorder=2)
    pirogue_geodata.plot(ax=ax,facecolor='none', edgecolor='black' ,marker = 'o', linewidth =0.5,markersize=10,zorder=2)
     
    ax.set_xlabel('X Coordinates (km)',fontsize=15)
    ax.set_ylabel('Y Coordinates (km)',fontsize=15)
    ax.set_title("Ghana's Inshore Fishing Area : spatial dynamics of fish and pirogues, time =" + str(int(time)),fontsize=15) # title
    
    # set axis limit as boundaries of Ghana
    ax.set_xlim(([minx1, maxx1]))
    ax.set_ylim(([miny1, maxy1])) 
    # ax.set_yticks([]) ; ax.set_xticks([]) 
    # ax.set_axis_off()
    
    # make axis number label not scientific notation
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        
    # # change axis from meters to kilometers
    # m2km = lambda x, _: f'{x/1000:g}'
    # ax.xaxis.set_major_formatter(m2km)
    # ax.yaxis.set_major_formatter(m2km)
    # change axis from meters to kilometers
    m2km = lambda x, _: f'{ round((x + abs(miny1)) / 1000 ,0) :g}'
    ax.yaxis.set_major_formatter(m2km)
    ax.xaxis.set_major_formatter(m2km)
    
    # make insert figure within the main plot
    axins = inset_axes(ax,width='40%', height='40%', bbox_to_anchor=(0.07, 0.3, 0.5, 0.7), bbox_transform=ax.transAxes, loc='upper left') #, borderpad=4 
    axins1 = inset_axes(ax,width='40%', height='40%', bbox_to_anchor=(0.37, 0.3, 0.5, 0.7), bbox_transform=ax.transAxes, loc='upper left') # (0.37, 0.3, 0.5, 0.7)
    # axins2 = inset_axes(ax,width='40%', height='40%', bbox_to_anchor=(0.49, 0.09, 0.5, 0.7), bbox_transform=ax.transAxes, loc='lower right') #(0.55, 0.3, 0.5, 0.7)
    axins.plot(time1,TOTAL_FISH, color='black',linewidth=2)
    axins1.plot(time1,CURRENT_CATCH, color='black',linewidth=2)
    
    # number = 100
    # cmap = plt.get_cmap('gnuplot')
    # colors = [cmap(i) for i in np.linspace(0, 1, number)]
    # for idx, row in pirogue_geodata.iterrows():
    #     # print(list(row[1:3]))
    #     axins2.plot(time1,list(row[8:]),linewidth=0.8, color=colors[idx])

    axins.set_xlabel('Time (days)',fontsize=12)
    axins.set_ylabel('Fish ($10^3$)',fontsize=12,color='black')
    axins1.set_xlabel('Time (days)',fontsize=12,color='black')
    axins1.set_ylabel('Catch ($10^3$)',fontsize=12,color='black')
    # axins2.set_xlabel('Time (days)',fontsize=12,color='black')
    # axins2.set_ylabel('Ind. catch ($10^3$)',fontsize=12,color='black')
    
    inset_axis = lambda x, _: f'{x/1000:g}' # setting in 10^3
    axins.yaxis.set_major_formatter(inset_axis)
    axins1.yaxis.set_major_formatter(inset_axis)
    # axins2.yaxis.set_major_formatter(inset_axis)
     
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85) 
    plt.savefig('day_%04d.png' %time, bbox_inches='tight', pad_inches=0.1, dpi=1000) 

Observe()

#---------------------------------------------------------------------------------------------------------------------------------#

### updating fish and pirogues asyncronously
def update_one_time():
    global time, time1
    
    time += 1  # update time
    
    # updating fish
    t = 0.
    while t < 1 :
        t += 1. / len(fish_geodata)
        Fish_Update()
    
    # updating pirogues
    t = 0.
    while t < 1 :
        t += 1. / len(pirogue_geodata)
        Pirogue_Update()
     
    # housekeeping the data
    time1.append(time) # update time
    TOTAL_FISH.append(len(fish_geodata)) # update total fishes
    TOTAL_CATCH.append(pirogue_geodata['catch'].sum()) # update total catch
    CURRENT_CATCH.append(pirogue_geodata['current catch'].sum()) # update total catch
    pirogue_geodata['time%d'% (time)] = pirogue_geodata.apply(lambda row: row['current catch'] , axis =1) # keep current catch at each time step
    pirogue_geodata.to_csv("final_pirogue_geodata.csv", header=True) # convert final pirogue geodata to csv (containing currents catches per time) 
    pirogue_geodata['current catch'] = pirogue_geodata.apply(lambda row: 0, axis =1) # reset current catch to zero
   

    csvfile = "updated_mpa_sim_data.csv"   # a csv-file output 
    header = ['time','number_fish','catch','current catch']
    main_data = [time1, TOTAL_FISH, TOTAL_CATCH, CURRENT_CATCH]
    with open(csvfile, "w") as output:
        writer = csv.writer(output) 
        writer.writerow(header)
        writer.writerows(zip(*main_data))  
          
#---------------------------------------------------------------------------------------------------------------------------------#

# ### simulate over a number of time steps
Param_Initialize() # read files and initialize all parameters
Regular_Cell() # set IFA into regular cells
MPA_Characteristics() # set the characteristics of MPA
Fish_Initialize() # initialize clusters of fishes
Pirogue_Initialize() # initialize the pirogues
Observe() # plot to observe

for j in range(1,5):  
    update_one_time()
    Observe()
os.system("ffmpeg -v quiet -r 5 -i day_%04d.png -vcodec mpeg4  -y -s:v 1920x1080 updated_mpa_sim_movie.mp4") # convert png files to a movie






