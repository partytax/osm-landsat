"""
Runs sequence to process Landsat imagery, gather building counts from OpenStreetMap, and model the relationship between them with TensorFlow.

AREA STATUS CODES
0: area unprocessed or unimportant for this analysis
1: imagery fully overlaps this area
2: tile extracted for area
3: >=95% of area overlaps valid imagery (not black pixels)
4 tile for area resized and placed in 'prepared' folder
"""

import raster
import grid
import osm
import learn

#reproject all necessary Landsat images
##raster.reproject()

#create new coordinate grid Numpy array
##grid.create()
#if extant, backup grid Numpy array to guard against corruption during large operations
grid.backup()
#mark status=1 on all areas in grid that overlap reprojected Landsat images
##grid.label_image_area()

#cut images into tiles, place them in a new folder, and write area status to '2'
##raster.tile_image()

#check if <5% of tile is blank, and if so, mark area status to '3'
##raster.tile_validate()
#resize tiles corresponding to areas where status=3 and mark status=4
##raster.tile_resize(scale=128)

#query the Overpass OpenStreetMap API for the number of buildings in each area where status=4
##osm.count_buildings(sleep=1)
#calculate buildings per square kilometer based on latitude
##osm.adjust_building_count()
#sort area building counts into approximately-equally-sized buckets for modeling
##osm.categorize_building_counts()

#extract data from coordinate grid array, process, and place in train/validate/test arrays
##learn.prepare(scale=128)
#define TensorFlow model, compile, and fit
learn.model(scale=128, epochs=1)
#load a model and display plots
##learn.stats()

#UTILITIES
#grid.replace('status', True, 0)
#grid.copy('buildings')
#grid.render(crop=True, key='building_category', type='imshow')
#raster.render()
