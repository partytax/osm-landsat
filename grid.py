def create(start_lon=-180, start_lat=90, increment=0.1, grid_path='data/grid/grid.npy', grid_meta_path='data/grid/grid_meta.json'):
    """
    Create a Numpy array to store coordinates and other information about square areas on the earth.

    Input starting longitude and latitude, the area size/increment in decimal degrees, and paths to save the Numpy array and a json metadata file.
    Return nothing.
    Results of function may be accessed at the specified grid path.
    """

    import numpy as np
    import json

    print('\nCREATING COORDINATE GRID')

    #calculate number of array rows/cols necessary to hold coordinate grid of desired precision
    rows = 2 * abs(int(start_lat / increment))
    cols = 2 * abs(int(start_lon / increment))

    #create empty array
    world = np.empty((rows, cols), dtype=object)

    #copy start longitude to variable for iterator
    lon = start_lon
    #lat is adjusted down one tile so coordinates represent tile's southwest corner
    lat = start_lat - increment

    #iterate through entire array and write coordinates and default values
    for row in range(world.shape[0]):
        lat_entry = round(lat,1)
        #calculate geographic area of one area in the array based on latitude
        geo_area = latitude_to_area(lat_entry+increment/2, increment)
        lat -= increment
        for tile in range(world.shape[1]):
            lon_entry = round(lon,1)
            lon += increment
            #write data to Numpy array
            world[row][tile] = {'origin':(lon_entry, lat_entry), 'tile_path':'', 'image_path':'', 'status':0, 'geo_area':geo_area, 'buildings':-1}
        #reset longitude at end of row
        lon = start_lon

    #save grid array to specified path
    np.save(grid_path, world)

    #save grid metadata to specified folder_end
    with open(grid_meta_path, 'w') as outfile:
        grid_meta = {
            'start_lon':start_lon,
            'start_lat':start_lat,
            'increment':increment
        }
        json.dump(grid_meta, outfile)
    print('COORDINATE GRID CREATED')


def label_image_area(grid_path='data/grid/grid.npy', grid_meta_path='data/grid/grid_meta.json', image_folder='data/reproject', image_index=-1):
    """
    If image_index is set to a positive number, label only tiles overlapping the file under that index in the image folder.
    If image_index is set to a negative number, iterate through all images in folder and label them.
    """

    import rasterio
    from raster import crop
    import numpy as np
    import os
    import json

    print('\nLABELING IMAGE AREA(S)')

    #load grid
    grid = np.load(grid_path, allow_pickle=True)

    #load grid metadata
    with open(grid_meta_path) as file:
        data = json.load(file)
        increment = data['increment']

    if image_index * -1 > 0:
        image_paths = [image_folder+'/'+file for file in os.listdir(path=image_folder)]
    else:
        image_path = image_paths[image_index]

    for image_path in image_paths:
        #load image to get its bounds
        with rasterio.open(image_path) as src:
            west = src.bounds[0]
            south = src.bounds[1]
            east = src.bounds[2]
            north = src.bounds[3]
        #iterate through all areas and label them status=1 if they're within the large image bounds
        for row in range(grid.shape[0]):
            for tile in range(grid.shape[1]):
                if grid[row][tile].get('status') == 0:
                    if grid[row][tile].get('origin')[0]>west and grid[row][tile].get('origin')[0]+increment<east:
                        if grid[row][tile].get('origin')[1]>south and grid[row][tile].get('origin')[1]+increment<north:
                            #flag area as containing image data
                            grid[row][tile]['status'] = 1
                            grid[row][tile]['image_path'] = image_path
    np.save(grid_path, grid)
    print('IMAGE AREA(S) LABELED')


def corner_to_bbox(sw_coords, scale):
    """
    Accepts lon, lat coordinate tuple for southwest corner of area and width/height scale of the area in decimal degrees, returning a polygon for rasterio masking.
    """
    west = sw_coords[0]
    south = sw_coords[1]
    east = sw_coords[0] + scale
    north = sw_coords[1] + scale
    return [
        {
            'type':'Polygon',
            'coordinates':
            [
                [
                    (west,south),
                    (east,south),
                    (east,north),
                    (west,north)
                ]
            ]
        }
    ]


def render(type='imshow', grid_path='data/grid/grid.npy', key='status', cmap='viridis', crop=True, margin=5, bins=25):
    import numpy as np
    import matplotlib.pyplot as plt

    print('\nRENDERING GRID')

    master_grid = np.load(grid_path, allow_pickle=True)

    #create temporary array to hold integers from particular attribute within world grid
    temp_grid = np.empty((master_grid.shape[0], master_grid.shape[1]))

    relevant_tiles = []
    relevant_rows = []

    for row in range(master_grid.shape[0]):
        for tile in range(master_grid.shape[1]):
            temp_grid[row][tile] = master_grid[row][tile].get(key, -1)
            if master_grid[row][tile].get(key, -1) > 0:
                relevant_tiles.append(tile)
                relevant_rows.append(row)

    #print(f'min_tile:{min_tile} max_tile:{max_tile} min_row:{min_row} max_row:{max_row}')
    if crop == True:
        min_tile = min(relevant_tiles) - margin
        max_tile = max(relevant_tiles) + margin
        min_row = min(relevant_rows) - margin
        max_row = max(relevant_rows) + margin
        temp_grid = temp_grid[min_row:max_row, min_tile:max_tile]
    if type == 'imshow':
        plt.title(f"render of '{key}' key from '{grid_path}'")
        plt.imshow(temp_grid, cmap=cmap)
    else:
        temp_grid = np.reshape(temp_grid, -1)
        temp_grid = temp_grid[temp_grid != -1]
        if type == 'hist':
            plt.title(f"histogram of '{key}' key from '{grid_path}'")
            plt.hist(temp_grid, bins=bins)
        if type == 'stats':
            print(f'median: {np.median(temp_grid)}')
            print(f'average: {np.average(temp_grid)}')
            print(f'standard deviation: {np.std(temp_grid)}')
            print(f'minimum: {np.amin(temp_grid)}')
            print(f'25th percentile: {np.percentile(temp_grid, 25)}')
            print(f'50th percentile: {np.percentile(temp_grid, 50)}')
            print(f'75th percentile: {np.percentile(temp_grid, 75)}')
            print(f'maximum: {np.amax(temp_grid)}')
    plt.show()
    print('GRID RENDERED')


def replace(key, find_vals, replace_val, grid_path='data/grid/grid.npy'):
    """
    For a given key, find all instances of a particular value and replace it with another across a geographic grid.
    """
    import numpy as np

    print('\nREPLACING KEYS IN GRID')

    if type(find_vals) == list:
        find_list = find_vals
    else:
        find_list = []
        find_list.append(find_vals)

    print(find_list)

    grid = np.load(grid_path, allow_pickle=True)
    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            for val in find_list:
                if val == 'True':
                    grid[row][tile][key] = replace_val
                elif grid[row][tile].get(key) == val:
                    grid[row][tile][key] = replace_val
    np.save(grid_path, grid)
    print(f"WROTE {key} FROM {find_list}'s TO {replace_val}'s")


def latitude_to_area(latitude, increment, earth_radius=6371.0088):
    """
    Calculate kilometer area of square plot of land based on its center latitude and width/height (increment).
    """
    import math
    latitude_line_radius = math.cos(math.radians(latitude)) * earth_radius
    latitude_line_circumference = math.pi * latitude_line_radius * 2
    areas_around_earth = 360 / increment
    area_width = latitude_line_circumference / areas_around_earth
    return area_width**2


def copy(key, from_grid_path='data/grid/old_grid.npy', to_grid_path='data/grid/grid.npy'):
    """
    Copy all of a key's values from one geographic grid to another.
    """
    print('\nCOPYING KEY DATA BETWEEN GRIDS')
    import numpy as np
    from_grid = np.load(from_grid_path, allow_pickle=True)
    print('source grid loaded')
    to_grid = np.load(to_grid_path, allow_pickle=True)
    print('destination grid loaded')
    for row in range(from_grid.shape[0]):
        for tile in range(from_grid.shape[1]):
            to_grid[row][tile][key] = from_grid[row][tile][key]
    np.save(to_grid_path, to_grid)
    print('DATA COPIED')


def backup(grid_path='data/grid/grid.npy', grid_backup_path='data/grid/grid.npy.bak'):
    import numpy as np
    import shutil
    print('\nSTARTING GRID BACKUP')
    try:
        np.load(grid_path, allow_pickle=True)
    except:
        print('grid invalid\nstopping backup')
        print('BACKUP FAILED')
    else:
        print('grid valid\nbacking up')
        shutil.copy2(grid_path, grid_backup_path)
        print('GRID BACKUP COMPLETED')
