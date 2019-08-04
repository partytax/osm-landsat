import numpy as np


def count_buildings(grid_path='data/grid/grid.npy', grid_meta_path='data/grid/grid_meta.json',sleep=10):
    import time
    import overpy
    import random
    import json

    print('\nCOUNTING BUILDINGS')

    #load grid
    grid = np.load(grid_path, allow_pickle=True)

    #load grid metadata
    with open(grid_meta_path) as file:
        data = json.load(file)
        increment = data.get('increment')

    api = overpy.Overpass()

    tiles_processed = 0

    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            if grid[row][tile].get('status') == 4 and grid[row][tile].get('buildings')*-1 > 0:
                west = grid[row][tile]['origin'][0]
                south = grid[row][tile]['origin'][1]
                east = west + increment
                north = south + increment

                success = False
                sleep_time = sleep

                while success == False:
                    try:
                        #overpass api takes lat, lon, lat, lon format
                        result = api.query(f"""
                            (
                                way["building"]({south},{west},{north},{east});
                            );
                            out body;
                        """)
                        print('query completed')
                        time.sleep(sleep_time)
                        tiles_processed += 1
                        success = True
                    except overpy.exception.OverpassTooManyRequests:
                        print('OverpassTooManyRequests')
                        sleep_time *= 2
                        print(f'sleeping for {sleep_time} seconds')
                        time.sleep(sleep_time)
                        success = False
                    except overpy.exception.OverpassGatewayTimeout:
                        print('OverpassGatewayTimeout')
                        sleep_time *= 5
                        print(f'sleeping for {sleep_time} seconds')
                        time.sleep(sleep_time)
                        success = False
                    except overpy.exception:
                        print('Other Overpass Error')
                        sleep_time *= 1.5
                        print(f'sleeping for {sleep_time} seconds')
                        time.sleep(sleep_time)
                        success = False
                    else:
                        building_count = len(result.ways)
                        grid[row][tile]['buildings'] = building_count
                        print(f'{building_count} buildings counted')
                    finally:
                        print(f'{tiles_processed} tiles processed')
                print('')
    np.save(grid_path, grid)
    print('BUILDINGS COUNTED')


def adjust_building_count(grid_path='data/grid/grid.npy'):
    """Calculate and record number of buildings per square kilometer based on geographic size of given map area."""

    print('\nCALCULATING ADJUSTED BUILDING COUNT')

    grid = np.load(grid_path, allow_pickle=True)

    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            if grid[row][tile].get('buildings')*-1 <= 0:
                grid[row][tile]['buildings_per_square_km'] = grid[row][tile].get('buildings') / grid[row][tile].get('geo_area')

    np.save(grid_path, grid)
    print('ADJUSTED BUILDING COUNT CALCULATED')


def categorize_building_counts(grid_path='data/grid/grid.npy', dividers=[(-1,20),(20,65),(65,135),(135,100000)]):
    """
    Sort areas into buckets by number of buildings per square kilometer.
    Label areas with integers according to their buckets.
    """

    print('\nCATEGORIZING BUILDING COUNTS')

    grid = np.load(grid_path, allow_pickle=True)

    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            if grid[row][tile].get('buildings_per_square_km', -1) >= 0:
                building_count = grid[row][tile].get('buildings_per_square_km')
                for index, divider in enumerate(dividers):
                    if building_count > divider[0] and building_count <= divider[1]:
                        grid[row][tile]['building_category'] = index
    np.save(grid_path, grid)
    print('BUILDING COUNTS CATEGORIZED')
