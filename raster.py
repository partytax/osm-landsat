def reproject(source_folder='data/source', reproject_folder='data/reproject', projection='EPSG:4326', image_index=-1):
    """
    Reproject GeoTIF to a given EPSG coordinate system.
    Load file from given folder and save to destination folder.
    Set image index to desired file number in image folder.
    Set it to a negative number to reproject all available images.
    """

    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import os
    from raster import clean_filename

    print('\nREPROJECTING IMAGE(S)')

    #create list of images that still need reprojected
    need_reprojected = [item for item in os.listdir(path=source_folder) if item not in os.listdir(path=reproject_folder)]

    if image_index < 0:
        source_paths = [source_folder + '/' + file for file in need_reprojected]
    else:
        source_paths = [source_folder + '/' + need_reprojected[image_index]]

    for path in source_paths:
        print(f'reprojecting {path}')
        with rasterio.open(path) as src:
            transform, width, height = calculate_default_transform(src.crs, projection, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': projection,
                'transform': transform,
                'width': width,
                'height': height
            })

            reproject_path = reproject_folder + '/' + clean_filename(path, keep_ext=True)
            print(f'reproject path: {reproject_path}')

            with rasterio.open(reproject_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=projection,
                        resampling=Resampling.nearest)
    print('IMAGE(S) REPROJECTED')


def crop(image_path, tile_folder, rasterio_bounds):
    """Cut a square area out of an image based on geographic bounds in Rasterio format."""

    import rasterio
    import rasterio.mask

    with rasterio.open(image_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, rasterio_bounds, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
        lon = str(round(rasterio_bounds[0]['coordinates'][0][0][0], 4))
        lat = str(round(rasterio_bounds[0]['coordinates'][0][0][1], 4))

        folder_end = max([pos for pos, char in enumerate(image_path) if char == '/'])
        #strip out folder part of input filepath, and remove file extension
        filename = image_path[folder_end:-4]

        tile_path = tile_folder + '/' + filename[:-4] + '_LON' + lon + '_LAT' + lat + '.TIF'
        with rasterio.open(tile_path, "w", **out_meta) as dest:
            dest.write(out_image)
    return tile_path


def tile_image(grid_path='data/grid/grid.npy', grid_meta_path='data/grid/grid_meta.json', image_folder='data/reproject', tile_folder='data/tile', image_index=-1):
    """
    Slice images into tiles based on geographic coordinate grid.
    Set image index to desired file number in image folder.
    Set it to a negative number to tile all available images.
    """

    import numpy as np
    import os
    import json
    from grid import corner_to_bbox

    print('\nTILING IMAGE')

    grid = np.load(grid_path, allow_pickle=True)

    with open(grid_meta_path) as file:
        data = json.load(file)
        increment = data['increment']

    if image_index < 0:
        image_paths = [image_folder + '/' + path for path in os.listdir(path=image_folder)]
    else:
        image_paths = [image_folder + '/' + os.listdir(path=image_folder)[image_index]]

    for image_path in image_paths:
        print(image_path)
        for row in range(grid.shape[0]):
            for tile in range(grid.shape[1]):
                #before a tile is cut out, check to see if the area status is 1 and the overlapping satellite image is currently loaded
                if grid[row][tile]['status'] == 1 and grid[row][tile]['image_path']==image_path:
                    tile_lon = grid[row][tile]['origin'][0]
                    tile_lat = grid[row][tile]['origin'][1]
                    bbox = corner_to_bbox((tile_lon,tile_lat), increment)
                    grid[row][tile]['tile_path'] = crop(image_path, tile_folder, bbox)
                    grid[row][tile]['status'] = 2
    np.save(grid_path, grid)
    print('IMAGE TILED')


def tile_validate(grid_path='data/grid/grid.npy', grid_meta_path='data/grid/grid_meta.json', threshold=.05):
    """Set area status to 3 if image tile has fewer zero pixels than the threshold."""

    import numpy as np
    import rasterio
    import os
    import json

    print('\nVALIDATING TILES')

    grid = np.load(grid_path, allow_pickle=True)

    #with open(grid_meta_path) as file:
    #    data = json.load(file)
    #    increment = data['increment']

    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            if grid[row][tile].get('status') == 2:
                tile_path = grid[row][tile].get('tile_path')
                tile_array = rasterio.open(tile_path).read(1)
                blank_pixels = (tile_array == 0).sum()
                total_pixels = tile_array.shape[0] * tile_array.shape[1]
                blank_ratio = blank_pixels/total_pixels
                if blank_ratio < threshold:
                    grid[row][tile]['status'] = 3
    np.save(grid_path, grid)
    print('TILES VALIDATED')


def tile_resize(prepared_folder='data/prepared', grid_path='data/grid/grid.npy', scale=64, image_index=-1):
    """
    Query grid Numpy array for areas with status of '3'.
    Load area tile image based on path stored in grid
    Resize image based on scale argument and save to 'prepared' folder.
    """

    import os
    from PIL import Image
    from raster import clean_filename
    import numpy as np

    print('\nRESIZING TILES')

    grid = np.load(grid_path, allow_pickle=True)

    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            if grid[row][tile].get('status') == 3:
                tile_path = grid[row][tile].get('tile_path')
                tile_img = Image.open(tile_path)
                prepared_img = tile_img.resize((scale,scale))
                prepared_path = prepared_folder + '/' + clean_filename(tile_path)
                prepared_img.save(prepared_path)
                grid[row][tile]['prepared_path'] = prepared_path
                grid[row][tile]['status'] = 4
    np.save(grid_path, grid)
    print('TILES RESIZED')


def clean_filename(path, keep_ext=True, ext_len=3):
    """Remove folder and extension from file path."""

    folder_end = max([pos for pos, char in enumerate(path) if char == '/']) + 1
    if keep_ext == False:
        ext_slice = (ext_len + 1) * -1
        filename = path[folder_end:ext_slice]
    else:
        filename = path[folder_end:]
    return filename


def render(image_path, cmap='viridis'):
    """Render the image at the given path."""

    from matplotlib import pyplot as plt
    import rasterio

    print('\nRENDERING IMAGE')

    plt.title(f"render of '{image_path}'")
    plt.imshow(rasterio.open(image_path).read(1), cmap=cmap)
    print('IMAGE RENDERED')
