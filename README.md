Modeling the Relationship between Landsat Imagery and OpenStreetMap Data

To use this library, begin by creating a directory called 'data'. Inside this directory, create the following folders:

'grid'
'source'
'reproject'
'tiles'
'prepared'
'train'
'validate'
'test'

Download several GeoTIF files from https://glovis.usgs.gov/ and place them in the 'source' folder.

Open the file 'process.py'. Be sure that all functions, with the exception of utilities, are uncommented. 

After the initial run, be careful re-running grid.create() as it will overwrite stored data.

On later runs, check function docstrings to see which functions are necessary to run again. 
