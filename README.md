# Raster Extraction Functions
This Python script provides functions to process raster files in chunks and extract values at specified coordinates using multiprocessing. The script is designed to handle large raster datasets efficiently by dividing the workload across multiple CPU cores.

## NOTE
Note: GDAL cannot be installed via pip on Windows. Download the wheel for your python version here: 

https://github.com/cgohlke/geospatial-wheels/releases

Then install with: pip install "path/to/.whl"

## Usage
1. Create a virtual environment with `pythom -m venv 'C:\path\to\cloned\repo\venv'`
2. Activate the virtual environment with `C:\path\to\cloned\repo\venv\scripts\activate`
3. Install using ```pip install raster-extraction-tools```
4. Install GDAL (For Windows, see the note above)
4. Copy the code from example_run.py or from Example down below
5. Carefully change the parameters keeping in mind the commented instructions (all rasters should be in 1 directory with no subdirectories)
6. Save and run the python file

### Example
```python
import raster_extraction_functions as ref

if __name__ == "__main__":
    ref.extract_values(
        input_csv="path/to/csv.csv",  # file with at least 2 coordinate columns called "X" and "Y".
        raster_folder='path/to/raster/folder',  # directory where rasters to be extracted are saved, does not read subdirs.
        output_csv='path/to/output.csv',  # output file to be created.
        in_crs='EPSG:28992',  # Coordinate Reference System of input csv.
        raster_crs='EPSG:3035',  # Coordinate Reference System of rasters to be used (EPSG:3035 in case of EXPANSE rasters).
        sep=';',  # column separator in input file, output will always be semi-colon.
        decimal=',',  # decimal separator in input file.
        writemethod='concat',  # method to write output to csv, "concat" is fast but memory-intensive, "rows" is slow but requires no extra memory.
    )
```
## Requirements
- GDAL
- NumPy
- Pandas
- PyProj

## Installation
Make sure you have the required libraries installed. You can install them using pip:

```
pip install numpy pandas pyproj 
```

## Functions
### `extract_values`
Main function to orchestrate the multiprocessing of raster files for extracting values at specified coordinates.

**Parameters:**
- `input_csv (str)`: Path to the input CSV file containing coordinates.
- `raster_folder (str)`: Directory containing the raster files.
- `output_csv (str)`: Path to the output CSV file where results will be saved.
- `in_crs (str or int)`: Coordinate reference system of the input coordinates.
- `raster_crs (str or int)`: Coordinate reference system of the raster files.
- `sep (str, optional)`: Delimiter to use in the input CSV file. Default is ';'.
- `decimal (str, optional)`: Character to recognize as decimal point in the input CSV file. Default is '.'.
- `writemethod (str, optional)`: Method to write output CSV ('concat' or 'rows'). Default is 'concat'.

**Returns:**
- `None`

### `process_raster`
Processes a single raster file in chunks and extracts values at specified coordinates.

**Parameters:**
- `x_coords (numpy.ndarray)`: Array of x coordinates (longitude) for points of interest.
- `y_coords (numpy.ndarray)`: Array of y coordinates (latitude) for points of interest.
- `raster_folder (str)`: Directory containing the raster files.
- `raster_file (str)`: Name of the raster file to process.
- `buffer_size (int, optional)`: Buffer size around each coordinate to consider. Default is 0.

**Returns:**
- `tuple`: (column_name, values) where column_name is the raster file name without extension, and values is a numpy array of extracted values at the specified coordinates.

### Helper functions
### `calculate_chunk_size`

Helper function. Calculates the chunk size for reading the raster based on the desired maximum memory footprint.
**Parameters:**
- `dataset (gdal.Dataset)`: The GDAL dataset representing the raster.
- `max_chunk_memory (int, optional)`: Maximum memory (in bytes) to be used for a chunk. Default is 500 MB.

**Returns:**
- `tuple`: (chunk_width, chunk_height) representing the dimensions of the chunk.

### `update_progress`
Helper function. Callback function to update progress.
**Parameters:**
- `result`: The result from the multiprocessing pool.
- `progress_counter (multiprocessing.Value)`: The progress counter.
- `total_rasters (int)`: Total number of rasters to process.

