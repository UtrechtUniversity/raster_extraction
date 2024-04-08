import os
# Increase GDAL Cache Size to 512MB
os.environ['GDAL_CACHEMAX'] = '512'
import gc
import pandas as pd
import numpy as np
from pyproj import Transformer
from osgeo import gdal, gdal_array
from multiprocessing import Pool, cpu_count, Manager
import time
import csv

def _calculate_chunk_size(dataset, max_chunk_memory=500 * 1024**2):
    """Calculate the chunk size for reading the raster based on the desired maximum memory footprint."""
    band = dataset.GetRasterBand(1)
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)  # Convert GDAL data type to NumPy data type
    dtype_size = np.dtype(dtype).itemsize
    raster_x_size = dataset.RasterXSize
    raster_y_size = dataset.RasterYSize

    # Calculate the number of pixels that can be read within the memory limit
    max_pixels = max_chunk_memory // dtype_size

    # Determine a suitable chunk size (width and height) that respects the raster dimensions
    if raster_x_size * raster_y_size <= max_pixels:
        return raster_x_size, raster_y_size  # No need to chunk
    else:
        chunk_width = raster_x_size 
        chunk_height = max(1, max_pixels // chunk_width)
    
        return chunk_width, chunk_height

def _update_progress(result, progress_counter, total_rasters):
    """Callback function to update progress."""
    progress_counter.value += 1
    print(f"Processed {progress_counter.value}/{total_rasters} rasters")
    # if progress_counter.value % 100 == 0:
    #     gc.collect()

def process_raster(x_coords, y_coords,raster_folder,raster_file):
    """Worker function to process a single raster file in chunks."""
    raster_path = os.path.join(raster_folder, raster_file)

    dataset = gdal.Open(raster_path)
    if not dataset:
        return None, None

    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()

    chunk_width, chunk_height = _calculate_chunk_size(dataset)

    # Initialize array to hold values
    values = np.full(len(x_coords), np.nan, dtype='float32')  

    for y_offset in range(0, dataset.RasterYSize, chunk_height):
        rows_to_read = min(chunk_height, dataset.RasterYSize - y_offset)
        raster_array = band.ReadAsArray(0, y_offset, chunk_width, rows_to_read).astype('float32')

        # Calculate global x and y indices for the dataframe points
        x_indices = np.floor((x_coords - geotransform[0]) / geotransform[1]).astype(int)
        y_indices = np.floor((y_coords - geotransform[3]) / geotransform[5]).astype(int) - y_offset

        # Mask for points falling within the current chunk
        mask = (x_indices >= 0) & (x_indices < chunk_width) & (y_indices >= 0) & (y_indices < rows_to_read)
        chunk_mask = mask & np.isnan(values)  # Only update values that haven't been set yet

        values[chunk_mask] = raster_array[y_indices[chunk_mask], x_indices[chunk_mask]]

    column_name = os.path.splitext(raster_file)[0]
    del dataset
    return column_name, values

def extract_values(input_csv, raster_folder, output_csv, in_crs, raster_crs, writemethod='concat',sep= ';',decimal='.'):
    """Main function to orchestrate the multiprocessing of raster files."""
    start_time = time.time() 
    print(f'Reading input csv...')
    df = pd.read_csv(input_csv, sep=sep,decimal = decimal)
    df['X'] = pd.to_numeric(df['X'], errors='coerce', downcast='float')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce', downcast='float')
    df.dropna(subset=['X', 'Y'], inplace=True)

    #transform coordinates
    print(f'Transforming coordinates from {in_crs} to {raster_crs}...')
    transformer = Transformer.from_crs(in_crs, raster_crs, always_xy=True)
    x_transformed, y_transformed = transformer.transform(df['X'].values, df['Y'].values)
    df['X'], df['Y'] = x_transformed.astype('float32'), y_transformed.astype('float32')

    raster_files = [f for f in os.listdir(raster_folder) if f.endswith(".tif")]
    total_rasters = len(raster_files)

    manager = Manager()
    progress_counter = manager.Value('i', 0)
    x_coords = df['X'].values
    y_coords = df['Y'].values

    results = []
    with Pool(processes=cpu_count()) as pool:
        for raster_file in raster_files:
            result = pool.apply_async(process_raster, args=(x_coords,y_coords,raster_folder,raster_file), callback=lambda result: _update_progress(result, progress_counter, total_rasters))
            results.append(result)
        pool.close()
        pool.join()

    print('Writing extracted raster values to dictionary...')
    # Prepare a dictionary to collect new column data
    new_columns_data = {}
    for result in results:
        column_name, values = result.get()
        if column_name:  # Ensure the result was successful
            new_columns_data[column_name] = values

    print('Writing values dictionary to dataframe...')
    # Create a new DataFrame from the aggregated results and concatenate it to the original DataFrame
    new_columns_df = pd.DataFrame(new_columns_data)
    new_columns_data = None
    gc.collect()
    
    #Concatening dataframes before writing to CSV, this is not memory-efficient, but fast if enough memory is available
    if writemethod == 'concat':
        print(f'Concatenating dataframes...')
        final_df = pd.concat([df.reset_index(drop=True), new_columns_df.reset_index(drop=True)], axis=1) #NORMAL METHOD
        print(f'Writing dataframe to output csv...')
        final_df.to_csv(output_csv, index=False, sep=';')
        print(f"Processed and output to {output_csv}")

    #Writing to CSV row by row without concatenating dataframes. This is memory efficient (no in-memory concatenation), but slower.
    elif writemethod == 'rows':
        print('Writing dataframe to csv...')
        assert len(df) == len(new_columns_df), f"DataFrames must have the same number of rows. Original: {len(df)}, new values: {len(new_columns_df)}" # Ensure both DataFrames have the same number of rows

        with open(output_csv, 'w', newline='') as csvfile:
            headers = df.columns.tolist() + new_columns_df.columns.tolist()
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            
            for index in range(len(df)):
                if index % 10000 == 0:
                    print(f'Writing row {index}')
                row_to_write = df.iloc[index].tolist() + new_columns_df.iloc[index].tolist()
                #TODO: WRITE MORE ROWS AT ONCE (faster?)
                # Write the constructed row to the CSV file
                csv_writer.writerow(row_to_write)
    end_time = time.time()  
    total_time = end_time - start_time  

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"Completed in {hours} hours, {minutes} minutes, and {seconds} seconds.")