import os
os.environ['GDAL_CACHEMAX'] = '512'
import gc
import pandas as pd
import numpy as np
from pyproj import Transformer
from osgeo import gdal
from multiprocessing import Pool, cpu_count, Manager
from datetime import datetime
import csv
from raster_extraction_tool.helper_functions import calculate_chunk_size, update_progress

def process_raster(x_coords, y_coords, raster_folder, raster_file):
    """
    Worker function to process a single raster file in chunks and extract values at specified coordinates.

    Parameters:
    x_coords (numpy.ndarray): Array of x coordinates (longitude) for points of interest.
    y_coords (numpy.ndarray): Array of y coordinates (latitude) for points of interest.
    raster_folder (str): Directory containing the raster files.
    raster_file (str): Name of the raster file to process.

    Returns:
    tuple: (column_name, values) where column_name is the raster file name without extension,
           and values is a numpy array of extracted values at the specified coordinates.
    """
    raster_path = os.path.join(raster_folder, raster_file)

    dataset = gdal.Open(raster_path)
    if not dataset:
        return None, None

    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()

    #calculate chunk size for reading the raster in memory-efficient chunks
    chunk_width, chunk_height = calculate_chunk_size(dataset)

    values = np.full(len(x_coords), np.nan, dtype='float32')  
    

    for y_offset in range(0, dataset.RasterYSize, chunk_height):
        rows_to_read = min(chunk_height, dataset.RasterYSize - y_offset)
        raster_array = band.ReadAsArray(0, y_offset, chunk_width, rows_to_read).astype('float32')


        x_indices = np.floor((x_coords - geotransform[0]) / geotransform[1]).astype(int)
        y_indices = np.floor((y_coords - geotransform[3]) / geotransform[5]).astype(int) - y_offset

        mask = (x_indices >= 0) & (x_indices < chunk_width) & (y_indices >= 0) & (y_indices < rows_to_read)
        chunk_mask = mask & np.isnan(values)  # Only update values that haven't been set yet

        values[chunk_mask] = raster_array[y_indices[chunk_mask], x_indices[chunk_mask]]


    column_name = os.path.splitext(raster_file)[0]
    
    del dataset
    
    return column_name, values

def extract_values(input_csv, raster_folder, output_csv, in_crs, raster_crs, writemethod='concat', sep=';', decimal='.'):
    """
    Main function to orchestrate the multiprocessing of raster files for extracting values at specified coordinates.

    Parameters:
    input_csv (str): Path to the input CSV file containing coordinates.
    raster_folder (str): Directory containing the raster files.
    output_csv (str): Path to the output CSV file where results will be saved.
    in_crs (str or int): Coordinate reference system of the input coordinates.
    raster_crs (str or int): Coordinate reference system of the raster files.
    writemethod (str, optional): Method to write output CSV ('concat' or 'rows'). Default is 'concat'.
    sep (str, optional): Delimiter to use in the input CSV file. Default is ';'.
    decimal (str, optional): Character to recognize as decimal point in the input CSV file. Default is '.'.

    Returns:
    None
    """
    start_time = datetime.now()
    
    print(f'Reading input CSV...')
    df = pd.read_csv(input_csv, sep=sep, decimal=decimal)
    df['X'] = pd.to_numeric(df['X'], errors='coerce', downcast='float')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce', downcast='float')
    df.dropna(subset=['X', 'Y'], inplace=True)


    print(f'Transforming coordinates from {in_crs} to {raster_crs}...')
    transformer = Transformer.from_crs(in_crs, raster_crs, always_xy=True)
    x_transformed, y_transformed = transformer.transform(df['X'].values, df['Y'].values)
    df['X'], df['Y'] = x_transformed.astype('float32'), y_transformed.astype('float32')


    raster_files = [f for f in os.listdir(raster_folder) if f.endswith(".tif")]
    total_rasters = len(raster_files)

    #setup multiprocessing
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    x_coords = df['X'].values
    y_coords = df['Y'].values

    results = []
    with Pool(processes=cpu_count()) as pool:
        for raster_file in raster_files:
            result = pool.apply_async(
                process_raster, 
                args=(x_coords, y_coords, raster_folder, raster_file), 
                callback=lambda result: update_progress(result, progress_counter, total_rasters)
            )
            results.append(result)
        pool.close()
        pool.join()

    print('Writing extracted raster values to dictionary...')
    new_columns_data = {}
    for result in results:
        column_name, values = result.get()
        if column_name: 
            new_columns_data[column_name] = values

    print('Writing values dictionary to DataFrame...')
    new_columns_df = pd.DataFrame(new_columns_data)
    new_columns_data = None
    gc.collect()

    #write to CSV using the specified method
    if writemethod == 'concat':
        print(f'Concatenating DataFrames...')
        final_df = pd.concat([df.reset_index(drop=True), new_columns_df.reset_index(drop=True)], axis=1)
        print(f'Writing DataFrame to output CSV...')
        final_df.to_csv(output_csv, index=False, sep=sep, decimal=decimal)
        print(f"Processed and output to {output_csv}")

    elif writemethod == 'rows':
        print('Writing DataFrame to CSV row by row...')
        assert len(df) == len(new_columns_df), f"DataFrames must have the same number of rows. Original: {len(df)}, new values: {len(new_columns_df)}"
        
        with open(output_csv, 'w', newline='') as csvfile:
            headers = df.columns.tolist() + new_columns_df.columns.tolist()
            csv_writer = csv.writer(csvfile, delimiter=sep)
            csv_writer.writerow(headers)
            
            for index in range(len(df)):
                if index % 10000 == 0:
                    print(f'Writing row {index}')
                row_to_write = df.iloc[index].tolist() + new_columns_df.iloc[index].tolist()
                csv_writer.writerow(row_to_write)

    end_time = datetime.now()
    total_time = end_time - start_time  
    print(f"Completed in {total_time}")
