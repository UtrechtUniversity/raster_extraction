import raster_extraction_functions as ref

if __name__ == "__main__":
    ref.extract_values(
        input_csv=r'C:\path\to\input.csv', #file at least 2 coordinate columns calld X and Y
        raster_folder=r'C:\path\to\rasterfolder', #directory where rasters to be used are saved, no sub dir
        output_csv=r'C:\path\to\output.csv', #output file to be created
        in_crs='EPSG:28992', #crs of input csv
        raster_crs='EPSG:3035', #crs of rasters to be used (EPSG:3035 in case of EXPANSE rasters)
        sep = ';', #column separator in input file, output will always be semi-colon
        decimal=',', #decimal separator in input file
        writemethod='concat' #method to write output to csv, "concat"  is fast but memory-intensive, "rows" is slow but requires no extra memory"
    )
