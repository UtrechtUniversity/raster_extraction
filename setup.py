from setuptools import setup, find_packages

setup(
    name="raster_extraction_tool",
    version="0.1.2",
    author="L. Roga",
    author_email="l.q.roga@uu.nl",
    description="Extract raster data at input coordinates. Meant for processing many rasters at once.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/UtrechtUniversity/raster_extraction",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "pandas",
        "pyproj",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)