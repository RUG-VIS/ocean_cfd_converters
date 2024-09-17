import iris
from os import path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="conversion tool from fluid NetCDF (oceanic data) to GRIB2")
    parser.add_argument("-i", "--input", dest="input_path", type=str, default="test.nc", help="path to input NetCDF file")
    parser.add_argument("-o", "--output", dest="output_path", type=str, default="test.grb", help="target output path of GRIB2 file")
    parser.add_argument("-xn", "--lonname", dest="lonname", type=str, default="longitude", help="name of the 'longitude' variable in input NetCDF (default: longitude).")
    parser.add_argument("-yn", "--latname", dest="latname", type=str, default="latitude", help="name of the 'latitude' variable in input NetCDF (default: latitude).")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    outdir = path.dirname(output_path)
    fname = path.basename(output_path)
    pfname, pfext = path.splitext(fname)


    cubes = iris.load(input_path)       # each variable in the netcdf file is a cube
    cube_list = list(cubes)
    # print(cubes)
    for i in range(len(cube_list)):
        print(cube_list[i])
        cube_list[i].coord(args.latname).coord_system = iris.coord_systems.GeogCS(4326)
        cube_list[i].coord(args.lonname).coord_system = iris.coord_systems.GeogCS(4326)
        varname = cube_list[i].var_name
        print(varname)
        fname_out = path.join(outdir, pfname+"_"+varname+pfext)
        iris.save(cube_list[i],fname_out)  # save a specific variable to grib
