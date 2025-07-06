import xarray as xr
import geopandas as gpd
try:
    from lxml import etree
    print("running with lxml.etree")
except ImportError:
    import xml.etree.ElementTree as etree
    print("running with Python's xml.etree.ElementTree")
from os import path, mkdir
from argparse import ArgumentParser


def parcels_to_geopandas(ds, ram_gb_limit=4, suppress_warnings=False):
    """
    Converts your parcels data to a geopandas dataframe containing a point for
    every observation in the dataframe. Custom particle variables come along
    for the ride during the transformation. Any undefined observations are removed
    (correspond to the particle being deleted, or not having entered the simulation).

    Assumes your parcel output is in lat and lon.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset object in the format of parcels output.

    suppress_warnings : bool
        Whether to ignore RAM warning.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with point data for each particle observation in the dataset.
    """
    RAM_LIMIT_BYTES = ram_gb_limit * 1024 * 1024 * 1024  # 4 GB RAM limit

    if ds.nbytes > RAM_LIMIT_BYTES and not suppress_warnings:
        raise MemoryError(
            "Dataset is %i bytes, but RAM_LIMIT_BYTES set max to be %i bytes." % (ds.nbytes, RAM_LIMIT_BYTES)
        )

    df = (
        ds.to_dataframe().reset_index()  # Convert `obs` and `trajectory` indices to be columns instead
    )

    gdf = (
        gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df["lon"], y=df["lat"]))
        .drop(
            ["lon", "lat"], axis=1
        )  # No need for lon and lat cols. Included in geometry attribute
        .set_crs(
            "EPSG:4326"
        )  # Set coordinate reference system to EPSG:4326 (aka. WGS84; the lat lon reference system)
    )

    # Remove observations with no time from gdf (indicate particle has been removed, or isn't in simulation)
    invalid_observations = gdf["time"].isna()
    invalid_field = gdf["time"]
    return gdf[~invalid_observations]


def parcels_geopandas_to_kml(
        gdf: gpd.GeoDataFrame,
        output_path,
        document_name="Parcels Particle Trajectories",
        rubber_ducks=True,
):
    """Writes parcels trajectories to KML file.

    Converts the GeoDataFrame from the `parcels_to_geopandas` function into KML
    for use in Google Earth. Each particle trajectory is converted to a gx:Track item
    to include timestamp information in the path.

    Only uses the trajectory ID, time, lon, and lat variables.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame as output from the `parcels_to_geopandas` function

    path : pathlike
        Path to save the KML to.

    rubber_ducks : bool
        Replace default particle marker with rubber duck icon.


    See also
    --------
    More on gx:Track in KML:
    https://developers.google.com/kml/documentation/kmlreference#gx:track
    """
    outdir = path.dirname(output_path)
    fname = path.basename(output_path)
    pfname, pfext = path.splitext(fname)

    # Define namspaces
    kml_ns = "http://www.opengis.net/kml/2.2"
    gx_ns = "http://www.google.com/kml/ext/2.2"

    # Custom icon styling
    icon_style_string = """<Style id="iconStyle">
            <IconStyle>
                <scale>0.8</scale>
                <Icon>
                    <href>https://icons.iconarchive.com/icons/thesquid.ink/free-flat-sample/256/rubber-duck-icon.png</href>
                </Icon>
              </IconStyle>
            </Style>
        """

    lead_df = gdf.groupby("trajectory")
    total_items = len(lead_df)
    current_item = 0
    # Generating gx:Track items
    for trajectory_idx, trajectory_gdf in lead_df:
        trajectory_gdf = trajectory_gdf.sort_values("obs")
        # print("{}\n".format(len(trajectory_gdf["time"])))
        if len(trajectory_gdf["time"]) < 3:
            current_item += 1
            continue

        kml_out = etree.Element("{%s}kml" % kml_ns, nsmap={None: kml_ns, "gx": gx_ns})
        document = etree.SubElement(kml_out, "Document", id = "1", name=document_name)
        if rubber_ducks:
            icon_styling = etree.fromstring(icon_style_string)
            document.append(icon_styling)

        placemark = etree.SubElement(document, "Placemark")
        name_element = etree.SubElement(placemark, "name")
        name_element.text = str(trajectory_idx)

        # Link custom icon styling
        if rubber_ducks:
            style_url = etree.SubElement(placemark, "styleUrl")
            style_url.text = "#iconStyle"

        gx_track = etree.SubElement(placemark, "{%s}Track" % gx_ns)
        etree.SubElement(gx_track, "{%s}altitudeMode" % gx_ns, text="clampToGround")

        for time in trajectory_gdf["time"]:
            when_element = etree.SubElement(gx_track, "datetime")
            when_element.text = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        for _, row in trajectory_gdf.iterrows():
            gx_coord_element = etree.SubElement(gx_track, "{%s}coord" % gx_ns)
            gx_coord_element.text = "%f %f 0.0" % (row['geometry'].x, row['geometry'].y)

        # Save the KML to a file
        trajectory_path = path.join(outdir, pfname+"_"+str(trajectory_idx)+pfext)
        with open(trajectory_path, "wb") as f:
            try:
                f.write(etree.tostring(kml_out, pretty_print=True))
            except:
                f.write(etree.tostring(kml_out))

        current_item += 1
        workdone = current_item / total_items
        print("\rProgress - writing KML instances: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)

    print("\n")
    return

    # # Save the KML to a file
    # with open(output_path, "wb") as f:
    #     try:
    #         f.write(etree.tostring(kml_out, pretty_print=True))
    #     except:
    #         f.write(etree.tostring(kml_out))
    # return


def parcels_geopandas_to_kml_time(
        gdf: gpd.GeoDataFrame,
        output_path,
        document_name="Parcels Particle Trajectories",
        rubber_ducks=True,
):
    """Writes parcels trajectories to KML file.

    Converts the GeoDataFrame from the `parcels_to_geopandas` function into KML
    for use in Google Earth. Each particle trajectory is converted to a gx:Track item
    to include timestamp information in the path.

    Only uses the trajectory ID, time, lon, and lat variables.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame as output from the `parcels_to_geopandas` function

    path : pathlike
        Path to save the KML to.

    rubber_ducks : bool
        Replace default particle marker with rubber duck icon.


    See also
    --------
    More on gx:Track in KML:
    https://developers.google.com/kml/documentation/kmlreference#gx:track
    """
    outdir = path.dirname(output_path)
    fname = path.basename(output_path)
    pfname, pfext = path.splitext(fname)

    # Define namspaces
    kml_ns = "http://www.opengis.net/kml/2.2"
    gx_ns = "http://www.google.com/kml/ext/2.2"

    # Custom icon styling
    icon_style_string = """<Style id="iconStyle">
            <IconStyle>
                <scale>0.8</scale>
                <Icon>
                    <href>https://icons.iconarchive.com/icons/thesquid.ink/free-flat-sample/256/rubber-duck-icon.png</href>
                </Icon>
              </IconStyle>
            </Style>
        """

    # lead_df = gdf.groupby("trajectory")
    lead_df = gdf.groupby("obs")
    total_items = len(lead_df)
    current_item = 0
    # Generating gx:Track items
    # for trajectory_idx, trajectory_gdf in lead_df:
    for obs_idx, obs_gdf in lead_df:
        # trajectory_gdf = trajectory_gdf.sort_values("obs")
        obs_gdf = obs_gdf.sort_values("trajectory")

        # print("{}\n".format(len(trajectory_gdf["time"])))
        # if len(obs_gdf["time"]) < 3:
        #     current_item += 1
        #     continue

        kml_out = etree.Element("{%s}kml" % kml_ns, nsmap={None: kml_ns, "gx": gx_ns})
        document = etree.SubElement(kml_out, "Document", id = "1", name=document_name)
        if rubber_ducks:
            icon_styling = etree.fromstring(icon_style_string)
            document.append(icon_styling)


        for rowidx, item in obs_gdf.iterrows():
            trajectory_idx = rowidx
            # trajectory = obs_gdf.iloc[trajectory_idx]

            placemark = etree.SubElement(document, "Placemark")
            name_element = etree.SubElement(placemark, "name")
            name_element.text = str(trajectory_idx)

            # Link custom icon styling
            if rubber_ducks:
                style_url = etree.SubElement(placemark, "styleUrl")
                style_url.text = "#iconStyle"

            gx_track = etree.SubElement(placemark, "{%s}Track" % gx_ns)
            etree.SubElement(gx_track, "{%s}altitudeMode" % gx_ns, text="clampToGround")

            when_element = etree.SubElement(gx_track, "datetime")
            when_element.text = item["time"].strftime("%Y-%m-%dT%H:%M:%SZ")

            gx_coord_element = etree.SubElement(gx_track, "{%s}coord" % gx_ns)
            gx_coord_element.text = "%f %f 0.0" % (item['geometry'].x, item['geometry'].y)

        # Save the KML to a file
        trajectory_path = path.join(outdir, pfname+"_t"+str(obs_idx)+pfext)
        with open(trajectory_path, "wb") as f:
            try:
                f.write(etree.tostring(kml_out, pretty_print=True))
            except:
                f.write(etree.tostring(kml_out))

        current_item += 1
        workdone = current_item / total_items
        print("\rProgress - writing KML instances: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)

    print("\n")
    return


# ==================================================================================================================== #
# Example:
#  python3 convert_zarr_geojson-shp-kmz.py
#     -i /media/christian/DATA/data/hydrodynamics/Mediterranean/benchmark_doublegyre_noMPI_p_n50000_365d_fwd_add_age.zarr
#     -o /media/christian/DATA/data/hydrodynamics/Mediterranean/benchmark_doublegyre_noMPI_p_n50000_365d_fwd_add_age.kml
#
#  python3 convert_zarr_geojson-shp-kmz.py
#     -i /media/christian/DATA/data/hydrodynamics/Mediterranean/benchmark_doublegyre_noMPI_p_n50000_365d_fwd_add_age.zarr
#     -o /media/christian/DATA/data/hydrodynamics/Mediterranean/benchmark_doublegyre_noMPI_p_n50000_365d_fwd_add_age.geojson
# ==================================================================================================================== #
if __name__ == "__main__":
    parser = ArgumentParser(description="conversion tool from fluid zarraz (oceanic data) to GeoJSON or KML")
    parser.add_argument("-i", "--input", dest="input_path", type=str, default="test.nc", help="path to input zarray file")
    parser.add_argument("-o", "--output", dest="output_path", type=str, default="test.grb", help="target output file path")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    outdir = path.dirname(output_path)
    fname = path.basename(output_path)
    pfname, pfext = path.splitext(fname)

    ds_parcels = xr.open_zarr(input_path)

    gdf_parcels = parcels_to_geopandas(ds_parcels, ram_gb_limit=20)
    if "json" in pfext:
        gdf_parcels.to_file(output_path)
    elif ("kml" in pfext) or ("kmz" in pfext):
        outdir = path.join(outdir, "Google")
        if not path.exists(outdir):
            mkdir(outdir)
        output_path = path.join(outdir, fname)
        # parcels_geopandas_to_kml(gdf_parcels, output_path)
        parcels_geopandas_to_kml_time(gdf_parcels, output_path)
