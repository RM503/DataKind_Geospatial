import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Geocoding distributor locations

        Here is a simple function for performing geocoding on the distributor locations using the `Locations_of_Distributors_vDK.xlsx` file provided as part of the dataset. 

        **Note:** Initially, I attempted to use the `Nominatim` geocoder method of `geopy` to perform the geoencoding. However, these locations are not listed in OpenStreetMaps. Hence, I used GoogleV3 geocoder. Also, the geocoded points need to be verified as address formatting issues may have lead to errors.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    from geopy.geocoders import GoogleV3 
    from shapely import Point
    import matplotlib.pyplot as plt
    import seaborn as sns
    import leafmap 

    import warnings
    warnings.filterwarnings("ignore")
    return GoogleV3, Point, gpd, leafmap, mo, pd, plt, sns, warnings


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Formatting

        The only formatting step that is necessary is to append 'Kenya' to the end of the address.
        """
    )
    return


@app.cell
def _(pd):
    df = pd.read_excel('Locations_of_Distributors_vDK.xlsx')

    df
    return (df,)


@app.cell
def _(GoogleV3, gpd, pd):
    def geolocator(x: str) -> pd.Series:
        '''
        This function takes in a string and returns its longitude and latitude.
        '''
        geolocator = GoogleV3(api_key='AIzaSyBXxVB4Xp3ol3_t7KAelHMTNlk4Gdb4XVM')

        loc = geolocator.geocode(x)
        lon = loc.longitude
        lat = loc.latitude

        return pd.Series((lon, lat))

    def format_and_geocode(df: pd.DataFrame, filename: str) -> gpd.GeoDataFrame:
        '''
        This function formats the `Locations` column and geocodes them using GoogleV3 geocoder.
        Furthermore, the longitude and latitude information are converted into a `Point` geometry
        denoted in WKT format.

        Args: df - dataframe containing street address column for geocoding
              filename - export file name
        Return: geopandas dataframe with geocoding information
        '''

        # Cleaning the locations by appending `, Kenya` to the end
        df['Locations_cleaned'] = df['Locations'].apply(lambda x: x + ', Kenya' if 'Kenya' not in x else x)

        # Applying the `geolocator()` function on all `Locations_cleaned` to obtained 
        # associated longigutde and latitude
        df[['Longitude', 'Latitude']] = df['Locations_cleaned'].apply(geolocator)

        # Converting longitude and latitude to point geometry and convert to GeoPandas dataframe
        df['geometry'] = gpd.points_from_xy(df['Longitude'], df['Latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        gdf.to_file(filename + '_geocoded.json', driver='GeoJSON')
        gdf.to_file(filename + '_geocoded.json', driver='GPKG')

        return gdf
    return format_and_geocode, geolocator


@app.cell
def _(gpd):
    gdf = gpd.read_file('distributor_locations_geocoded.json')

    gdf.head()
    return (gdf,)


@app.cell
def _(gdf):
    gdf.to_csv('distributor_locations_geocoded.csv')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Priority list

        There is a list of Caategory A distributors deemed as priority. This also requires formatting since the addresses have been broken down along multiple columns. In order to merge then into a single column for processing, the NaN values are filled with `''`.
        """
    )
    return


@app.cell
def _(pd):
    df_priority = pd.read_csv('Distributor_Locations_Priority.csv')

    # fill NaN with blanks
    df_priority['Unnamed: 6'].fillna('', inplace=True)
    df_priority['Unnamed: 7'].fillna('', inplace=True)
    df_priority['Unnamed: 8'].fillna('', inplace=True)
    return (df_priority,)


@app.cell
def _(df_priority):
    df_priority['Locations'] = (
        df_priority['Locations'] 
        + ' '
        + df_priority['Unnamed: 6'] 
        + ' '
        + df_priority['Unnamed: 7'] 
        + ' '
        + df_priority['Unnamed: 8']
    )

    df_priority.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], inplace=True)
    return


@app.cell
def _(mo):
    mo.md(r"""One particular distributor 'Jawapa store' lacked a properly defined street addressed. Thus, it was removed from the list.""")
    return


@app.cell
def _(df_priority):
    df_priority.drop(index=4, inplace=True)
    return


@app.cell
def _(df_priority):
    df_priority.reset_index(drop=True, inplace=True)
    return


@app.cell
def _(mo):
    mo.md(r"""## Geocoded priority distributors""")
    return


@app.cell
def _(df_priority, format_and_geocode):
    gdf_priority = format_and_geocode(df_priority, 'distributor_locations_priority')

    gdf_priority
    return (gdf_priority,)


@app.cell
def _(df_priority, gdf_priority):
    df_priority.loc[6, 'County'] = 'Narok'
    gdf_priority.loc[6, 'County'] = 'Narok'
    return


@app.cell
def _(gdf_priority):
    gdf_priority.to_csv('distributor_locations_priority_geocoded.csv')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Leads in the vicinity of distributors

        Here, we use the cleaned data produced by James. This file does not contain any `County` column. Hence, a simple merge might not exist.
        """
    )
    return


@app.cell
def _(pd):
    leads_df = pd.read_csv('./../data/leads_clean_20250131.csv')

    leads_df.head()
    return (leads_df,)


@app.cell
def _(leads_df, plt, sns):
    sns.histplot(leads_df, x='Total farm size', color='blue', edgecolor='black')
    plt.xscale('log')
    plt.show()
    return


@app.cell
def _(leads_df):
    print(f"Minimum farm size: {leads_df['Total farm size'].min()} acres")
    print(f"Maximum farm size: {leads_df['Total farm size'].max()} acres")
    return


@app.cell
def _(leads_df):
    leads_df['Sales manager area'].unique()
    return


@app.cell
def _(df_priority, leads_df):
    set(leads_df['Sub-County'].tolist()) & set(df_priority['County'].tolist())
    return


@app.cell
def _(mo):
    mo.md(r"""Creating a rough list of counties by combining the `Sales manager area` and `Sub-County` columns from the `leads_df` dataframe with the `County` column in the `df_priority` dataframe.""")
    return


@app.cell
def _(pd):
    def generate_county_list(df_1: pd.DataFrame, df_2: pd.DataFrame) -> list:
        ''' 
        Args: df_1 - priority distributor dataframe
              df_2 - cleaned leads dataframe
        Return: a list of counties (might not be exhaustive)
        '''

        s1 = set(df_2['Sales manager area'].unique()) 
        s2 = set(df_2['Sub-County'].unique())

        s = s1.union(s2) 

        counties = set(df_1['County'].tolist())

        return s & counties
    return (generate_county_list,)


@app.cell
def _(mo):
    mo.md(r"""We obtain five county matches this way. Now, we filter the data using this list, further whittling it down to only those entries with longitude and latitude coordinates.""")
    return


@app.cell
def _(df_priority, generate_county_list, leads_df):
    counties = generate_county_list(df_priority, leads_df)

    print(counties)
    return (counties,)


@app.cell
def _(counties, leads_df):
    leads_filtered_df = leads_df[
        (leads_df['Sales manager area'].isin(counties)) | (leads_df['Sub-County'].isin(counties))
    ]

    leads_filtered_df.shape
    return (leads_filtered_df,)


@app.cell
def _(leads_df, leads_filtered_df):
    leads_filtered = leads_filtered_df[
                                            (leads_df['Final Lat'].notna())
                                            &
                                            (leads_df['Final Long'].notna())
                        ]

    leads_filtered.reset_index(drop=True, inplace=True)
    return (leads_filtered,)


@app.cell
def _(leads_filtered):
    leads_filtered
    return


@app.cell
def _(leads_filtered, leafmap):
    m = leafmap.Map(center = [1, 38] ,google_map='HYBRID', zoom=5)
    m.add_points_from_xy(leads_filtered, x='Final Long', y='Final Lat')
    m
    return (m,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
