from flask import Flask, render_template, request, jsonify, Response, render_template_string, send_file
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
import time
import os
import numpy as np 
from owslib.wms import WebMapService
import math
from PIL import Image as IMG
import io
from IPython.display import Image
from matplotlib.offsetbox import AnchoredText
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from werkzeug.wrappers import Response
import pickle
import warnings
import rasterio
from rasterio.enums import Resampling
import rasterio.warp
from rasterio.transform import from_origin
from celery import Celery
import geone.covModel as gcm
import time 
import sys
sys.path.append('./phenix')
import ArchPy.base as ap
import ArchPy.inputs as inputs
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import imread
import plotly.graph_objects as go
import plotly.io as pio
import zipfile
import tempfile
import shutil
from pyproj import Proj, transform
from datetime import datetime
from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import folium

def create_folder_if_not_exist(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

folders_to_check = ["userdata", "tmp"]

for folder in folders_to_check:
    create_folder_if_not_exist(folder)


warnings.filterwarnings("ignore")
app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'  # change this line
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_polygon', methods=['POST'])
def get_polygon():
    coords = request.get_json()[0]
    area = get_modelBool()
    
    booleanArea = [area.contains(Point(coord['lng'], coord['lat'])) for coord in coords]
    if ~np.all(booleanArea):
      response = jsonify({'message': 'Smoe points are outside the modeling domain'})
      response.status_code = 400  # or any appropriate http status code you want
      return response
    
    points = [deg2ch1903plus(coord['lat'], coord['lng']) for coord in coords]
    userid = str(time.time()) + str(np.random.randint(0,100,1)[0])
    userid = userid.replace(".", "" )
    os.mkdir('userdata/'+userid)
    np.savetxt('userdata/'+userid+'/polygon.txt', np.array(points))
    
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    if len(bh_inarea) == 0:
      response = jsonify({'message': 'No boreholes are in the selected area.'})
      response.status_code = 400  # or any appropriate http status code you want
      return response
    return userid

def get_modelBool():
    polygon = np.loadtxt('static/extend_gva.csv', skiprows=1, delimiter=',')
    polygon = Polygon(polygon)
    return polygon

def loadPolygon(userid, shapely=False):
    polygon = np.loadtxt('userdata/'+userid+'/polygon.txt')
    if shapely:
        polygon = Polygon(polygon)
    return polygon

@app.route('/computing/<userid>')
def computing(userid):
    
    try:
      polygon = loadPolygon(userid)
    except:
      return render_template('error.html')
    
    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    
    topo = load_geotiff_and_resample(extend, [25,25], tiff_type='DEM')
    SedElev = load_geotiff_and_resample(extend, [25,25], tiff_type='BEM')
    SedDepth = topo - SedElev
    SedDepth[SedDepth < 0] = 0
    
    results = dict()
    results['userid'] = userid
    
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    tables = getBHReport(userid) 
    
    results['Number of Boreholes in Area'] = len(bh_inarea)
    results['Number of Descibed Layers'] = len(db_bh_inarea)
    
    
    model = {}
    model['extend'] = extend
    model['Lx'] = extend[2] - extend[0]
    model['Ly'] = extend[3] - extend[1]
    model['bhMaxDepth'] = np.nanmax(bh_inarea.bh_depth)
    model['bhMinDepth'] = np.nanmin(bh_inarea.bh_depth)
    model['bhMeanDepth'] = np.round(np.nanmean(bh_inarea.bh_depth),2)
    
    model['bhMaxQuatDepth'] = np.nanmin(db_bh_inarea[db_bh_inarea.Strat_ID == 'Quaternaire'].bot)
    model['SwissTopoMinQuatAlt'] = np.nanmin(SedElev.flatten())
    
    model['SwissTopoMaxDepth'] =  np.nanmax(SedDepth.flatten())
    model['SwissTopoMinDepth'] =  np.nanmin(SedDepth.flatten())
    model['SwissTopoMeanDepth'] =  np.round(np.nanmean(SedDepth.flatten()),2)
    
    model['TopoMin'] = np.floor(np.nanmin(topo.flatten()))
    model['TopoMean'] = np.round(np.nanmean(topo.flatten()),2)
    model['TopoMax'] = np.ceil(np.nanmax(topo.flatten()))

    return render_template('computing.html', result = results, tables=tables, model=model)
    
if __name__ == '__main__':
    app.run(debug=True) 
    
def deg2ch1903plus(lat, lng):
    
    def DECtoSEX(angle):
        deg = int(angle)
        min = int((angle - deg) * 60)
        sec = ((angle - deg) * 60 - min) * 60
        return deg + min/100 + sec/10000
    
    
    def DEGtoSEC(angle):
        deg = int(angle)
        min = int((angle - deg) * 100)
        sec = ((angle - deg) * 100 - min) * 100
        sec = sec + min * 60 + deg * 3600
        return sec
    
    lat = DECtoSEX(lat)
    lng = DECtoSEX(lng)
    lat = DEGtoSEC(lat)
    lng = DEGtoSEC(lng)
    
    lat_aux = (lat - 169028.66) / 10000
    lng_aux = (lng - 26782.5) / 10000
    
    # Process Y
    x = 600072.37 + 211455.93 * lng_aux - 10938.51 * lng_aux * lat_aux - 0.36 * lng_aux * math.pow(lat_aux, 2) - 44.54 * math.pow(lng_aux, 3)
    
    # Process X
    y = 200147.07 + 308807.95 * lat_aux + 3745.25 * math.pow(lng_aux, 2) + 76.63 * math.pow(lat_aux, 2) - 194.56 * math.pow(lng_aux, 2) * lat_aux + 119.79 * math.pow(lat_aux, 3)
    
    return x+2000000, y+1000000
    
def ch1903p_to_wgs84(x, y):
    in_proj = Proj(init='epsg:2056')
    out_proj = Proj(init='epsg:4326')  # WGS84 EPSG code
    
    lon, lat = transform(in_proj, out_proj, x, y)
    return lat, lon


def getAerial(userid):
    polygon = np.loadtxt('userdata/'+userid+'/polygon.txt')
    polygon = np.insert(polygon,0,polygon[-1],axis=0)

    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    nx = int((extend[2]-extend[0])/2)
    ny = int((extend[3]-extend[1])/2)
    wms_url = "https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=fr"
    wms = WebMapService(wms_url, version="1.3.0")

    img = wms.getmap(
    layers=['ch.swisstopo.swissimage'],
    srs="EPSG:2056",
    size=[nx, ny],
    bbox=extend,
    format="image/jpeg")

    dat = Image(img.read())
    img = IMG.open(io.BytesIO(dat.data))
    arr = np.asarray(img)
    return arr, polygon, extend
    
def getHillshade(userid):
    polygon = np.loadtxt('userdata/'+userid+'/polygon.txt')
    polygon = np.insert(polygon,0,polygon[-1],axis=0)

    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    nx = int((extend[2]-extend[0])/2)
    ny = int((extend[3]-extend[1])/2)
    wms_url = "https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=fr"
    wms = WebMapService(wms_url, version="1.3.0")

    img = wms.getmap(
    layers=['ch.swisstopo.swissalti3d-reliefschattierung'],
    srs="EPSG:2056",
    size=[nx, ny],
    bbox=extend,
    format="image/jpeg")

    dat = Image(img.read())
    img = IMG.open(io.BytesIO(dat.data))
    arr = np.asarray(img)
    return arr, polygon, extend
 
def getSedThickModel(userid):
    polygon = np.loadtxt('userdata/'+userid+'/polygon.txt')
    polygon = np.insert(polygon,0,polygon[-1],axis=0)

    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    nx = int((extend[2]-extend[0])/2)
    ny = int((extend[3]-extend[1])/2)
    wms_url = "https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=fr"
    wms = WebMapService(wms_url, version="1.3.0")

    img = wms.getmap(
    layers=['ch.swisstopo.geologie-lockergestein_maechtigkeitsmodell'],
    srs="EPSG:2056",
    size=[nx, ny],
    bbox=extend,
    format="image/jpeg")

    dat = Image(img.read())
    img = IMG.open(io.BytesIO(dat.data))
    arr = np.asarray(img)
    return arr, polygon, extend
    
def getSedThickSurfaceModel(userid):
    polygon = np.loadtxt('userdata/'+userid+'/polygon.txt')
    polygon = np.insert(polygon,0,polygon[-1],axis=0)

    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    nx = int((extend[2]-extend[0])/2)
    ny = int((extend[3]-extend[1])/2)
    wms_url = "https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=fr"
    wms = WebMapService(wms_url, version="1.3.0")

    img = wms.getmap(
    layers=['ch.swisstopo.geologie-felsoberflaeche_hoehenmodell'],
    srs="EPSG:2056",
    size=[nx, ny],
    bbox=extend,
    format="image/jpeg")

    dat = Image(img.read())
    img = IMG.open(io.BytesIO(dat.data))
    arr = np.asarray(img)
    return arr, polygon, extend
 
@app.route('/plotSedThick/<userid>')
def plotSedThick(userid):
    arr, polygon, extend = getSedThickModel(userid)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(arr, extent=[extend[0],extend[2],extend[1],extend[3]])
    ax.plot(polygon[:,0],polygon[:,1], linewidth=5, color='orange', alpha=0.8)
    # Save it to a temporary buffer.
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/plotHillshade/<userid>')
def plotHillshade(userid):
    arr, polygon, extend = getHillshade(userid)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(arr, extent=[extend[0],extend[2],extend[1],extend[3]])
    ax.plot(polygon[:,0],polygon[:,1], linewidth=5, color='orange', alpha=0.8)
    # Save it to a temporary buffer.
    
    bbox = fig.get_tightbbox(FigureCanvas(fig).get_renderer())
    padding = 0.05 * fig.dpi # adjust the padding as needed
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=bbox, pad_inches=padding)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')
    
@app.route('/plotarea/<userid>')
def plotarea(userid):
    arr, polygon, extend = getAerial(userid)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(arr, extent=[extend[0],extend[2],extend[1],extend[3]])
    ax.plot(polygon[:,0],polygon[:,1], linewidth=5, color='orange', alpha=0.8)
    # Save it to a temporary buffer.
    bbox = fig.get_tightbbox(FigureCanvas(fig).get_renderer())
    padding = 0.05 * fig.dpi # adjust the padding as needed
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=bbox, pad_inches=padding)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')
    
def get_Boreholes(userid):
    if os.path.exists("userdata/"+userid+"/BH_extracted.pickle"):
        with open("userdata/"+userid+"/BH_extracted.pickle", 'rb') as f:
            db_bh_inarea, bh_inarea = pickle.load(f)
        return db_bh_inarea, bh_inarea
    else:
        polygon = np.loadtxt('userdata/'+userid+'/polygon.txt')
        polygon = np.insert(polygon,0,polygon[-1],axis=0)

        with open("data/BH_Gva.pickle", 'rb') as f:
            final_db,list_bhs = pickle.load(f)
        
        areaofint = Polygon(polygon)
        isinArea = np.zeros(len(list_bhs))
        for index, x, y in zip(range(len(list_bhs)), list_bhs.bh_x.values, list_bhs.bh_y.values):
            pt = Point(x,y)
            if pt.within(areaofint):
                isinArea[index] = 1
        bh_inarea = list_bhs.iloc[isinArea == 1]
        db_bh_inarea = final_db.loc[bh_inarea.index]
        
        with open("userdata/"+userid+"/BH_extracted.pickle", 'wb') as f:
            pickle.dump([db_bh_inarea, bh_inarea], f)
        
               
        return db_bh_inarea, bh_inarea

@app.route('/plotBH/<userid>')
def plotBH(userid):    
    arr, polygon, extend = getAerial(userid)
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(arr, extent=[extend[0],extend[2],extend[1],extend[3]])
    ax.plot(polygon[:,0],polygon[:,1], linewidth=5, color='orange', alpha=0.8)
    
    ax.scatter(bh_inarea.bh_x.values, bh_inarea.bh_y.values, label='boreholes')
    ax.legend()
    # Save it to a temporary buffer.

    bbox = fig.get_tightbbox(FigureCanvas(fig).get_renderer())
    padding = 0.05 * fig.dpi # adjust the padding as needed
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=bbox, pad_inches=padding)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/computing/BoreholesAnalysis/<userid>')
def BHReport(userid): 
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    bh_inarea['bh_lat'], bh_inarea['bh_lon'] = ch1903p_to_wgs84(bh_inarea['bh_x'].values, bh_inarea['bh_y'].values)
    # Prepare the data for rendering
    layers = db_bh_inarea.groupby('bh_ID').apply(lambda x: x[['Strat_ID', 'Facies_ID', 'top', 'bot']].to_dict('records')).to_dict()
    borehole_depths = bh_inarea['bh_depth']
    facies_occurrence = db_bh_inarea['Facies_ID'].value_counts()
    depth_facies = db_bh_inarea.groupby('Facies_ID')[['top', 'bot']].mean()

    # Generate plots
    depth_distribution_plot = generate_depth_distribution_plot(borehole_depths)
    facies_occurrence_plot = generate_facies_occurrence_plot(facies_occurrence)
    depth_facies_plot = generate_depth_facies_plot(depth_facies)
    borehole_map = generate_borehole_map(db_bh_inarea, bh_inarea)

    # Convert plots to base64-encoded strings for embedding in HTML
    depth_distribution_plot_str = plot_to_base64(depth_distribution_plot)
    facies_occurrence_plot_str = plot_to_base64(facies_occurrence_plot)
    depth_facies_plot_str = plot_to_base64(depth_facies_plot)

    # Render the template with the data
    return render_template('pandas.html',
                           layers=layers,
                           depth_distribution_plot=depth_distribution_plot_str,
                           facies_occurrence_plot=facies_occurrence_plot_str,
                           depth_facies_plot=depth_facies_plot_str,
                           borehole_map=borehole_map._repr_html_())

# Generate plot for borehole depth distribution
def generate_depth_distribution_plot(borehole_depths):
    plt.figure()
    plt.hist(borehole_depths, bins=10)
    plt.xlabel('Depth')
    plt.ylabel('Count')
    plt.title('Borehole Depth Distribution')
    return plt

# Generate plot for facies occurrence
def generate_facies_occurrence_plot(facies_occurrence):
    plt.figure()
    facies_occurrence.plot(kind='bar')
    plt.xlabel('Facies ID')
    plt.ylabel('Count')
    plt.title('Facies Occurrence')
    return plt

# Generate plot for depth vs. facies
def generate_depth_facies_plot(depth_facies):
    plt.figure()
    depth_facies.plot(kind='bar', stacked=True)
    plt.xlabel('Facies ID')
    plt.ylabel('Depth')
    plt.title('Depth vs. Facies')
    return plt

# Generate map of borehole locations
def generate_borehole_map(db_bh_inarea, bh_inarea):
    map = folium.Map(location=[bh_inarea['bh_lat'].mean(), bh_inarea['bh_lon'].mean()], zoom_start=10)
    for index, row in bh_inarea.iterrows():
        folium.Marker(
            location=[row['bh_lat'], row['bh_lon']],
            popup=f"Borehole ID: {index}",
            icon=folium.Icon(icon="cloud"),
        ).add_to(map)
    return map

# Convert a plot to a base64-encoded string
def plot_to_base64(plot):
    buffer = io.BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    plot_str = base64.b64encode(buffer.getvalue()).decode()
    return plot_str

    
    
def getBHReport(userid):
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    bh_inarea.loc[:,'bh_x'] = np.round(bh_inarea.loc[:,'bh_x'],4).values
    bh_inarea.loc[:,'bh_y'] = np.round(bh_inarea.loc[:,'bh_y'],4).values

    db_bh_inarea.loc[:,'top'] = np.round(db_bh_inarea.loc[:,'top'],2).values
    db_bh_inarea.loc[:,'bot'] = np.round(db_bh_inarea.loc[:,'bot'],2).values

    tables = {"db_bh_inarea":db_bh_inarea.to_html(na_rep='Absent', bold_rows=True, border=1, float_format=lambda x: f'{x:.4f}'),
               "bh_inarea":bh_inarea.to_html(na_rep='Absent', bold_rows=True, border=1, float_format=lambda x: f'{x:.4f}'),
              }
    return tables



def load_geotiff_and_resample(extent, res, tiff_type='DEM'):
    xmin, ymin, xmax, ymax = extent
    Sx, Sy = res
    
    files_paths = {'DEM':'data/DEM25-2021.tif',
                   'BEM':'data/BEM25-2021-commonref.tif'}
        
    
    with rasterio.open(files_paths[tiff_type]) as src:
        # Determine the subset of the image that corresponds to the extent
        window = src.window(xmin, ymin, xmax, ymax)
        subset = src.read(1, window=window)

        # Determine the new dimensions of the resampled image
        factor_x = src.res[0] / Sx
        factor_y = src.res[1] / Sy
        new_width = int(subset.shape[1] / factor_x)
        new_height = int(subset.shape[0] / factor_y)

        # Resample the subset to the new resolution
        resampled = np.empty((new_height, new_width), dtype=subset.dtype)
        rasterio.warp.reproject(
            source=subset,
            destination=resampled,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            dst_transform=rasterio.transform.from_bounds(*extent, new_width, new_height),
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )

        return resampled
    

@app.route('/ArchPy_init', methods=['POST', 'GET'])
def ArchPy_init():
    data = request.form
    userid = data['userid']

    task = compute_model.apply_async(args=[userid, data], task_id=userid)
    time.sleep(1)    # Pause 5.5 seconds

    return render_template('ArchPyInitialized.html', data = data, task_id = userid)

def getCovModels():
    with open('data/VarioAnalysis.pickle','rb') as f:
        varios = pickle.load(f)
        
    return varios

@celery.task(bind=True)
def compute_model(self, userid, data):
    pile_Best = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    names = {'Couverture': 0,
     'Alluvion ancienne': 8,
     'Interglaciaire Riss Würm': 9,
     'Moraine würmienne': 7,
     'Moraine rissienne': 11,
     'Ruisselement; Colluvions': 2,
     'Remblais hétérogènes': 1,
     'Retrait würmien': 6,
     'Alluvions récentes': 5,
     'Dépôts lacustres': 3,
     'Retrait rissien': 10,
     'Dépôts palustres': 4}
    
    self.update_state(state='PROGRESS', meta={'step': 1, 'total_steps': 100, 'step_status': "Loading the data"})
    polygon = loadPolygon(userid)
    
    varios = getCovModels()
    
    wd = 'userdata/'+userid+'/AP_Results'

    nreal_units = 10
    name = 'My Quaternary Model'
    nx = int(data['nx'])
    ny = int(data['ny'])
    nz = int(data['nz'])
    sx = float(data['res_x'])
    sy = float(data['res_x'])
    sz = float(data['res_z'])
    oz = float(data['ModelBot'])
    ox = np.min(polygon[:,0])
    oy = np.min(polygon[:,1])


    extent = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]


    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    
    top = 'data/BathyDEM.tif'

    self.update_state(state='PROGRESS', meta={'step': 3, 'total_steps': 100, 'step_status': "Model Initialized"})

    T1 = ap.Arch_table(name = name, working_directory=wd, seed = 20, verbose = 0, ncpu=5)

    covmodelA = gcm.CovModel2D(elem=[('spherical', {'w':1665, 'r':[4028, 4028]}), 
                                     ('gaussian', {'w':400, 'r':[1193, 1193]}), 
                                     ('nugget', {'w':10})])


    # create Lithologies

    units = []

    facies_list = [('Couverture', 'sienna'),
    ('Alluvion ancienne', 'lightblue'),
    ('Interglaciaire Riss Würm', 'maroon'),
    ('Moraine würmienne','darkgoldenrod'),
    ('Moraine rissienne', 'goldenrod'),
    ('Ruisselement; Colluvions', 'teal'),
    ('Remblais hétérogènes', 'chocolate'),
    ('Retrait würmien', 'cadetblue'),
    ('Alluvions récentes', 'steelblue'),
    ('Dépôts lacustres', 'indigo'),
    ('Retrait rissien', 'darkturquoise'),
    ('Dépôts palustres', 'mediumpurple')]

    for i in range(len(facies_list)):
        posinPile = np.nonzero(pile_Best == names[facies_list[i][0]])[0][0] +1
        dic_f_T = {"f_method":"homogenous"}
        surf = ap.Surface(contact="onlap",dic_surf={"int_method" : "grf_ineq","covmodel" : varios[i]})
        Unit = ap.Unit(name=facies_list[i][0], order = posinPile, ID = names[facies_list[i][0]] + 1, color=facies_list[i][1],contact="onlap",surface=surf, dic_facies=dic_f_T)
        units.append(Unit)


    dic_f_T = {"f_method":"homogenous"}
    Tertiaire_surf = ap.Surface(contact="onlap",dic_surf={"int_method" : "grf_ineq","covmodel" : covmodelA})
    Tertiaire = ap.Unit(name="Tertiaire",order=len(facies_list)+1,ID = len(facies_list)+1,color="orange",contact="onlap",surface=Tertiaire_surf, dic_facies=dic_f_T)
    units.append(Tertiaire)
   
    self.update_state(state='PROGRESS', meta={'step': 10, 'total_steps': 100, 'step_status': "DEM Processing and resampling"})    

    dimensions = (nx, ny, nz)
    spacing = (sx, sy, sz)
    origin = (ox, oy, oz)

    T1.add_grid(dimensions, spacing, origin, polygon=Polygon(polygon), top = top) #adding the grid
    self.update_state(state='PROGRESS', meta={'step': 15, 'total_steps': 100, 'step_status': "Boreholes Processing"})
    P1 = ap.Pile('P1')
    P1.add_unit(units)
    T1.set_Pile_master(P1)

    units_to_ignore =  (np.nan, 'Indéterminé', 'Crétacé',
           'Jurassique', 'Trias', 'Permien')
    
    
        
    for index, row in db_bh_inarea.groupby('bh_ID'):
        if len(row[row.Strat_ID == 'Tertiaire']) > 0:
            first_T = np.nonzero((row.Strat_ID == 'Tertiaire').values )[0][0]
            if first_T > 0 and row.iloc[first_T-1].Facies_ID is None:
                print(index)
            
                line = pd.DataFrame({"Strat_ID": 'Quaternaire', "Facies_ID": 'Moraine rissienne', 'top':row.iloc[first_T].top, 'bot':row.iloc[first_T].top}, index=[index])
                
                db_bh_inarea = pd.concat([db_bh_inarea, line])
    db_bh_inarea.index.rename('bh_ID', inplace=True)
    db_bh_inarea.sort_values(by=["bh_ID","top","Strat_ID"], ascending=[True, False, True], inplace=True)

    db_bh_inarea.loc[db_bh_inarea.Strat_ID == 'Quaternaire','Strat_ID'] = db_bh_inarea.loc[db_bh_inarea.Strat_ID == 'Quaternaire','Facies_ID'].values


    boreholes_AP  = inputs.extract_bhs(db_bh_inarea, bh_inarea, T1, units_to_ignore=units_to_ignore, updater = self)
    T1.add_bh(boreholes_AP)
    T1.process_bhs()
    
    
    self.update_state(state='PROGRESS', meta={'step': 20, 'total_steps': 100, 'step_status': "Units Simulation"})

    T1.compute_surf(50, updater = self)

    self.update_state(state='PROGRESS', meta={'step': 98, 'total_steps': 100, 'step_status': "Facies Simulation"})

    # arr=np.zeros([nz, ny, nx])
    # for i in np.unique(T1.get_facies()):# compute probabilities
    #     for iu in range(nreal_units):
    #         for ifa in range(1):
    #             facies=T1.get_facies(iu=iu, ifa=ifa, all_data=False)
    #             arr+=(facies == i)
    #     arr/=(nreal_units)


    # im=img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=arr, varname="P [-]") #create img object
    self.update_state(state='PROGRESS', meta={'step': 99, 'total_steps': 100, 'step_status': "Saving the results"})

    inputs.save_project(T1)

    equalities = np.c_[T1.get_unit('Tertiaire').surface.x, T1.get_unit('Tertiaire').surface.y, T1.get_unit('Tertiaire').surface.z, np.ones(len(T1.get_unit('Tertiaire').surface.z))*np.nan , np.ones(len(T1.get_unit('Tertiaire').surface.z)) * np.nan ]
    inequalities = np.array(T1.get_unit('Tertiaire').surface.ineq)
    inequalities[:,2] = np.nan

    all_points = np.r_[equalities, inequalities]
    all_points = np.c_[all_points, np.ones(all_points.shape[0])*np.nan]
    
    for i in range(all_points.shape[0]):
        all_points[i,-1] = bh_inarea[np.logical_and(bh_inarea.bh_x == all_points[i,0], bh_inarea.bh_y == all_points[i,1])].index[0]
    
    np.savetxt(wd+'/conditionning.txt',all_points)
    
    self.update_state(state='PROGRESS', meta={'step': 100, 'total_steps': 100, 'step_status': "Finished"})
    result = 'success'
    return result

@app.route('/status/<task_id>')
def status(task_id):
    if task_id is None:
        return jsonify({'status': 'ERROR'})
    else :
        task = compute_model.AsyncResult(task_id)
        if task.state == 'PROGRESS':
            progress = task.info.get('step', 0) / task.info.get('total_steps', 1)
            return jsonify({'status': 'PROGRESS', 'progress': progress, 'step_status':task.info.get('step_status', 2)})
        elif task.state == 'SUCCESS':
            result = task.get()
            return jsonify({'status': 'SUCCESS', 'result': result})
        else:
            return jsonify({'status': task.state})
        
        
@app.route('/visu/<userid>')
def visu(userid):
    try:
      Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    except:
      return render_template('error.html')
      
    data = {}
    
    polygon = loadPolygon(userid)
    
    lat, lon = ch1903p_to_wgs84(polygon[:,0], polygon[:,1])
    polygonToDraw = []
    for lat1, lon1 in zip(lat, lon):
        polygonToDraw.append([lat1, lon1])
    data['centerlat'] = np.mean(lat)
    data['centerlon'] = np.mean(lon)

    data['polygon'] = polygonToDraw
    data['maxCrossx'] = 1/Table.nx
    data['maxCrossy'] = 1/Table.ny
    data['initCrossx'] = int(Table.nx / 2)
    data['initCrossy'] = int(Table.ny / 2)
    data['nReal'] = Table.get_units_domains_realizations().shape[0] - 1
    return render_template('visualisation.html', userid = userid, data=data)
    
@app.route('/serveimage/meanDepth/<userid>/<typ>/<depthV>')
def meanDepth(userid, typ, depthV):    
    typ = int(typ)
    depthV = int(depthV)
    
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    # Generate the figure **without using pyplot**.
    polygon = loadPolygon(userid)
    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    
    lowerstd = np.quantile(Table.get_surface()[0][-1,:,:,:], 0.025, axis=0)
    upperstd = np.quantile(Table.get_surface()[0][-1,:,:,:], 0.925, axis=0)
    meanSurf = np.mean(Table.get_surface()[0][-1,:,:,:],axis=0)
    BedRockSwisstopo = load_geotiff_and_resample(extend, [Table.sx,Table.sy], tiff_type='BEM')
    

    if depthV == 1:
        lowerstd = lowerstd - Table.top
        upperstd = upperstd - Table.top
        meanSurf = meanSurf - Table.top
        Topo = load_geotiff_and_resample(extend, [Table.sx,Table.sy], tiff_type='DEM')
        BedRockSwisstopo = BedRockSwisstopo - Topo
    if typ == 0:
        toplot = lowerstd
    elif typ == 1:
        toplot = upperstd
    elif typ == 2:
        toplot = meanSurf
    elif typ == 3:
        toplot = np.flipud(BedRockSwisstopo)
    else:
        toplot = np.zeros((20,20))
    
    if typ == 4:  
        Uncert = 2*np.std(Table.get_surface()[0][-1,:,:,:],axis=0)
        fig = Figure()
        ax = fig.subplots()
        im = ax.imshow(Uncert, extent=[extend[0],extend[2],extend[1],extend[3]], cmap = 'copper', origin='lower')
        fig.colorbar(im,orientation='horizontal',label='Uncertainty (2sigma) [m]')
    
    else:
        minColor = np.nanmin(lowerstd.flatten())   
        maxColor = np.nanmax(upperstd.flatten())   
    
        fig = Figure()
        ax = fig.subplots()
        im = ax.imshow(toplot, extent=[extend[0],extend[2],extend[1],extend[3]], cmap = 'terrain', origin='lower', vmin = minColor, vmax = maxColor)
        fig.colorbar(im,orientation='horizontal',label='Elevation [m]')
   
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/serveimage/crossy/<userid>/<x_cross>/<showCond>/<showID>/<showReal>')
def crossy(userid, x_cross, showCond, showID, showReal):    
    

    fig = Figure()
    ax = fig.subplots()
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    
    x_cross = int(float(x_cross) * (Table.nx))
    if x_cross > (Table.nx - 1):
        x_cross = Table.nx - 1
    lowerstd = np.quantile(Table.get_surface()[0][-1,:,:,x_cross], 0.025, axis=0)
    upperstd = np.quantile(Table.get_surface()[0][-1,:,:,x_cross], 0.925, axis=0)

    points_cond = np.loadtxt('userdata/'+userid+'/AP_Results/conditionning.txt')

    if showReal == '1':
        for i in range(Table.get_surface()[0].shape[1]):
            ax.plot(Table.ygc - Table.oy, Table.get_surface()[0][-1,i,:,x_cross], color='gray', alpha=0.5)

    bh_inarea = points_cond[np.abs(points_cond[:,0] - Table.xgc[x_cross]) <= (Table.sx / 2) ]
    meanSurf = np.mean(Table.get_surface()[0][-1,:,:,x_cross],axis=0)
    upper = np.std(Table.get_surface()[0][-1,:,:,x_cross],axis=0)

    ax.plot(Table.ygc - Table.oy,Table.get_surface()[0][0,0,:,x_cross], label='Topography')
    ax.plot(Table.ygc - Table.oy, meanSurf, label='Bottom of Quaternary')

    ax.fill_between(Table.ygc - Table.oy, upperstd, lowerstd, color='gray', alpha=0.2, label='95% uncertainty')

    if bh_inarea.shape[0] > 0 and showCond == '1':
        ax.scatter(bh_inarea[:,1] - Table.oy, bh_inarea[:,2], label='equality', marker='x') 
        ax.scatter(bh_inarea[:,1] - Table.oy, bh_inarea[:,3], label='Lower Ineq', marker=6) 
        ax.scatter(bh_inarea[:,1] - Table.oy, bh_inarea[:,4], label='Upper Ineq', marker=7) 
    if bh_inarea.shape[0] == 0 and showCond == '1':
        ax.set_title('No Boreholes available in the Cross Section')

    if bh_inarea.shape[0] > 0 and showID == '1':
        ylabel = np.nanmax(bh_inarea[:,2:5], axis=1)
        for i in range(len(ylabel)):
            ax.text(bh_inarea[i,1] - Table.oy, ylabel[i], str(int(bh_inarea[i,-1]))) 

    ax.set_xlabel('Distance from Origin [m]')
    ax.set_ylabel('Elevation [m]')
    ax.legend()
    
    anchored_text = AnchoredText("South", loc=2)
    ax.add_artist(anchored_text)

    anchored_text = AnchoredText("North", loc=1)
    ax.add_artist(anchored_text)

    # Save it to a temporary buffer.
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

  
    
@app.route('/serveimage/crossx2d/<userid>/<xcross>')
def crossx2d(userid, xcross):    
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    arr, polygon, extend = getAerial(userid)
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(arr, extent=[extend[0],extend[2],extend[1],extend[3]])
    ax.plot(polygon[:,0],polygon[:,1], linewidth=5, color='orange', alpha=0.8)
    
    ax.scatter(bh_inarea.bh_x.values, bh_inarea.bh_y.values, label='boreholes')
    ax.legend()
    ax.vlines(Table.xgc[int(xcross)], extend[1],extend[3], color='red')
    # Save it to a temporary buffer.
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/serveimage/crossx/<userid>/<y_cross>/<showCond>/<showID>/<showReal>')
def crossx(userid, y_cross, showCond, showID, showReal):    
    
    fig = Figure()
    ax = fig.subplots()
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    
       
    y_cross = int(float(y_cross) * (Table.ny))
    if y_cross > (Table.ny - 1):
        y_cross = Table.ny - 1
    
    lowerstd = np.quantile(Table.get_surface()[0][-1,:,y_cross,:], 0.025, axis=0)
    upperstd = np.quantile(Table.get_surface()[0][-1,:,y_cross,:], 0.925, axis=0)

    points_cond = np.loadtxt('userdata/'+userid+'/AP_Results/conditionning.txt')

    if showReal == '1':
        for i in range(Table.get_surface()[0].shape[1]):
            ax.plot(Table.xgc - Table.ox, Table.get_surface()[0][-1,i,y_cross], color='gray', alpha=0.5)

    bh_inarea = points_cond[np.abs(points_cond[:,1] - Table.ygc[y_cross]) <= (Table.sy/2) ]
    meanSurf = np.mean(Table.get_surface()[0][-1,:,y_cross,:],axis=0)
    upper = np.std(Table.get_surface()[0][-1,:,y_cross,:],axis=0)

    ax.plot(Table.xgc - Table.ox,Table.get_surface()[0][0,0,y_cross,:], label='Surface DEM')
    ax.plot(Table.xgc - Table.ox, meanSurf, label='Top of Tertiary')

    ax.fill_between(Table.xgc - Table.ox, upperstd, lowerstd, color='gray', alpha=0.2, label='95% uncertainty')

    if bh_inarea.shape[0] > 0 and showCond == '1':
        ax.scatter(bh_inarea[:,0] - Table.ox, bh_inarea[:,2], label='equality', marker='x') 
        ax.scatter(bh_inarea[:,0] - Table.ox, bh_inarea[:,3], label='Lower Ineq', marker=6) 
        ax.scatter(bh_inarea[:,0] - Table.ox, bh_inarea[:,4], label='Upper Ineq', marker=7) 
    if bh_inarea.shape[0] == 0 and showCond == '1':
        ax.set_title('No Boreholes available in the Cross Section')

    if bh_inarea.shape[0] > 0 and showID == '1':
        ylabel = np.nanmax(bh_inarea[:,2:5], axis=1)
        for i in range(len(ylabel)):
            ax.text(bh_inarea[i,0] - Table.ox, ylabel[i], str(int(bh_inarea[i,-1]))) 

    ax.set_xlabel('Distance from Origin [m]')
    ax.set_ylabel('Elevation [m]')
    ax.legend()
    
     # Add text for cross section orientation
    anchored_text = AnchoredText("West", loc=2)
    ax.add_artist(anchored_text)

    anchored_text = AnchoredText("East", loc=1)
    ax.add_artist(anchored_text)

    # Save it to a temporary buffer.
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

def compute_MostFreq(array_4d):
    # Compute the mode along the first axis
    mode_array = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=array_4d)
    # Return the mode array
    return mode_array

def compute_Proba(array_4d):
    # Compute the count of each facies along the first axis
    facies_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(array_4d) + 1), axis=0, arr=array_4d)
   
    facies_probas = facies_counts / array_4d.shape[0]
    # Return the facies probabilities array
    return facies_probas.T

@app.route('/serveimage/crossxfacies/<userid>/<y_cross>/<showCond>/<showID>/<showBest>/<Real>')
def crossxfacies(userid, y_cross, showCond, showID, showBest, Real):
    colors, inv_names = getFaciesList()
    fig = Figure(figsize=(6.4, 7))
    ax = fig.subplots()
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    
    y_cross = int(float(y_cross) * (Table.ny))
    realis = int(Real)

    if showBest == '1':
        toplot = Table.get_units_domains_realizations()[realis][:,y_cross,:].astype(float)
    else:
        toplot = compute_MostFreq(Table.get_units_domains_realizations())[:,y_cross,:].astype(float)
    mask = ~Table.mask[:,y_cross,:]
    toplot[mask] = np.nan

    non_nan_indices = np.where(~mask)

    # find min and max indices of non-NaN values
    min_row, min_col = np.min(non_nan_indices, axis=1)
    max_row, max_col = np.max(non_nan_indices, axis=1)
    min_col = 0
    max_col = toplot.shape[1]-1

    toplot = toplot[min_row:max_row+1, min_col:max_col+1]

    extend = [Table.xg[min_col], Table.xg[max_col+1], Table.zg[min_row], Table.zg[max_row+1]]

    category_cmap = ListedColormap([colors[inv_names[i]] for i in range(len(inv_names))])

    # Plot the category array with the colorbar
    im = ax.imshow(toplot, cmap=category_cmap, origin='lower', extent=extend, aspect='auto', vmin=1, vmax=len(inv_names))
    
    fig.subplots_adjust(bottom=0.25)

    if showCond == '1':
        _ , bh_inarea = get_Boreholes(userid)
        bhMask = np.abs(bh_inarea['bh_y'].values - Table.ygc[y_cross]) <= Table.sy 
        bh_inline =  bh_inarea[bhMask] 
        if bh_inline.shape[0] == 0:
            ax.set_title('No Boreholes available in the Cross Section')
        else:
            for i in range(len(bh_inline)):
                ax.plot([bh_inline.iloc[i].bh_x]*2, [bh_inline.iloc[i].bh_z, bh_inline.iloc[i].bh_z - bh_inline.iloc[i].bh_depth], color = 'k', linewidth=4)
                if showID == '1':
                    label = str(bh_inline.iloc[i].name)
                    x, y = bh_inline.iloc[i].bh_x, bh_inline.iloc[i].bh_z - bh_inline.iloc[i].bh_depth - 5

                    ax.text(x, y, label, color='red', fontsize = 14)

    # Create colorbar and adjust labels
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', fraction=0.1, pad=0.15)
    cbar.ax.tick_params(labelrotation=50, labelsize=8, pad=2)
    tick_locs = np.linspace(1.5, len(inv_names) - 0.5, len(inv_names))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([inv_names[i] for i in range(len(inv_names))], ha="right")
    cbar.ax.set_xlabel('Unit', labelpad=10, fontsize=10)

    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_ylim(Table.zg[min_row]+7, Table.zg[max_row+1])
    
     # Add text for cross section orientation
    anchored_text = AnchoredText("West", loc=2)
    ax.add_artist(anchored_text)

    anchored_text = AnchoredText("East", loc=1)
    ax.add_artist(anchored_text)
    
    # Save it to a temporary buffer.
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/serveimage/crossyfacies/<userid>/<x_cross>/<showCond>/<showID>/<showBest>/<Real>')
def crossyfacies(userid, x_cross, showCond, showID, showBest, Real):
    colors, inv_names = getFaciesList()
    fig = Figure(figsize=(6.4, 6.4))
    ax = fig.subplots()
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    
       
    x_cross = int(float(x_cross) * (Table.nx))
    realis = int(Real)

    if showBest == '1':
        toplot = Table.get_units_domains_realizations()[realis][:,:,x_cross].astype(float)
    else:
        toplot = compute_MostFreq(Table.get_units_domains_realizations())[:,:,x_cross].astype(float)    
        
    mask = ~Table.mask[:,:, x_cross]
    toplot[mask] = np.nan

    non_nan_indices = np.where(~mask)

    # find min and max indices of non-NaN values
    min_row, min_col = np.min(non_nan_indices, axis=1)
    max_row, max_col = np.max(non_nan_indices, axis=1)
    min_col = 0
    max_col = toplot.shape[1]-1

    toplot = toplot[min_row:max_row+1, min_col:max_col+1]

    extend = [Table.yg[min_col], Table.yg[max_col+1], Table.zg[min_row], Table.zg[max_row+1]]

    category_cmap = ListedColormap([colors[inv_names[i]] for i in range(len(inv_names))])

    # Plot the category array with the colorbar
    im = ax.imshow(toplot, cmap=category_cmap, origin='lower', extent=extend, aspect='auto', vmin=1, vmax=len(inv_names))
    fig.subplots_adjust(bottom=0.25)

    if showCond == '1':
        _ , bh_inarea = get_Boreholes(userid)
        bhMask = np.abs(bh_inarea['bh_x'].values - Table.xgc[x_cross]) <= Table.sx
        bh_inline =  bh_inarea[bhMask] 
        if bh_inline.shape[0] == 0:
            ax.set_title('No Boreholes available in the Cross Section')
        else:
            for i in range(len(bh_inline)):
                ax.plot([bh_inline.iloc[i].bh_y]*2, [bh_inline.iloc[i].bh_z, bh_inline.iloc[i].bh_z - bh_inline.iloc[i].bh_depth], color = 'k', linewidth=4)
                if showID == '1':
                    label = str(bh_inline.iloc[i].name)
                    x, y = bh_inline.iloc[i].bh_y, bh_inline.iloc[i].bh_z - bh_inline.iloc[i].bh_depth - 5

                    ax.text(x, y, label, color='red', fontsize = 14)

    # Create colorbar and adjust labels
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', fraction=0.1, pad=0.15)
    cbar.ax.tick_params(labelrotation=50, labelsize=8, pad=2)
    tick_locs = np.linspace(1.5, len(inv_names) - 0.5, len(inv_names))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([inv_names[i] for i in range(len(inv_names))], ha="right")
    cbar.ax.set_xlabel('Unit', labelpad=10, fontsize=10)

    ax.set_xlabel('Northing [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_ylim(Table.zg[min_row]+7, Table.zg[max_row+1])
     # Add text for cross section orientation
    anchored_text = AnchoredText("South", loc=2)
    ax.add_artist(anchored_text)

    anchored_text = AnchoredText("North", loc=1)
    ax.add_artist(anchored_text)
    # Save it to a temporary buffer.
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/serveimage/crossy2d/<userid>/<ycross>')
def crossy2d(userid, ycross):    
    arr, polygon, extend = getAerial(userid)
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(arr, extent=[extend[0],extend[2],extend[1],extend[3]])
    ax.plot(polygon[:,0],polygon[:,1], linewidth=5, color='orange', alpha=0.8)
    
    ax.scatter(bh_inarea.bh_x.values, bh_inarea.bh_y.values, label='boreholes')
    ax.legend()
    # Save it to a temporary buffer.
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    bbox = fig.get_tightbbox(FigureCanvas(fig).get_renderer())
    padding = 0.05 * fig.dpi # adjust the padding as needed
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=bbox, pad_inches=padding)
    # Embed the result in the html output.
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/boreholeQuery', methods=['GET', 'POST'])
def BHQuery():
    
    
    # Get the ID from the form
    BHid = int(request.form['BHid'])
    userid = request.form['userid']

    with open('data/BH_Gva.pickle', 'rb') as f :
        Bh_db, bh_list = pickle.load(f)
    bhQuery = bh_list.loc[[BHid]]
    bhQuery.columns = ['Northing', 'Easting', 'Elevation', 'Depth']
    bhQuery.index.name = 'Borehole ID'
    html1 = bhQuery.T.to_html(na_rep='Absent', bold_rows=True, border=1, float_format=lambda x: f'{x:.4f}')

    Layers = Bh_db.loc[[BHid]]
    Layers.columns = ['Unit', 'Facies', 'Top Altitude', 'Bottom Altitude']

    Layers.index.name = 'Borehole ID'
    html2 = Layers.to_html(na_rep='No Description', bold_rows=True, border=1)

    points_cond = np.loadtxt('userdata/'+ userid +'/AP_Results/conditionning.txt')

    Conditionning = pd.DataFrame(points_cond[points_cond[:,-1] == BHid])
    Conditionning.columns  = ['Northing', 'Easting','Equality', 'Inequality inf', 'Inequality sup', 'Borehole ID']
    Conditionning.set_index('Borehole ID', inplace=True)
    html3 = Conditionning.T.to_html(na_rep='Absent', bold_rows=False, border=1, float_format=lambda x: f'{x:.4f}')


    # Return the HTML as a JSON object
    response = {
        'data1': html1,
        'data2': html2,
        'data3': html3,
        'data4': str(BHid)
    }
    
    return jsonify(response)

def getFaciesList():
    facies_list = [('Couverture', 'sienna'),
    ('Alluvion ancienne', 'lightblue'),
    ('Interglaciaire Riss Würm', 'maroon'),
    ('Moraine würmienne','darkgoldenrod'),
    ('Moraine rissienne', 'goldenrod'),
    ('Ruisselement; Colluvions', 'teal'),
    ('Remblais hétérogènes', 'chocolate'),
    ('Retrait würmien', 'cadetblue'),
    ('Alluvions récentes', 'steelblue'),
    ('Dépôts lacustres', 'indigo'),
    ('Retrait rissien', 'darkturquoise'),
    ('Dépôts palustres', 'mediumpurple'),
    ('Molasse', 'grey')]

    names = {'Couverture': 0,
     'Alluvion ancienne': 8,
     'Interglaciaire Riss Würm': 9,
     'Moraine würmienne': 7,
     'Moraine rissienne': 11,
     'Ruisselement; Colluvions': 2,
     'Remblais hétérogènes': 1,
     'Retrait würmien': 6,
     'Alluvions récentes': 5,
     'Dépôts lacustres': 3,
     'Retrait rissien': 10,
     'Dépôts palustres': 4,
     'Molasse': 12}

    colors = dict(facies_list)
    inv_names = {v: k for k, v in names.items()}

    return colors, inv_names

@app.route('/visu/3dplot/<userid>')
def Plot3D(userid):

    colors, inv_names = getFaciesList()
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    data = compute_MostFreq(Table.get_units_domains_realizations()).astype(float).flatten()    


    z, y, x = np.meshgrid(Table.zgc, Table.ygc, Table.xgc, indexing='ij')
    z, y, x = z.flatten(), y.flatten(), x.flatten()

    z, y, x = z[data != 0], y[data != 0], x[data != 0]
    data = data[data != 0]
    # Create a mask for NaN values
    # Create a 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        mode='markers',
        marker=dict(
            size=6,
            color=data.flatten(),  # Use categorical array for color
            colorscale=list(colors.values()),  # Use the color names from the dictionary
            cmin=1,  # Set the minimum value for the colorscale
            cmax=13,  # Set the maximum value for the colorscale
            showscale=True,  # Show the colorscale
            colorbar=dict(
                title='Category'  # Set the title for the colorscale
            ),
            cauto=False,  # Prevent auto-scaling of the colorscale
            opacity=1
        ),
        text=[inv_names[i-1] for i in data.flatten()],  # Add the label text for each point
    ))

    # Set axis labels
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='auto' # Set aspect ratio to auto

    ), 
    coloraxis_colorbar=None
    #dict(
    #  	title="Unit",
    #  	tickvals= [inv + 1 for inv in inv_names.keys()],
    #  	ticktext= [inv_names[inv] for inv in inv_names.keys()],
    #  	lenmode="pixels", len=100)
	)

    # Export the figure as an HTML string
    html_string = pio.to_html(fig, full_html=False)
    return render_template_string(html_string)

@app.route('/visu/3dslices/<userid>')
def slice3D(userid):

    colors, inv_names = getFaciesList()
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    
    volume = compute_MostFreq(Table.get_units_domains_realizations()).astype(float)
    volume[volume == 0] = np.nan

    # Define frames
    nb_frames = volume.shape[0]  # Assuming the Z-axis is the third dimension in your volume data

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(k) * np.ones(volume[k,:,:].shape),
        surfacecolor=volume[k,:,:],
        cmin=1, cmax=13,
        colorscale=list(colors.values()),
        opacityscale=[[0, 0], [1/13, 1], [1, 1]]),
        name=str(k),
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=0 * np.ones(volume[0,:,:].shape),
        surfacecolor=volume[0,:,:],
        colorscale=list(colors.values()),
        cmin=1, cmax=13
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
            title='Slices at constant depth',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[0, nb_frames], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders,
        coloraxis_colorbar=dict(
      	title="Unit",
      	tickvals= [inv + 1 for inv in inv_names.keys()],
      	ticktext= [inv_names[inv] for inv in inv_names.keys()],
      	lenmode="pixels", len=100,
	)
    )

    # Export the figure as an HTML string
    html_string = pio.to_html(fig, full_html=False)
    return render_template_string(html_string)
    
@app.route('/download_model/<userid>')
def download_model(userid):
    
    # Define the paths
    downloadfiles_path = './downloadfiles'
    ap_results_path = './userdata/'+userid+'/AP_Results'

    # Create a unique temporary folder within the /tmp directory for each user
    temp_folder = tempfile.mkdtemp(prefix='model_export_', dir='./tmp')

    # Create a README file
    readme_content = '''
    This zip file contains the model exported from ArchPy, and a jupyter notebook that shows how to read, import and resimulate it.

    Contents:
    - README file
    - A jupyter notebook file
    - The "AP_Results" folder with the results inside

    This model was automatically generated using publically available data. It is only provided as demonstration of the ArchPy capabilities.
    The user should use it at his own responsability.

    © Université de Neuchâtel 2023, PheniX project.
    
    based on the work of :

    Alexis Neven
    Ludovic Schorpp
    Julien Straubhaar
    Philippe Renard

    '''
    readme_filepath = os.path.join(temp_folder, 'README.txt')
    with open(readme_filepath, 'w') as readme_file:
        readme_file.write(readme_content)


    # Create a temporary zip file
    zip_filename = 'ModelExport.zip'
    zip_filepath = os.path.join(temp_folder, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        # Add files from the downloadfiles folder to the zip
        for root, dirs, files in os.walk(downloadfiles_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, downloadfiles_path))

        # Add the AP_Results folder to the zip in a subfolder called "AP_Results"
        for root, dirs, files in os.walk(ap_results_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.join('AP_Results', os.path.relpath(file_path, ap_results_path)))

    # Send the zip file as a response for download
    response = send_file(zip_filepath, as_attachment=True)

    # Remove the temporary folder and its contents
    os.remove(zip_filepath)
    shutil.rmtree(temp_folder, ignore_errors=True)
    return response

@app.route('/download_boreholes/<userid>')
def download_boreholes(userid):
    db_bh_inarea, bh_inarea = get_Boreholes(userid)
  
    # Create Excel writer
    temp_folder = tempfile.mkdtemp(prefix='excel_export_', dir='./tmp')
    excel_filepath = os.path.join(temp_folder, 'ModelExport.xlsx')
    writer = pd.ExcelWriter(excel_filepath, engine='xlsxwriter')

    # Write DataFrames to separate sheets
    bh_inarea.to_excel(writer, sheet_name='Borehole List', na_rep='Non Défini')
    db_bh_inarea.to_excel(writer, sheet_name='Strati List', na_rep='Non Défini')

    # Save and close the Excel writer
    writer.save()
    writer.close()

    # Send the Excel file as a response for download
    response = send_file(excel_filepath, as_attachment=True)

    # Remove the temporary Excel file and folder
    os.remove(excel_filepath)
    shutil.rmtree(os.path.dirname(excel_filepath), ignore_errors=True)

    return response
    
    
def generate_lithological_log(zg, mostCommon, Proba, color, Labels, names):
    # Create a new figure and gridspec
    fig = Figure(figsize=(8.27, 11.69))  # A4 size (landscape orientation)
    gs = fig.add_gridspec(3, 3, width_ratios=[0.4, 0.2, 0.4], height_ratios=[1.3, 14, 0.3])  # 3 rows: table, subplots, colorbar

    # Add the logo above the table
    ax_logo = fig.add_subplot(gs[0, 0])
    logo_image = imread('./static/logocompressed.png')
    ax_logo.imshow(logo_image)
    ax_logo.axis('off')  # Turn off axis for the logo subplot

    # Add the table subplot
    ax_table = fig.add_subplot(gs[0, 1:])
    table_data = []
    for label, value in Labels.items():
        table_data.append([label, value])
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.5, 1.5)
    ax_table.axis('off')  # Turn off axis for the table subplot

    # Define the vertical positions for the lithology column
    layer_positions = np.cumsum(np.abs(np.diff(zg)))  # Vertical positions for layer boundaries
    layer_positions = np.insert(layer_positions, 0, 0)  # Insert 0 for the topmost boundary
    layer_positions = np.flipud(layer_positions)

    # Add the lithology subplot
    ax_lithology = fig.add_subplot(gs[1, 0])
    for i, (top, bottom) in enumerate(zip(layer_positions[:-1], layer_positions[1:])):
        facies = mostCommon[i]
        facies_color = color[names[facies-1]]
        ax_lithology.fill_betweenx([top, bottom], 0, 1, facecolor=facies_color)
    ax_lithology.set_ylim(layer_positions[0], layer_positions[-1])
    ax_lithology.set_yticks(layer_positions[:-1] + np.diff(layer_positions) / 2)
    ax_lithology.set_xticklabels([])  # Remove y-axis tick labels
    ax_lithology.set_xlim(0, 1)
    ax_lithology.set_xlabel('Most Probable Facies')
    ax_lithology.set_ylabel('Depth [m]')

    # Add the probabilities subplot
    ax_probabilities = fig.add_subplot(gs[1, 2])
    for i, pos in enumerate(layer_positions[:-1]):
        probas = Proba[i]
        proba_colors = [color[names[facies]] for facies in range(0, len(probas))]
        bar_width = np.array(probas)
        left = [0] + list(np.cumsum(bar_width)[:-1])
        ax_probabilities.barh((layer_positions[i+1] + layer_positions[i])/2, bar_width, height=layer_positions[i+1] - layer_positions[i],
                              left=left, color=proba_colors, align='center')
    ax_probabilities.set_ylim(layer_positions[0], layer_positions[-1])
    ax_probabilities.set_yticks(layer_positions[:-1] + np.diff(layer_positions) / 2)
    ax_probabilities.set_yticklabels([])  # Remove y-axis tick labels
    ax_probabilities.set_xlabel('Probability')
    # Set the x-axis limits and labels for the probabilities subplot
    ax_probabilities.set_xlim(0, 1)

    # Add the labels subplot
    ax_labels = fig.add_subplot(gs[1, 1])
    ax_labels.set_ylim(layer_positions[0], layer_positions[-1])
    ax_labels.set_xlim(0, 1)

    ax_labels.set_yticks(layer_positions[:-1] + np.diff(layer_positions) / 2)
    ax_labels.set_yticklabels([])  # Remove y-axis tick labels
    ax_labels.spines['left'].set_visible(False)
    ax_labels.spines['top'].set_visible(False)
    ax_labels.spines['right'].set_visible(False)
    ax_labels.tick_params(left=False, top=False, right=False, labelleft=False)
    for i in range(len(layer_positions[:-1])):
        facies = mostCommon[i]
        facies_name = names.get(facies-1, '')
        pos = (layer_positions[i] + layer_positions[i+1])/2
        ax_labels.text(0.5, pos, facies_name, ha='center', va='center')
        ax_labels.plot([0,1],[layer_positions[i],layer_positions[i]],'k')
    ax_labels.axis('off')
    
    # Add the colorbar subplot
    category_cmap = ListedColormap([color[names[i]] for i in range(len(names))])


    ax_colorbar = fig.add_subplot(gs[2, :])
    colorbar_image = np.repeat(np.arange(len(names)).reshape(1, -1), 10, axis=0)  # Dummy image for colorbar
    ax_colorbar.imshow(colorbar_image, aspect='auto', cmap=category_cmap, extent=[0, len(names), 0, 1])
    ax_colorbar.set_xticks(np.arange(len(names)) + 0.5)
    ax_colorbar.set_xticklabels([names[i] for i in range(len(names))], rotation=50, fontsize=8, ha='right')
    ax_colorbar.set_xlabel('Category', labelpad=10, fontsize=10)
    ax_colorbar.spines['top'].set_visible(False)
    ax_colorbar.spines['bottom'].set_visible(False)
    ax_colorbar.spines['left'].set_visible(False)
    ax_colorbar.spines['right'].set_visible(False)
    ax_colorbar.tick_params(left=False, bottom=False)
    ax_colorbar.set_xlim(0, len(names))
    ax_colorbar.set_ylim(0, 1)
    ax_colorbar.yaxis.set_visible(False)

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    
    return fig
    
@app.route('/getvirtualborehole/<userid>', methods=['POST'])
def get_virtual_borehole(userid):
    data = request.get_json()
    lat = data['lat']
    lng = data['lng']
    x,y = deg2ch1903plus(lat, lng)
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    
    
    a  = Table.coord2cell(x,y)
    # Generate the zg, mostCommon, Proba, color, Labels, names based on lat and lng
    if a is None:
        error_message = "Point chosen is not in the area of the model"
        return jsonify({'error': error_message})
    else:
        celly, cellx = a
        
    colors, inv_names = getFaciesList()

    borehole = Table.get_units_domains_realizations()[:,:,celly,cellx]
    mask = borehole != 0
    most_freq = compute_MostFreq(borehole)[mask[0]]

    borehole = borehole[:,mask[0]]
    zg = Table.zg[0:most_freq.shape[0]]
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    Labels = {'X': np.round(x, 2) , 'Y': np.round(y, 2), 'Altitude Top': np.round(Table.top[celly,cellx], 2), 'Generated on the':dt_string }  # Example Labels dictionary

    proba = compute_Proba(borehole)[:,1:]

    
    # Generate the lithological log figure
    fig = generate_lithological_log(zg, most_freq, proba, colors, Labels, inv_names)

    copyright_text = "Copyright © 2023 Phenix UNINE. No liability. userid: " + userid
    fig.text(0.95, 0.01, copyright_text, ha='right', fontsize=6)
    fig.text(0.1, 0.95, "Virtual Borehole", ha='left', fontsize=20, fontweight='bold')
    
    # Save the figure as a PDF in memory
    pdf_bytes = io.BytesIO()
    with PdfPages(pdf_bytes) as pdf:
        pdf.savefig(fig)
    pdf_bytes.seek(0)

    # Create a response with the PDF file
    response = Response(pdf_bytes, mimetype='application/pdf')
    response.headers['Content-Disposition'] = 'attachment; filename=lithological_log.pdf'
    return response
    
    
def generate_geotiffs(folder_path, ox, oy, sx, sy, nx, ny, arrays, names):

    # Create the transformation matrix
    transform = from_origin(ox, oy, sx, sy)

    # Generate the three arrays for the GeoTiffs
    for arr, name in zip(arrays, names):
      arr = np.flipud(np.nan_to_num(arr, nan=-9999))

      with rasterio.open(os.path.join(folder_path, name), 'w', driver='GTiff', width=nx, height=ny, count=1, dtype=arr.dtype, crs='EPSG:2056', transform=transform) as dst:
          dst.write(arr, 1)
          dst.nodata = -9999
    return
    
@app.route('/download_geotiff/<userid>')
def download_geotiff(userid):
    
    Table = inputs.import_project(project_name='My Quaternary Model', ws='./userdata/'+userid+'/AP_Results', import_bhs=False, verbose= False)
    # Generate the figure **without using pyplot**.
    polygon = loadPolygon(userid)
    extend = [polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()]
    
    lowerstd = np.quantile(Table.get_surface()[0][-1,:,:,:], 0.025, axis=0)
    upperstd = np.quantile(Table.get_surface()[0][-1,:,:,:], 0.925, axis=0)
    meanSurf = np.mean(Table.get_surface()[0][-1,:,:,:],axis=0)
    stdSurf = np.std(Table.get_surface()[0][-1,:,:,:],axis=0)
    BedRockSwisstopo = load_geotiff_and_resample(extend, [Table.sx,Table.sy], tiff_type='BEM')
    

    # Create a unique temporary folder within the /tmp directory for each user
    temp_folder = tempfile.mkdtemp(prefix='GeoTiff_', dir='./tmp')
    downloadfiles_path = os.path.join(temp_folder, 'Tiffs')
    os.mkdir(downloadfiles_path)
    generate_geotiffs(downloadfiles_path, Table.ox, Table.yg[-1], Table.sx, Table.sy, Table.nx, Table.ny, [meanSurf, stdSurf, upperstd, lowerstd, BedRockSwisstopo], ['01_MeanSurface.tiff', '02_StdSurface.tiff', '03_UpperInterval.tiff','04_LowerInteral.tiff','05_SwissTopoBEM.tiff'])
    
    # Create a README file
    readme_content = '''
    The zip file containts 5 Geotiffs, showing the Bedrock Elevation.

    Contents:
    '01_MeanSurface.tiff' : The mean surface over the realizations
    '02_StdSurface.tiff' : The standard deviation (uncertainty) over the realizations
    '03_UpperInterval.tiff' 95% upper interval (2 times the Std)
    '04_LowerInteral.tiff' : 5% lower interval (2 times the Std)
    '05_SwissTopoBEM.tiff' : Bedrock Elevation model from Swisstopo
    
    The BEM from swisstopo is freely available on https://www.swisstopo.admin.ch/en/geodata/geology/models/bedrock-elevation.html and the copyright belongs to the Federal Office of Topography swisstopo. 

    This model was automatically generated using publically available data. It is only provided as demonstration of the ArchPy capabilities.
    The user should use it at his own responsability.

    © Université de Neuchâtel 2023, PheniX project.
    
    based on the work of :

    Alexis Neven
    Ludovic Schorpp
    Julien Straubhaar
    Philippe Renard

    '''
    readme_filepath = os.path.join(downloadfiles_path, 'README.txt')
    with open(readme_filepath, 'w') as readme_file:
        readme_file.write(readme_content)

    # Create a temporary zip file
    zip_filename = 'BedRockElevationModel.zip'
    zip_filepath = os.path.join(temp_folder, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        # Add files from the downloadfiles folder to the zip
        for root, dirs, files in os.walk(downloadfiles_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, downloadfiles_path))

       
    # Send the zip file as a response for download
    response = send_file(zip_filepath, as_attachment=True)

    # Remove the temporary folder and its contents
    os.remove(zip_filepath)
    shutil.rmtree(temp_folder, ignore_errors=True)
    return response
    

