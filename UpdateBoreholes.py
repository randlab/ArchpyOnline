import sys
sys.path.append('phenix')
from io import BytesIO, StringIO

from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd
import numpy as np
import pickle
import ArchPy.base as ap
import ArchPy.inputs as inputs

import ArchPy as ap
import geone.covModel as gcm

from pyproj import Proj, transform

def ch1903p_to_wgs84(x, y):
    in_proj = Proj(init='epsg:2056')
    out_proj = Proj(init='epsg:4326')  # WGS84 EPSG code
    
    lon, lat = transform(in_proj, out_proj, x, y)
    return lat, lon
    
    
print('Connecting to the SITG server...')
url = 'https://ge.ch/sitg/geodata/SITG/OPENDATA/2998/CSV_GOL_SONDAGE_GEOLOGIE.zip'
resp = urlopen(url)
myzip = ZipFile(BytesIO(resp.read()))
Wells = pd.read_csv(myzip.open('GOL_SONDAGE_GEOLOGIE.csv'), sep=';').sort_values(by='PROFONDEUR_TOIT')
print('Initial Processing...')
for id_sondage, Well in Wells.sort_values(by='PROFONDEUR_TOIT').groupby('ID_SONDAGE'):
    depthMax = np.nanmax([np.abs(Well['PROFONDEUR_BASE']), np.abs(Well['PROFONDEUR_TOIT'])])
    Wells.loc[Wells['ID_SONDAGE'] == id_sondage,'DEPTH'] = depthMax
    if np.any(np.isnan(Well['ALTITUDE_TERRAIN'])):
        altitudeterrain = np.max(Well['ALTTITUDE_TOIT']) + np.min(Well['PROFONDEUR_TOIT'])
        Wells.loc[Wells['ID_SONDAGE'] == id_sondage,'ALTITUDE_TERRAIN'] = altitudeterrain
Wells['PROFONDEUR_TOIT'] = np.abs(Wells['PROFONDEUR_TOIT'])
Wells.sort_values(['ID_SONDAGE','PROFONDEUR_TOIT'],inplace=True)
keys = {'Jurassique sup.':'Jurassique',
       'Jurassique moy.':'Jurassique',
       'Jurassique inf.':'Jurassique',
       'Crétacé inf.':'Crétacé',
       'Trias sup.':'Trias',
       'Trias moy.':'Trias',
       'Trias inf.':'Trias'}
keys_facies= {'Couverture; Terre végétale':'Couverture',
           'Terrains de couverture':'Couverture',
           'Alluvions indifférenciées':'Alluvions récentes',
           'Moraine indifférenciée':'Moraine würmienne',
           'Dépôts intramorainiques':'Moraine würmienne'}
print('Processing...')

final_db, list_bhs = inputs.load_bh_files(Wells.drop_duplicates('ID_SONDAGE'), Wells, Wells, 
                     lbhs_bh_id_col="ID_SONDAGE", u_bh_id_col="ID_SONDAGE", fa_bh_id_col="ID_SONDAGE",
                  u_top_col="ALTTITUDE_TOIT",u_bot_col="ALTITUDE_BASE",u_ID="PERIODE",
                  fa_top_col="ALTTITUDE_TOIT",fa_bot_col="ALTITUDE_BASE",fa_ID="UNITE_GEOL",
                  bhx_col="COORD_X", bhy_col='COORD_Y', bhz_col='ALTITUDE_TERRAIN', bh_depth_col='DEPTH',
                  dic_units_names=keys,
                  dic_facies_names=keys_facies, altitude=True, verbose = 0);


print('Exporting...')
               
with open('data/BH_Gva.pickle','wb') as f:
    pickle.dump([final_db,list_bhs],f)

list_bhs['newx'], list_bhs['newy'] = ch1903p_to_wgs84(list_bhs.bh_x.values, list_bhs.bh_y.values) 
list_bhs[['newx','newy']].to_csv('static/boreholes.csv', header=['X', 'Y'])
print('Done !')
