import os.path
from os import walk

import ee
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gamma, f, chi2
import IPython.display as disp
import folium

_key = "key.json"
_service_account = 'proyecto-0002@ee-tucho.iam.gserviceaccount.com'
credenciales = ee.ServiceAccountCredentials(_service_account, key_file=_key)
ee.Initialize(credentials=credenciales)


def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds Earth Engine layers to a folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True).add_to(self)


class Tools:
    aoi: ee.Geometry.Point = None
    polygon: ee.Geometry.Polygon = None
    im_coll: ee.ImageCollection = None
    timestamplist: List[str] = None
    im_list: ee.List = None
    vv_list = None
    mp: folium.Map = None
    folium.Map.add_ee_layer = add_ee_layer

    def __init__(self, coords):
        self.aoi = ee.Geometry.Point(coords).buffer(40).bounds()
        self.polygon = ee.Geometry.Polygon(self.aoi.getInfo()['coordinates'])
        self.im_coll = self.collection_image("COPERNICUS/S1_GRD_FLOAT",
                                             "2020-11-15", '2021-04-30',
                                             "VV", "DESCENDING")

        im_list = self.im_coll.toList(self.im_coll.size())
        self.im_list = im_list.map(lambda img: ee.Image(img).clip(self.polygon))
        self.timestamplist = self.list_times()
        self.vv_list = self.im_list.map(lambda current: ee.Image(current).select('VV'))
        # Add EE drawing method to folium.

    @staticmethod
    def chi2cdf(chi2, df) -> ee.Image:
        """Calculates Chi square cumulative distribution function for
           df degrees of freedom using the built-in incomplete gamma
           function gammainc().
        """
        return ee.Image(chi2.divide(2)).gammainc(ee.Number(df).divide(2))

    @staticmethod
    def det(im):
        """Calculates determinant of 2x2 diagonal covariance matrix."""
        return im.expression('b(0)*b(1)')

    @staticmethod
    def func_qom(feature: ee.Feature) -> ee.Feature:
        ft = feature.set('Date', ee.Date(feature.get('Date'))) \
            .set('ids', feature.get('ids'))
        return ee.Feature(ft)

    def clip_img(self, img):
        """Clips a list of images."""
        return ee.Image(img).clip(self.polygon)

    @staticmethod
    def selectvv(current) -> ee.Image:
        return ee.Image(current).select('VV')

    @staticmethod
    def omnibus(im_list, m=4.7) -> ee.image.Image:
        """ Calculates the omnibus test statistic, monovariate case. """

        def log(current):
            return ee.Image(current).log()

        im_list = ee.List(im_list)
        k = im_list.length()
        klogk = k.multiply(k.log())
        klogk = ee.Image.constant(klogk)
        sumlogs = ee.ImageCollection(im_list.map(log)).reduce(ee.Reducer.sum())
        logsum = ee.ImageCollection(im_list).reduce(ee.Reducer.sum()).log()

        return klogk.add(sumlogs).subtract(logsum.multiply(k)).multiply(-2 * m)

    def collection_image(self, catalogo: str, begin_date: str, end_date: str,
                         polarisation: str, orientation: str):
        return (ee.ImageCollection(catalogo)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarisation))
                .filterBounds(self.polygon)
                .filterDate(ee.Date(begin_date), ee.Date(end_date))
                .filter(ee.Filter.eq('orbitProperties_pass', orientation))
                .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
                .sort('date'))

    def list_times(self):
        return (self.im_coll.aggregate_array('date')
                .map(lambda d: ee.String('T').cat(ee.String(d)))
                .getInfo())

    def list_img(self):
        self.im_list = self.im_coll.toList(self.im_coll.size())
        self.im_list = ee.List(self.im_list.map(self.clip_img))

    def long_im_list(self):
        return self.im_list.length().getInfo()

    def first_map(self):
        location = self.polygon.centroid().coordinates().getInfo()[::-1]
        mp = folium.Map(location=location, zoom_start=25)
        rgb_images = (ee.Image.rgb(self.vv_list.get(0), self.vv_list.get(1), self.vv_list.get(2))
                      .log10().multiply(10))
        mp.add_ee_layer(rgb_images, {'min': -20, 'max': 0}, 'rgb composite')
        mp.add_child(folium.LayerControl())
        self.save_map(mp)

    @staticmethod
    def file_name(startwith: str, type: str):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mapas")
        files = list(filter(lambda file: type in file, next(walk(path), (None, None, []))[2]))
        files.sort()
        return os.path.join("mapas", f"{startwith}{ int(files[-1][4:-5]) + 1}{type}" if len(files) else f"{startwith}01{type}")

    def save_map(self, mapa):
        mapa.save(self.file_name("mapa0", ".html"))
        self.comparasion()

    @staticmethod
    def get_files():
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mapas")
        lista = list(filter(lambda file: ".html" in file, next(walk(path), (None, None, []))[2]))
        lista.sort()
        return lista

    def comparasion(self):
        k = self.long_im_list()
        hist = (self.omnibus(self.vv_list.slice(0, k))
                .reduceRegion(ee.Reducer.fixedHistogram(0, 10, 200),
                              geometry=self.polygon,
                              scale=1)
                .get('constant')
                .getInfo())
        a = np.array(hist)
        x = a[:, 0]
        y = a[:, 1] / np.sum(a[:, 1])
        plt.plot(x, y, '.', label='data')
        plt.plot(x, chi2.pdf(x, k - 1), '-r', label='chi square')
        plt.legend()
        plt.grid()
        plt.savefig(self.file_name("plot0", ".png"))
