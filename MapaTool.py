import traceback

import ee
from typing import List

import google.auth.exceptions
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gamma, f, chi2
import IPython.display as disp
import rasterio
import folium
import files as fl
import response



def initialize(account, key):
    try:
        credenciales = ee.ServiceAccountCredentials(account, key_file=key)
        ee.Initialize(credentials=credenciales)
        return True
    except Exception:
        raise traceback.format_exc()

class Tools:
    aoi: ee.Geometry.Point = None
    polygon: ee.Geometry.Polygon = None
    im_coll: ee.ImageCollection = None
    timestamplist: List[str] = None
    im_list: ee.List = None
    vv_list = None
    mp: folium.Map = None

    def __init__(self, coords: list, params: list):
        self.aoi = ee.Geometry.Point(coords).buffer(params.pop()).bounds()
        self.polygon = ee.Geometry.Polygon(self.aoi.getInfo()['coordinates'])
        self.im_coll = self.collection_image(*params)
        im_list = self.im_coll.toList(self.im_coll.size())
        self.im_list = im_list.map(lambda img: ee.Image(img).clip(self.polygon))
        self.timestamplist = self.list_times()
        self.vv_list = self.im_list.map(lambda current:
                                        ee.Image(current).select('VV')
                                        )
        folium.Map.add_ee_layer = Funk.add_ee_layer
        # Add EE drawing method to folium.

    @staticmethod
    def func_qom(feature: ee.Feature) -> ee.Feature:
        ft = feature.set('Date', ee.Date(feature.get('Date'))) \
            .set('ids', feature.get('ids'))
        return ee.Feature(ft)

    def clip_img(self, img):
        """Clips a list of images."""
        return ee.Image(img).clip(self.polygon)

    def collection_image(self, catalogo: str, begin_date: str, end_date: str,
                         polarisation: str, orientation: str):
        param1, param2 = 'transmitterReceiverPolarisation', 'orbitProperties_pass'
        return (ee.ImageCollection(catalogo)
                .filter(ee.Filter.listContains(param1, polarisation))
                .filterBounds(self.polygon)
                .filterDate(ee.Date(begin_date), ee.Date(end_date))
                .filter(ee.Filter.eq(param2, orientation))
                .map(lambda img: img.set('date',
                                         ee.Date(img.date()).format('YYYYMMdd')
                                         )
                     )
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

    # %%
    def polygon_loc(self):
        return self.polygon.centroid().coordinates().getInfo()[::-1]

    def folium_map(self, name, params):
        mp = folium.Map(location=self.polygon_loc(), zoom_start=25)
        for args in params.values():
            mp.add_ee_layer(args['rgb'], args['params'], args['title'])
        mp.add_child(folium.LayerControl())

        mp.save(fl.file("mapas", name))

    def first_layer(self, map_name: str):
        rgb_images = (ee.Image.rgb(self.vv_list.get(0),
                                   self.vv_list.get(1),
                                   self.vv_list.get(2))
                      .log10().multiply(10))
        url = rgb_images.getDownloadURL({'scale': 100, 'filePerBand': False})
        values = {'one': {'params': {'min': -20, 'max': 0},
                          'title': 'Rgb Composite',
                          'rgb': rgb_images}}
        self.folium_map(f"1-{map_name}", values)
        self.comparasion(f"1-{map_name}")
        self.second_layer(map_name)

    def second_layer(self, map_name: str):
        k = self.long_im_list()
        alpha = 0.01
        p_value = ee.Image.constant(1).subtract(
            Funk.second_layer(self.vv_list, k)
        )
        c_map = p_value.multiply(0).where(p_value.lt(alpha), 1)
        # Make the no-change pixels transparent.
        c_map = c_map.updateMask(c_map.gt(0))
        # Overlay onto the folium map.
        values = {"1": {'params': {'min': 0, 'max': 1,
                                   'palette': ['black', 'red']
                                   },
                        'title': 'Change Map',
                        'rgb': c_map
                        }
                  }
        self.folium_map(f"2-{map_name}", values)

        self.third_layer(map_name)

    def third_layer(self, map_name):
        # Funk.samples(self.vv_list, self.polygon)
        # cmap: the interval of the most recent change
        # smap: the interval of the first change
        # bmap: the changes in each interval
        try:
            result = Funk.change_maps(self.im_list, True, 0.01)
            # Extract the change maps and display.
            color_codes = ['cmap', 'smap', 'fmap']
            palette = ['black', 'blue', 'cyan', 'yellow', 'red']
            values = {f"{i}": {'rgb': ee.Image(result.get(code)),
                               'params': {'min': 0, 'max': 25,
                                          'palette': palette
                                          },
                               'title': 'C Map'
                               }
                      for i, code in enumerate(color_codes)
                      }
            self.folium_map(f"3-{map_name}", values)
        except Exception as ex:
            raise ex
        # Cuarta capa
        self.four_layer(map_name)

    def four_layer(self, map_name):

        try:
            result = ee.Dictionary(Funk.change_maps2(self.im_list,
                                                     median=True, alpha=0.01))
        except Exception as ex:
            raise ex
        # Extract the change maps and export to assets.
        cmap = ee.Image(result.get('cmap'))
        smap = ee.Image(result.get('smap'))
        fmap = ee.Image(result.get('fmap'))
        bmap = ee.Image(result.get('bmap'))
        cmaps = ee.Image.cat(cmap, smap, fmap, bmap).rename(
            ['cmap', 'smap', 'fmap'] + self.timestamplist[1:])
        cmaps = cmaps.updateMask(cmaps.gt(0))

        palette = ['black', 'red', 'cyan', 'yellow']

        lista = cmaps.getInfo()['bands'][3:]
        values = {f"{i}": {'rgb': cmaps.select(each),
                           'params': {'min': 0, 'max': 3, 'palette': palette},
                           'title': each}
                  for i, each in enumerate(self.timestamplist)
                  if (list(filter(lambda ids: ids['id'] == each, lista)))
                  }
        self.folium_map(f"4-{map_name}", values)

    def comparasion(self, plot_name):
        k = self.long_im_list()
        hist = (Funk.omnibus(self.vv_list.slice(0, k))
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
        plt.savefig(fl.file("plots", plot_name))


class Funk:

    @staticmethod
    def second_layer(bb_list, k):
        return Funk.chi2cdf(Funk.omnibus(bb_list), k - 1)

    @staticmethod
    def omnibus(im_list, m=4.7) -> ee.image.Image:
        """ Calculates the omnibus test statistic, monovariate case. """

        im_list = ee.List(im_list)
        k = im_list.length()
        klogk = k.multiply(k.log())
        klogk = ee.Image.constant(klogk)
        sumlogs = ee.ImageCollection(
            im_list.map(lambda current: ee.Image(current).log())) \
            .reduce(ee.Reducer.sum())
        logsum = ee.ImageCollection(im_list).reduce(ee.Reducer.sum()).log()

        return klogk.add(sumlogs).subtract(logsum.multiply(k)).multiply(-2 * m)

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
    def log_det_sum(im_list, j):
        """Returns log of determinant of the sum of the
        first j images in im_list."""
        im_list = ee.List(im_list)
        sumj = ee.ImageCollection(im_list.slice(0, j)).reduce(ee.Reducer.sum())
        return ee.Image(Funk.det(sumj)).log()

    @staticmethod
    def log_det(im_list, j):
        """Returns log of the determinant of the jth image in im_list."""
        im = ee.Image(ee.List(im_list).get(j.subtract(1)))
        return ee.Image(Funk.det(im)).log()

    @staticmethod
    def pval(im_list: ee.List, j, m=4.4):
        """Calculates -2logRj for im_list and returns P value and -2logRj."""
        im_list = ee.List(im_list)
        j = ee.Number(j)
        m2logRj = (Funk.log_det_sum(im_list, j.subtract(1))
                   .multiply(j.subtract(1))
                   .add(Funk.log_det(im_list, j))
                   .add(ee.Number(2).multiply(j).multiply(j.log()))
                   .subtract(ee.Number(2).multiply(j.subtract(1))
                             .multiply(j.subtract(1).log()))
                   .subtract(Funk.log_det_sum(im_list, j).multiply(j))
                   .multiply(-2).multiply(m))
        pv = ee.Image.constant(1).subtract(Funk.chi2cdf(m2logRj, 2))
        return pv, m2logRj

    @staticmethod
    def p_values(im_list):
        """Pre-calculates the P-value array for a list of images."""
        im_list = ee.List(im_list)
        k = im_list.length()

        def ells_map(ell):
            """Arranges calculation of pval for combinations of k and j."""
            ell = ee.Number(ell)
            # Slice the series from k-l+1 to k (image indices start from 0).
            im_list_ell = im_list.slice(k.subtract(ell), k)

            def js_map(j):
                """Applies pval calculation for combinations of k and j."""
                j = ee.Number(j)
                pv1, m2logRj1 = Funk.pval(im_list_ell, j)
                return ee.Feature(None, {'pv': pv1, 'm2logRj': m2logRj1})

            # Map over j=2,3,...,l.
            js = ee.List.sequence(2, ell)
            pv_m2logRj = ee.FeatureCollection(js.map(js_map))

            # Calculate m2logQl from collection of m2logRj images.
            m2logQl = ee.ImageCollection(
                pv_m2logRj.aggregate_array('m2logRj')).sum()
            pvQl = ee.Image.constant(1).subtract(
                Funk.chi2cdf(m2logQl, ell.subtract(1).multiply(2)))
            pvs = ee.List(pv_m2logRj.aggregate_array('pv')).add(pvQl)
            return pvs

        # Map over l = k to 2.
        ells = ee.List.sequence(k, 2, -1)
        pv_arr = ells.map(ells_map)

        # Return the P value array ell = k,...,2, j = 2,...,l.
        return pv_arr

    @staticmethod
    def sample_vv_imgs(vv_list, polygon, j):
        """Samples the test statistics Rj in the region aoi_sub."""
        j = ee.Number(j)
        # Get the factors in the expression for Rj.
        sj = vv_list.get(j.subtract(1))
        jfact = j.pow(j).divide(j.subtract(1).pow(j.subtract(1)))
        sumj = ee.ImageCollection(vv_list.slice(0, j)).reduce(
            ee.Reducer.sum())
        sumjm1 = ee.ImageCollection(
            vv_list.slice(0, j.subtract(1))).reduce(ee.Reducer.sum())
        # Put them together.
        Rj = sumjm1.pow(j.subtract(1)).multiply(sj).multiply(jfact).divide(
            sumj.pow(j)).pow(5)
        # Sample Rj.
        sample = (
            Rj.sample(region=polygon, scale=10, numPixels=1000, seed=123)
            .aggregate_array('VV_sum'))
        return sample

    @staticmethod
    def samples(vv_list, polygon):
        # Sample the first few list indices.
        samples = ee.List([Funk.sample_vv_imgs(vv_list, polygon, j)
                           for j in range(2, 9)
                           ])
        # samples = ee.List.sequence(2, 8).map()
        # Calculate and display the correlation matrix.
        np.set_printoptions(precision=2, suppress=True)
        print(np.corrcoef(samples.getInfo()))

    @staticmethod
    def change_maps(im_list, median=False, alpha=0.01):
        """Calculates thematic change maps."""
        k = im_list.length()
        # Pre-calculate the P value array.
        pv_arr = ee.List(Funk.p_values(im_list))
        # Filter P values for change maps.
        cmap = ee.Image(im_list.get(0)).select(0).multiply(0)
        bmap = ee.Image.constant(ee.List.repeat(0, k.subtract(1))).add(cmap)
        alpha = ee.Image.constant(alpha)
        first = ee.Dictionary({'i': 1, 'alpha': alpha, 'median': median,
                               'cmap': cmap, 'smap': cmap, 'fmap': cmap,
                               'bmap': bmap})
        return ee.Dictionary(pv_arr.iterate(Funk.filter_i, first))

    @staticmethod
    def filter_j(current, prev):
        """Calculates change maps; iterates over j indices of pv_arr."""
        pv = ee.Image(current)
        prev = ee.Dictionary(prev)
        pvQ = ee.Image(prev.get('pvQ'))
        i = ee.Number(prev.get('i'))
        cmap = ee.Image(prev.get('cmap'))
        smap = ee.Image(prev.get('smap'))
        fmap = ee.Image(prev.get('fmap'))
        bmap = ee.Image(prev.get('bmap'))
        alpha = ee.Image(prev.get('alpha'))
        j = ee.Number(prev.get('j'))
        cmapj = cmap.multiply(0).add(i.add(j).subtract(1))
        # Check      Rj?            Ql?                  Row i?
        tst = pv.lt(alpha).And(pvQ.lt(alpha)).And(cmap.eq(i.subtract(1)))
        # Then update cmap...
        cmap = cmap.where(tst, cmapj)
        # ...and fmap...
        fmap = fmap.where(tst, fmap.add(1))
        # ...and smap only if in first row.
        smap = ee.Algorithms.If(i.eq(1), smap.where(tst, cmapj), smap)
        # Create bmap band and add it to bmap image.
        idx = i.add(j).subtract(2)
        tmp = bmap.select(idx)
        bname = bmap.bandNames().get(idx)
        tmp = tmp.where(tst, 1)
        tmp = tmp.rename([bname])
        bmap = bmap.addBands(tmp, [bname], True)
        return ee.Dictionary({'i': i, 'j': j.add(1), 'alpha': alpha, 'pvQ': pvQ,
                              'cmap': cmap, 'smap': smap, 'fmap': fmap,
                              'bmap': bmap})

    @staticmethod
    def filter_i(current, prev):
        """Arranges calculation of change maps; iterates over row-indices of pv_arr."""

        current = ee.List(current)
        pvs = current.slice(0, -1)
        pvQ = ee.Image(current.get(-1))
        prev = ee.Dictionary(prev)
        i = ee.Number(prev.get('i'))
        alpha = ee.Image(prev.get('alpha'))
        median = prev.get('median')
        # Filter Ql p value if desired.
        pvQ = ee.Algorithms.If(median, pvQ.focalMedian(2.5), pvQ)
        cmap = prev.get('cmap')
        smap = prev.get('smap')
        fmap = prev.get('fmap')
        bmap = prev.get('bmap')
        first = ee.Dictionary({'i': i, 'j': 1, 'alpha': alpha, 'pvQ': pvQ,
                               'cmap': cmap, 'smap': smap, 'fmap': fmap,
                               'bmap': bmap})
        result = ee.Dictionary(ee.List(pvs).iterate(Funk.filter_j, first))
        return ee.Dictionary({'i': i.add(1), 'alpha': alpha, 'median': median,
                              'cmap': result.get('cmap'),
                              'smap': result.get('smap'),
                              'fmap': result.get('fmap'),
                              'bmap': result.get('bmap')})

    @staticmethod
    def dmap_iter(current, prev):
        """Reclassifies values in directional change maps."""

        prev = ee.Dictionary(prev)
        j = ee.Number(prev.get('j'))
        image = ee.Image(current)
        avimg = ee.Image(prev.get('avimg'))
        diff = image.subtract(avimg)
        # Get positive/negative definiteness.
        posd = ee.Image(diff.select(0).gt(0).And(Funk.det(diff).gt(0)))
        negd = ee.Image(diff.select(0).lt(0).And(Funk.det(diff).gt(0)))
        bmap = ee.Image(prev.get('bmap'))
        bmapj = bmap.select(j)
        dmap = ee.Image.constant(ee.List.sequence(1, 3))
        bmapj = bmapj.where(bmapj, dmap.select(2))
        bmapj = bmapj.where(bmapj.And(posd), dmap.select(0))
        bmapj = bmapj.where(bmapj.And(negd), dmap.select(1))
        bmap = bmap.addBands(bmapj, overwrite=True)
        # Update avimg with provisional means.
        i = ee.Image(prev.get('i')).add(1)
        avimg = avimg.add(image.subtract(avimg).divide(i))
        # Reset avimg to current image and set i=1 if change occurred.
        avimg = avimg.where(bmapj, image)
        i = i.where(bmapj, 1)
        return ee.Dictionary(
            {'avimg': avimg, 'bmap': bmap, 'j': j.add(1), 'i': i})

        # We only have to modify the change_maps function to include the change direction in the bmap image:

    @staticmethod
    def change_maps2(im_list, median=False, alpha=0.01):
        """Calculates thematic change maps."""
        k = im_list.length()
        # Pre-calculate the P value array.
        pv_arr = ee.List(Funk.p_values(im_list))
        # Filter P values for change maps.
        cmap = ee.Image(im_list.get(0)).select(0).multiply(0)
        bmap = ee.Image.constant(ee.List.repeat(0, k.subtract(1))).add(cmap)
        alpha = ee.Image.constant(alpha)
        first = ee.Dictionary({'i': 1, 'alpha': alpha, 'median': median,
                               'cmap': cmap, 'smap': cmap, 'fmap': cmap,
                               'bmap': bmap})
        result = ee.Dictionary(pv_arr.iterate(Funk.filter_i, first))
        # Post-process bmap for change direction.
        bmap = ee.Image(result.get('bmap'))
        avimg = ee.Image(im_list.get(0))
        j = ee.Number(0)
        i = ee.Image.constant(1)
        first = ee.Dictionary({'avimg': avimg, 'bmap': bmap, 'j': j, 'i': i})
        dmap = ee.Dictionary(
            im_list.slice(1).iterate(Funk.dmap_iter, first)).get('bmap')
        return ee.Dictionary(result.set('bmap', dmap))

    def add_ee_layer(self, ee_image_object, vis_params, name):
        """Adds Earth Engine layers to a folium map."""
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        attribute = 'Map Data &copy; ' \
                    '<a href="https://earthengine.google.com/">Google Earth ' \
                    'Engine</a>'
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr=attribute,
            name=name,
            overlay=True,
            control=True).add_to(self)

    @staticmethod
    def indices(path):
        """Agrega los indices ndvi y ndwi(ambas ecuaciones) como
         las ultimas 3 bandas al raster"""
        dataset = rasterio.open(path)
        green = dataset.read(2)
        red = dataset.read(1)
        nir = dataset.read(4)
        swir = dataset.read(6)
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
        ndwi_mcfeeters = (nir.astype(float) - green.astype(float)) / (
                nir + green)
        ndwi_gao = (nir.astype(float) - swir.astype(float)) / (nir + swir)
        profile = dataset.meta
        profile['count'] = dataset.count + 3
        profile.update(driver='GTiff')
        profile.update(dtype=rasterio.float32)

        bands = dataset.read()
        bands = np.append(bands, np.array([ndvi, ndwi_mcfeeters, ndwi_gao]),
                          axis=0)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(bands, range(1, profile['count'] + 1))
        dataset.close()
