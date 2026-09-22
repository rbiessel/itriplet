from abc import ABC, abstractmethod
import rasterio
import glob
from osgeo import gdal
import os
from scipy.ndimage import uniform_filter
import numpy as np
from datetime import datetime as dt 
import rasterio
from rasterio.windows import transform as window_transform
from rasterio.transform import Affine

class SLCStack(ABC):
    @abstractmethod
    def read(self, index=1, window=None):
        """Read raster data."""
        pass

    def get_baselines(self):
        deltas = self.datetimes[None, :] - self.datetimes[:, None]
        days = np.vectorize(lambda x: x.days)(deltas)
        return days

    def get_pairs(self, diag=True, connectN=5, maxDays=None):
        tri = np.array(np.triu_indices(self.P))
        b = np.linspace(0, self.P - 1, self.P)
        b = b[None, :] - b[:, None]
        if maxDays is None:
            maxDays = self.P * 12
        baselines = self.get_baselines()

        pairs = np.logical_and(b >= (not diag), b <= connectN, baselines <= maxDays)
        pairs = np.logical_and(pairs, baselines <= maxDays)
        pairs = np.argwhere(pairs)
        pairs = [[int(pair[0]), int(pair[1])] for pair in pairs]
        return pairs


class StackH5(SLCStack):
    def __init__(self, stack, window=None, pol="VV", startDate=None, endDate=None):
        self.path = stack
        self.window = window
        self.slcs = np.array(sorted(glob.glob(os.path.join(stack, f'./*_{pol}_*.h5'))))
        self.configure_stack(startDate=startDate, endDate=endDate)

    def configure_stack(self, pol='VV', startDate=None, endDate=None):
        ## Get initial dates
        self.times = np.array([p.split('/')[-1].split('_')[4] for p in self.slcs])
        self.dates = np.array([t.split('T')[0] for t in self.times])
        self.datetimes = np.array([dt.strptime(d, '%Y%m%d') for d in self.dates])

        valid = np.ones(len(self.dates)).astype(bool)
        if startDate is not None:
            valid = self.datetimes >= startDate
        if endDate is not None:
            valid = valid & (self.datetimes <= endDate)
        self.slcs = self.slcs[valid]
        self.times = self.times[valid]
        self.dates = self.dates[valid]
        self.datetimes = self.datetimes[valid]
        self.P = len(self.slcs)

        ## Setup subdatasets
        self.subdatasets = [f'HDF5:"{path}"://data/{pol}' for path in self.slcs]
        self.vrt_path = os.path.join(self.path, 'stack.vrt')
        vrt = gdal.BuildVRT(self.vrt_path, self.subdatasets, separate=True)
        vrt = None

    def filter_dates(self, startDate, endDate):
        keep = np.argwhere(self.datetimes >= startDate & self.datetimes <= endDate)
        self.slcs = self.slcs[keep]  
        self.configure_stack()

    def read(self, index):
        with rasterio.open(self.vrt_path) as src:
            return src.read(index + 1, window=self.window)

    def write(self, data, output, dtype):
        with rasterio.open(self.subdatasets[0]) as src:

            profile = src.profile.copy()

            # Compute transform for subset if applicable
            if self.window is not None:
                transform = src.window_transform(self.window)
            else:
                transform = src.transform

            profile.update(
                driver="GTiff",
                dtype=dtype,
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                crs=src.crs,
                transform=transform.scale(1, -1),
                compress="lzw",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(output, "w", **profile) as dst:
                dst.write(data.astype(dtype), 1)


    def interferogram(self, pair, ml_size, sample_size, out_folder=None):
        assert len(pair) == 2, 'pair must be a list of two indices'
        assert len(ml_size) == 2, 'ml size must be a tuple of size 2'
        assert len(sample_size) == 2, 'sample size must be a tuple of size 2'
        with rasterio.open(self.vrt_path) as src:
            slcs = src.read([pair[0] + 1, pair[1] + 1], window=self.window)
            infgm = slcs[1] * slcs[0].conj()
        infgm = np.nan_to_num(infgm, 0)
        infgm = uniform_filter(infgm, size=ml_size)[::sample_size[0], ::sample_size[1]]
        if out_folder is not None:
            out_name = os.path.join(out_folder, f'{self.dates[pair[0]]}_{self.dates[pair[1]]}.infgm.ml')
            self.write(infgm, out_name, np.complex64)
            infgm = None
            return None
        else:
            return infgm


class StackTIF(SLCStack):
    def __init__(self, stack, window=None, pol="VV", startDate=None, endDate=None):
        self.path = stack
        self.window = window
        self.slcs = np.array(sorted(glob.glob(os.path.join(stack, f'./merged_*.tif'))))
        # self.slcs = np.array([str(f) for f in self.slcs])
        self.configure_stack(startDate=startDate, endDate=endDate)

    def configure_stack(self, pol='VV', startDate=None, endDate=None):
        ## Get initial dates
        print('getting dates')
        self.dates = np.array([p.split('/')[-1].split('_')[1].replace('.tif', '') for p in self.slcs])
        self.datetimes = np.array([dt.strptime(d, '%Y%m%d') for d in self.dates])
        valid = np.ones(len(self.dates)).astype(bool)
        # print(valid)
        if startDate is not None:
            valid = self.datetimes >= startDate
        if endDate is not None:
            valid = valid & (self.datetimes <= endDate)
        ## Filter stack
        self.slcs = self.slcs[valid]
        self.dates = self.dates[valid]
        self.datetimes = self.datetimes[valid]
        self.P = len(self.slcs)

        print('setting up stack')
        ## Setup subdatasets
        self.vrt_path = os.path.join(self.path, './stack.vrt')
        vrt = gdal.BuildVRT(self.vrt_path, list(self.slcs), separate=True)
        vrt = None

    def filter_dates(self, startDate, endDate):
        keep = np.argwhere(self.datetimes >= startDate & self.datetimes <= endDate)
        self.slcs = self.slcs[keep]  
        self.configure_stack()

    def read(self, index):
        with rasterio.open(self.vrt_path) as src:
            return src.read(index + 1, window=self.window)

    def write(self, data, output, dtype, sample_size = (1, 1)):
        with rasterio.open(self.slcs[0]) as src:
            profile = src.profile.copy()
            # Compute transform for subset if applicable
            if self.window is not None:
                transform = src.window_transform(self.window)
            else:
                transform = src.transform

             # Scale pixel size by `scale`
            transform = transform * Affine.scale(sample_size[1], sample_size[0])

            profile.update(
                driver="GTiff",
                dtype=dtype,
                transform=transform,
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                crs=src.crs,
            )

            with rasterio.open(output, "w", **profile) as dst:
                dst.write(data.astype(dtype), 1)


    def interferogram(self, pair, ml_size, sample_size, out_folder=None):
        assert len(pair) == 2, 'pair must be a list of two indices'
        assert len(ml_size) == 2, 'ml size must be a tuple of size 2'
        assert len(sample_size) == 2, 'sample size must be a tuple of size 2'
        with rasterio.open(self.vrt_path) as src:
            slcs = src.read([pair[0] + 1, pair[1] + 1], window=self.window)
            infgm = slcs[1] * slcs[0].conj()
        infgm = np.nan_to_num(infgm, 0)
        infgm = uniform_filter(infgm, size=ml_size)[::sample_size[0], ::sample_size[1]]
        if out_folder is not None:
            out_name = os.path.join(out_folder, f'{self.dates[pair[0]]}_{self.dates[pair[1]]}.infgm.ml.tif')
            self.write(infgm, out_name, np.complex64, sample_size=sample_size)
            infgm = None
            return None
        else:
            return infgm