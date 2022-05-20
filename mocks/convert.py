import h5py
import numpy as np
from astropy.table import Table

fstream = h5py.File(
    '/project/projectdirs/desi/cosmosim/FirstGenMocks/Uchuu/CubicBox/BGS/' +
    'z0.190/BGS_box_Uchuu.hdf5', 'r')

abs_mag = fstream['Data/abs_mag'][()]
mask = abs_mag < -19
abs_mag = abs_mag[mask]
g_r = fstream['Data/g_r'][()][mask]
cen = fstream['Data/galaxy_type'][()][mask] == 0
pos = fstream['Data/pos'][()][mask]
vel = fstream['Data/vel'][()][mask]

table = Table()
table['M'] = abs_mag.astype(np.float32)
table['g-r'] = g_r.astype(np.float32)
table['cen'] = cen
table['x'] = pos[:, 0].astype(np.float32)
table['y'] = pos[:, 1].astype(np.float32)
table['z'] = pos[:, 2].astype(np.float32)
table['vx'] = vel[:, 0].astype(np.float32)
table['vy'] = vel[:, 1].astype(np.float32)
table['vz'] = vel[:, 2].astype(np.float32)
table.write('uchuu_bgs.hdf5', path='mock', overwrite=True)
fstream.close()
