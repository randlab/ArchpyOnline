import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import numpy as np
import flopy as fp
import os
import pandas as pd
import shutil


def _getpathlocal():
    # path to libraries folder
    path = os.path.join(os.path.dirname(__file__),'..','exe')
    return path


class DataRealisation():
    """Class that any type of forward data, with realisation on mutliple Facies, Strati or Prop"""

    def __init__(self, name, data, positions, stratIndex, faciesIndex, propIndex):
        self.name = name
        self.datas = data
        self.positions = positions
        self.stratIndex = stratIndex
        self.faciesIndex = faciesIndex
        self.propIndex = propIndex
        self.nmodel = len(stratIndex)

    def append(self, DataRealisation):
        self.datas.extend(DataRealisation.datas)
        self.stratIndex.extend(DataRealisation.stratIndex)
        self.faciesIndex.extend(DataRealisation.faciesIndex)
        self.propIndex.extend(DataRealisation.propIndex)
        self.nmodel = self.nmodel + len(DataRealisation.stratIndex)


class TEM():
    """Class that contains TEM data"""

    def __init__(self, name, gates, values, positions, std=None):

        self.name = name
        self.gates = gates
        self.values = values
        self.positions = positions
        self.std = std

    def add_data(self, positions, values, gates):
        """This function add TEM data to a TEM object.

        positions : 2D array [n_data,3] - x,y,z position of sounding
        values : list of 2 [2D array HM [n_data,n_gates], 2D array LM [n_data,n_gates]]
        gates : list of 2 1D array [n_gates_hm,n_gates_lm]

        """

        if positions.shape[1] != 3:
            assert ('Size error. positions should be of size [n_data,3]')
        if values.shape[1] != len(gates) or values.shape[0] != positions.shape[0]:
            assert ('Size error. values should be of size [n_data,n_gates]')

        self.gates.append(gates)
        self.values.append(values)
        self.positions.append(positions)

    def plot(self, idx=0):
        plt.loglog(self.gates[0], self.values[0][idx], 'x-', label='HM')
        plt.loglog(self.gates[1], self.values[1][idx], 'x-', label='LM')
        plt.grid('on')
        plt.legend()
        plt.xlabel('time [s]')
        plt.ylabel('dB/dt')

    def plot_rhoa(self, idx=0):
        mu0 = 4e-7 * math.pi
        rhoaHM = (1 * 8 * mu0 / 20. / self.values[0]) ** (2. / 3.) * self.gates[0] ** (-5. / 3.) * mu0 / math.pi
        rhoaLM = (1 * 8 * mu0 / 20. / self.values[1]) ** (2. / 3.) * self.gates[1] ** (-5. / 3.) * mu0 / math.pi

        plt.loglog(self.gates[0], rhoaHM[idx], '--', label='HM')
        plt.loglog(self.gates[1], rhoaLM[idx], '--', label='LM')
        plt.grid('on')
        plt.legend()
        plt.xlabel('time [ns]')
        plt.ylabel('apparent rho [ohmm]')

        return

    def rhoa(self):
        mu0 = 4e-7 * math.pi
        rhoaHM = (1 * 8 * mu0 / 20. / self.values[0]) ** (2. / 3.) * self.gates[0] ** (-5. / 3.) * mu0 / math.pi
        rhoaLM = (1 * 8 * mu0 / 20. / self.values[1]) ** (2. / 3.) * self.gates[1] ** (-5. / 3.) * mu0 / math.pi
        return rhoaHM, rhoaLM

    def misfit(self, data):
        # clark distance
        return np.sqrt(
            np.nansum(((self.values[0] - data.values[0]) / (self.values[0] + data.values[0])) ** 2) + np.nansum(
                ((self.values[1] - data.values[1]) / (self.values[1] + data.values[1])) ** 2)) / (
                       np.sum(~np.isnan(data.values[1])) + np.sum(~np.isnan(data.values[0])))

    def point_misfit(self, data):
        # clark distance
        return np.sqrt(np.nansum(((self.values[0] - data.values[0]) / (self.values[0] + data.values[0])) ** 2, axis=1) + np.nansum(
                ((self.values[1] - data.values[1]) / (self.values[1] + data.values[1])) ** 2, axis=1)) / (
                       np.sum(~np.isnan(data.values[1])) + np.sum(~np.isnan(data.values[0])))

    def likelihood(self, data):
        return np.exp(
            -(np.sum((self.values[0] - data.values[0]) ** 2) + np.sum((self.values[1] - data.values[1]) ** 2)))

    def log_likelihood(self, data):
        return np.exp(
            -(np.sum((self.values[0] - data.values[0]) ** 2) + np.sum((self.values[1] - data.values[1]) ** 2)))

    def likelihood_rhoa(self, data):
        return np.exp(
            -(np.sum((self.rhoa()[0] - data.rhoa()[0]) ** 2) + np.sum((self.rhoa()[1] - data.rhoa()[1]) ** 2)))


class DC():
    """Class that contains DC resisitivity data"""

    def __init__(self, name, quadipoles, apparent_res, positions):
        self.name = name
        self.quadripoles = np.array(quadipoles)
        self.apparent_res = apparent_res
        self.positions = positions

    def plot(self):
        quadripoles = np.array(self.quadripoles.copy())

        AB2 = np.abs(quadripoles[:, 0] - quadripoles[:, 1]) / 2
        rx_midpoints = (quadripoles[:, 2] + quadripoles[:, 3]) / 2
        plt.scatter(rx_midpoints, AB2, 20, self.apparent_res, norm=LogNorm())

        plt.gca().invert_yaxis()
        a = plt.colorbar()
        a.ax.set_ylabel('Apparent Rho [ohm]', rotation=270)
        plt.ylabel('Pseudodepth AB/2')
        plt.xlabel('Electrode')


class Grav():
    """Class that contains DC resisitivity data"""

    def __init__(self, name, anomaly, positions):
        self.name = name
        self.Bougeranomaly = anomaly
        self.positions = positions

    def plot(self):
        plt.scatter(self.positions[:, 0], self.positions[:, 1], 10, self.Bougeranomaly)


class Heads_HD():
    """Class that Heads Hard data to be compared to an hydro model"""

    def __init__(self, x, y, z, head):
        '''x and y are the cell position. Z can be a value or an array, and is the open section of the borehole. Heads is the head.
        Multiple data can be provided as a list of muliple [x,y,z,head]'''

        self.x = x
        self.y = y
        self.z = z
        self.head = np.array(head)

    def likelihood(self, data):
        ref_Data = np.array(data[self.z, self.y, self.x])
        return np.exp(-(np.sum((self.head - ref_Data) ** 2)))

    def misfit(self, data):
        # clark distance
        ref_Data = np.array(data[self.z, self.y, self.x])

        return np.sqrt(np.sum(((self.head - ref_Data) / (self.head + ref_Data)) ** 2)) / len(ref_Data)


class HydroModel():
    """Class that contains a modflow model"""

    def __init__(self, name, path_to_workspace):
        self.name = name
        self.path_to_workspace = path_to_workspace

        self.sim_dr = path_to_workspace + "/../workspace_run"
        try:
            shutil.copytree(path_to_workspace, self.sim_dr)
        except:
            shutil.rmtree(self.sim_dr, self.name)
            shutil.copytree(path_to_workspace, self.sim_dr)
        if os.name == 'nt':
            self.sim = fp.mf6.MFSimulation.load(name + ".nam", sim_ws=self.sim_dr, exe_name=_getpathlocal() + "/mf6.exe")
        else:
            self.sim = fp.mf6.MFSimulation.load(name + ".nam", sim_ws=self.sim_dr, exe_name=_getpathlocal() + "/mf6")

        return

    def get_heads(self, kstpkper=(0, 0)):
        hfile = os.path.join(self.sim_dr, self.name + ".hds")
        return fp.utils.HeadFile(hfile).get_data(kstpkper)

    def get_K(self):
        mod = self.sim.get_model()
        return mod.npf.k

    def update_K(self, newK):
        mod = self.sim.get_model()
        mod.npf.k = np.flipud(10 ** newK)
        mod.npf.write()
        return

    def run_sim(self):
        self.sim.run_simulation()


class TransientHydroModel():
    """Class that contains a modflow transient model"""

    def __init__(self, name, path_to_workspace, imod=0):
        self.name = name
        self.path_to_workspace = path_to_workspace

        import shutil
        self.sim_dr = path_to_workspace + "/../"+self.name+"_run_" + str(imod)
        try:
            shutil.copytree(path_to_workspace, self.sim_dr)
        except:
            shutil.rmtree(self.sim_dr, self.name)
            shutil.copytree(path_to_workspace, self.sim_dr)

        if os.name == 'nt':
            self.sim = fp.mf6.MFSimulation.load(name + ".nam", sim_ws=self.sim_dr, exe_name=_getpathlocal() + "/mf6.exe", verbosity_level=0)
        else:
            self.sim = fp.mf6.MFSimulation.load(name + ".nam", sim_ws=self.sim_dr, exe_name=_getpathlocal() + "/mf6", verbosity_level=0)

        return

    def get_heads(self, kstpkper=(0, 0)):
        hfile = os.path.join(self.sim_dr, self.name + ".hds")
        return fp.utils.HeadFile(hfile).get_data(kstpkper)

    def get_K(self):
        mod = self.sim.get_model()
        return mod.npf.k

    def update_K(self, newK):
        mod = self.sim.get_model()
        mod.npf.k = np.flipud(10 ** newK)
        mod.npf.write()
        return

    def run_sim(self, silent=True):
        return self.sim.run_simulation(silent)

    def get_observationfile(self, name="head_obs.csv"):
        head_obs_df = pd.read_csv(os.path.join(self.sim_dr, name))

        def clean(val):
            try:
                val = float(val)
            except:
                val = float(val[:-4] + "E" + val[-4:])
            return val

        head_obs_df = head_obs_df.set_index("time").applymap(clean).astype(float)
        return head_obs_df

    def clean_files(self):
        shutil.rmtree(self.sim_dr, self.name)
        return


def generate_WS(nelectr):
    quadripoles = []
    for i in range(1, nelectr + 1 - 3):
        a = i
        if (a % 2) == 0:
            ds = 1
        else:
            ds = 0

        for b in range(nelectr - ds, a + 2, -2):
            m = int(np.floor((b - a) / 2 + a))
            n = int(np.ceil((b - a) / 2 + a))
            quadripoles.append([a, b, m, n])
    return quadripoles
