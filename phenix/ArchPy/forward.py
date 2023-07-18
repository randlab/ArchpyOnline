import numpy as np
import subprocess
import multiprocessing
from ArchPy.base import *
from ArchPy.databox import *
import os
import inspect
import json
import empymod

from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from SimPEG import maps
from SimPEG.utils import surface2ind_topo
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import apparent_resistivity_from_voltage
from SimPEG.potential_fields import gravity

from shutil import rmtree
from distutils.dir_util import copy_tree


def _getpathlocal():
    # path to libraries folder
    path = os.path.normpath(os.path.dirname(__file__) + '/../libraries')
    return path


def physicsforward(Arch_table, method, positions, stratIndex=None, faciesIndex=None, propIndex=None, idx=0, cpuID=0):
    """Function to go from a 3D model to 1D models"""

    # We want to work with array, so in case a int is passed, we make a 1D array of it

    if type(stratIndex) is int:
        stratIndex = np.array([stratIndex])
    if type(faciesIndex) is int:
        faciesIndex = np.array([faciesIndex])
    if type(propIndex) is int:
        propIndex = np.array([propIndex])

    if type(stratIndex) is type(None):
        stratIndex = np.arange(Arch_table.nreal_units)
    if type(faciesIndex) is type(None):
        faciesIndex = np.arange(Arch_table.nreal_fa)
    if type(propIndex) is type(None):
        propIndex = np.arange(Arch_table.nreal_prop)
    # Identify wich method is asked for

    if method == 'tTEM':
        models1D = _3Dto1D(10 ** Arch_table.getprop('logrho'), Arch_table.get_zgc(), positions, stratIndex, faciesIndex,
                           propIndex)
        dataholder_HM, Gates_HM, dataholder_LM, Gates_LM = TEMAarhusInv(models1D, idx,cpuID)
        databox = _1DTEMtoDatabox(dataholder_HM, Gates_HM, dataholder_LM, Gates_LM, positions, stratIndex, faciesIndex,
                                  propIndex)

    elif method == 'walkTEM':
        models1D = _3Dto1D(10 ** Arch_table.getprop('logrho'), Arch_table.get_zgc(), positions, stratIndex, faciesIndex,
                           propIndex)
        dataholder_HM, Gates_HM, dataholder_LM, Gates_LM = walkTEM(models1D)
        databox = _1DTEMtoDatabox(dataholder_HM, Gates_HM, dataholder_LM, Gates_LM, positions, stratIndex, faciesIndex,
                                  propIndex)

    elif method == 'DC':
        positions, quadripoles = positions
        mesh, topo_xyz = _3Dto3DSP(Arch_table)
        survey, mesh = _ERT_survey(mesh, positions, quadripoles, topo_xyz)
        dataholder = _simulate_ert(Arch_table, mesh, survey, topo_xyz, stratIndex, faciesIndex, propIndex)
        databox = _3DDCtoDatabox(dataholder, positions, quadripoles, stratIndex, faciesIndex, propIndex)

    elif method == 'DC2d':
        positions, quadripoles = positions
        dataholder = _3DtoDC2D(Arch_table, positions, quadripoles, stratIndex, faciesIndex, propIndex)
        databox = _3DDCtoDatabox(dataholder, positions, quadripoles, stratIndex, faciesIndex, propIndex)

    elif method == 'GRAV':
        mesh, topo_xyz = _3Dto3DSP(Arch_table)
        survey = _gravi_survey(mesh, positions, topo_xyz)
        dataholder = _simulate_gravi(survey, mesh, Arch_table, topo_xyz, stratIndex, faciesIndex, propIndex)
        databox = _3DGravtoDatabox(dataholder, positions, stratIndex, faciesIndex, propIndex)
    else:
        databox = None
    # if method == 'TEMPymod':

    return databox


def _3DDCtoDatabox(dataholder, positions, quadripoles, stratIndex, faciesIndex, propIndex):
    data = []
    for i in range(len(dataholder)):
        data.append(DC('model ' + str(i), quadripoles, dataholder[i], positions))
    Forward5D = DataRealisation('Forward_DC_1', data, positions, stratIndex, faciesIndex, propIndex)
    return Forward5D


def _3DGravtoDatabox(dataholder, positions, stratIndex, faciesIndex, propIndex):
    data = []
    for i in range(len(dataholder)):
        data.append(Grav('model ' + str(i), dataholder[i], positions))
    Forward5D = DataRealisation('Forward_Grav_1', data, positions, stratIndex, faciesIndex, propIndex)
    return Forward5D


def _1DTEMtoDatabox(dataholder_HM, Gates_HM, dataholder_LM, Gates_LM, positions, stratIndex, faciesIndex, propIndex):
    idx_start = 0
    ndata = len(positions)
    data = []
    all_strat = []
    all_facies = []
    all_prop = []
    for strat in range(len(stratIndex)):
        for facies in range(len(faciesIndex)):
            for prop in range(len(propIndex)):
                idx_end = idx_start + ndata
                data.append(TEM('model' + str(strat) + str(facies) + str(prop), [Gates_HM, Gates_LM],
                                [dataholder_HM[idx_start:idx_end], dataholder_LM[idx_start:idx_end]], positions))
                idx_start = idx_end

                all_strat.append(strat)
                all_facies.append(facies)
                all_prop.append(prop)
    Forward5D = DataRealisation('Forward_TEM_1', data, positions, all_strat, all_facies, all_prop)
    return Forward5D


def _3Dto1D(propArray, zg, positions, stratIndex, faciesIndex, propIndex):
    """Function to go from a 3D model to 1D models"""
    nmodels = positions.shape[0] * len(stratIndex) * len(faciesIndex) * len(propIndex)
    models1D = [0] * nmodels

    model_idx = 0
    for strat in stratIndex:
        for facies in faciesIndex:
            for prop in propIndex:
                for position in positions:
                    model1D = propArray[strat, facies, prop, :, position[1], position[0]]

                    depth = zg[~np.isnan(model1D)]
                    thickness = np.flipud((np.diff(depth - depth[0])))
                    model1D = np.flipud(model1D[~np.isnan(model1D)])

                    models1D[model_idx] = [model1D, thickness]
                    model_idx += 1

    return models1D


def _3DtoDC2D(Arch_table, positions, quadripoles, stratIndex, faciesIndex, propIndex):
    def generate_mesh(nx, nz, dx, dz):
        from discretize import TensorMesh

        hx = dx * np.ones(nx)
        hz = dz * np.ones(nz)
        mesh = TensorMesh([hx, hz], origin=[0, nz * -dz])
        return mesh

    min_res = min(Arch_table.get_sx(), Arch_table.get_sy())
    nz = Arch_table.get_nz()
    dz = Arch_table.get_sz()
    line_start = positions[0]
    line_end = positions[-1]
    n_elec = len(positions)
    spacingx = (line_end[0] - line_start[0]) / (n_elec - 1)
    spacingy = (line_end[1] - line_start[1]) / (n_elec - 1)
    t_dist = np.sqrt((line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2)
    spacing = t_dist / (n_elec - 1)
    dx = min(spacing / 2, min_res)
    nx = int(np.ceil(t_dist / dx))
    # for i in range(n_elec):
    mesh = generate_mesh(nx, nz, dx, dz)

    electrodes_alongline = np.arange(n_elec) * spacing
    x_alongline = np.arange(n_elec) * spacingx + line_start[0]
    y_alongline = np.arange(n_elec) * spacingy + line_start[1]

    x_alongline_cell = np.round(np.arange(n_elec) * spacingx + line_start[0]).astype(int)
    y_alongline_cell = np.round(np.arange(n_elec) * spacingy + line_start[1]).astype(int)

    x_cell_grid = (mesh.cell_centers[:, 0] / spacing * spacingx + line_start[0]).astype(int)
    y_cell_grid = (mesh.cell_centers[:, 0] / spacing * spacingy + line_start[1]).astype(int)

    topo2d = np.array([electrodes_alongline.T, Arch_table.top[y_alongline_cell, x_alongline_cell].T]).T
    source_list = []
    for quadripole in quadripoles:
        # AB electrode locations for source. Each is a (1, 3) numpy array
        A_location = electrodes_alongline[int(quadripole[0] - 1)]
        B_location = electrodes_alongline[int(quadripole[1] - 1)]

        # MN electrode locations for receivers. Each is an (N, 3) numpy array
        M_location = electrodes_alongline[int(quadripole[2] - 1)]
        N_location = electrodes_alongline[int(quadripole[3] - 1)]

        # Create receivers list. Define as pole or dipole.
        receiver_list = dc.receivers.Dipole(M_location, N_location)
        receiver_list = [receiver_list]

        # Define the source properties and associated receivers
        source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

    # Define survey
    survey = dc.survey.Survey(source_list)
    ind_active = surface2ind_topo(mesh, topo2d)

    survey.drape_electrodes_on_topography(mesh, ind_active, force=True, topography=topo2d)

    electrode_locations = np.r_[survey.locations_a, survey.locations_b, survey.locations_m, survey.locations_n]
    centroids = mesh.cell_centers
    z_cell = (centroids[:, 1] - min(centroids[:, 1])).astype(int)  # a controler
    # C = np.ones(conductivity_map.nC) * 40
    propArray = 10 ** Arch_table.getprop('logrho')

    conductivity_map = maps.InjectActiveCells(mesh, ind_active, 1 / 1e-8)
    models = []
    for strat in stratIndex:
        for facies in faciesIndex:
            for prop in propIndex:
                rhomodel = propArray[strat, facies, prop, z_cell, y_cell_grid, x_cell_grid]
                rhomodel = rhomodel[ind_active]

                simulation = dc.simulation.Simulation3DNodal(mesh, survey=survey, rhoMap=conductivity_map)
                rhomodel[np.isnan(rhomodel)] = 0
                dpred = simulation.dpred(rhomodel)
                models.append(apparent_resistivity_from_voltage(survey, dpred))
    return models


def _3Dto3DSP(Arch_table):
    def generate_mesh(nx, ny, nz, dx, dy, dz):
        from discretize import TensorMesh

        hx = dx * np.ones(nx)
        hy = dy * np.ones(ny)
        hz = dz * np.ones(nz)
        mesh = TensorMesh([hx, hy, hz], origin=[0, 0, nz * -dz])
        return mesh

    mesh = generate_mesh(Arch_table.get_nx(), Arch_table.get_ny(), Arch_table.get_nz(), Arch_table.get_sx(),
                         Arch_table.get_sy(), Arch_table.get_sz())
    y, x = np.meshgrid(mesh.cell_centers_y, mesh.cell_centers_x)
    topo_xyz = np.c_[x.flatten(), y.flatten(), Arch_table.top.flatten()]

    return mesh, topo_xyz


def _ERT_survey(mesh, positions, quadripoles, topo_xyz):
    positions = np.sort(positions, axis=0)

    source_list = []
    for quadripole in quadripoles:
        # AB electrode locations for source. Each is a (1, 3) numpy array
        A_location = positions[int(quadripole[0] - 1)][:2]
        B_location = positions[int(quadripole[1] - 1)][:2]

        # MN electrode locations for receivers. Each is an (N, 3) numpy array
        M_location = positions[int(quadripole[2] - 1)][:2]
        N_location = positions[int(quadripole[3] - 1)][:2]

        # Create receivers list. Define as pole or dipole.
        receiver_list = dc.receivers.Dipole(M_location, N_location)
        receiver_list = [receiver_list]

        # Define the source properties and associated receivers
        source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

    # Define survey
    survey = dc.survey.Survey(source_list)
    ind_active = surface2ind_topo(mesh, topo_xyz)

    survey.drape_electrodes_on_topography(mesh, ind_active, force=True)

    electrode_locations = np.r_[survey.locations_a, survey.locations_b, survey.locations_m, survey.locations_n]
    unique_locations = np.unique(electrode_locations, axis=0)

    return survey, mesh


def _simulate_ert(Arch_table, mesh, survey, topo_xyz, stratIndex, faciesIndex, propIndex):
    def assignprop(mesh, prop, topo_xyz, Arch_Table, unactive_value):
        ind_active = surface2ind_topo(mesh, topo_xyz)

        centroids = mesh.cell_centers
        cell_x, cell_y, cell_z = Arch_Table.indextocell(centroids[:, 0], centroids[:, 1], centroids[:, 2])

        prop_array = prop[cell_z, cell_y, cell_x]

        mod_map = maps.InjectActiveCells(mesh, ind_active, unactive_value)
        return prop_array[ind_active], mod_map

    models = []
    air_resisitivity = 1 / 1e-8
    propArray = 10 ** Arch_table.getprop('logrho')
    for strat in stratIndex:
        for facies in faciesIndex:
            for prop in propIndex:
                rhomodel, rho_map = assignprop(mesh, propArray[strat, facies, prop, :, :, :], topo_xyz, Arch_table,
                                               air_resisitivity)
                simulation = dc.simulation.Simulation3DNodal(mesh, survey=survey, rhoMap=rho_map)
                rhomodel[np.isnan(rhomodel)] = 0
                dpred = simulation.dpred(rhomodel)
                models.append(apparent_resistivity_from_voltage(survey, dpred))

    return models


def _gravi_survey(mesh, positions, topo_xyz):
    components = ["gz"]
    receiver_list = gravity.receivers.Point(positions, components=components)
    receiver_list = [receiver_list]
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)
    survey = gravity.survey.Survey(source_field)

    return survey


def _simulate_gravi(survey, mesh, Arch_table, topo_xyz, stratIndex, faciesIndex, propIndex):
    def assignprop_grav(mesh, prop, topo_xyz, Arch_table):
        ind_active = surface2ind_topo(mesh, topo_xyz)

        centroids = mesh.cell_centers
        cell_x = ((centroids[:, 0] - Arch_table.get_ox()) / Arch_table.get_sx()).astype(int)
        cell_y = ((centroids[:, 1] - Arch_table.get_oy()) / Arch_table.get_sy()).astype(int)
        cell_z = ((centroids[:, 2] - Arch_table.get_oz()) / Arch_table.get_sz()).astype(int)
        prop_array = prop[cell_z, cell_y, cell_x]

        nC = int(ind_active.sum())
        mod_map = maps.IdentityMap(nP=nC)

        return prop_array[ind_active], mod_map

    dpreds = []
    propArray = Arch_table.getprop('density')
    for strat in stratIndex:
        for facies in faciesIndex:
            for prop in propIndex:
                model, model_map = assignprop_grav(mesh, propArray[strat, facies, prop, :, :, :], topo_xyz, Arch_table)

                ind_active = surface2ind_topo(mesh, topo_xyz)
                simulation = gravity.simulation.Simulation3DIntegral(
                    survey=survey,
                    mesh=mesh,
                    rhoMap=model_map,
                    actInd=ind_active,
                    store_sensitivities="forward_only",
                )
                dpreds.append(simulation.dpred(model))

    return dpreds


def TEMAarhusInv(models1D, idx, cpuID):
    """
    TEM is a function that uses the AarhusInv executable to forward TEM data. Be sure that the excecutable is activated and that the proper license is available. The HM and LM are calculated each time.
    """
    # subfunctions for model writing
    idx = str(idx)

    def init_folder(idx):
        copy_tree(os.path.join(_getpathlocal(), 'AarhusINV', 'input'),
                  os.path.join(_getpathlocal(), 'AarhusINV', 'input' + idx))

    def writeHeaders(nmodels, file):
        file.write('ReferenceModel for forward calc.\n')
        file.write(str(int(nmodels * 2)) + ' 0\n')
        for i in range(0, nmodels * 2, 2):
            file.write(str(int(i + 1)) + ' 1 HighMoment.tem\n')
            file.write(str(int(i + 2)) + ' 1 LowMoment.tem\n')
        file.write('-1\n')
        return

    def writeLineScientific(Value, file):
        txt = "{:e} {:e}\n"
        file.write(txt.format(Value, -1))

        return

    def writeModel(thicknesses, res, nlay, file):
        depth = thicknesses.copy()
        depth = np.cumsum(depth)
        file.write(str(nlay) + '\n')
        for i in range(nlay):
            writeLineScientific(res[i], file)
        for i in range(nlay - 1):
            writeLineScientific(thicknesses[i], file)
        for i in range(nlay - 1):
            writeLineScientific(depth[i], file)
        return

    def loadModel(nmodels, idx, n_threads = 1):

        # load HM once to see how many gates we have
        HM = np.loadtxt(os.path.join(_getpathlocal(), 'AarhusINV', 'input' + idx, 'model1' + str(1).zfill(5) + '.fwr'),
                        skiprows=17)
        dataholder_HM = np.empty((nmodels, len(HM[:, 1])))

        for i in range(nmodels):
            HM = np.loadtxt(
                os.path.join(_getpathlocal(), 'AarhusINV', 'input' + idx, 'model1' + str(i * 2 + 1).zfill(5) + '.fwr'),
                skiprows=17)
            dataholder_HM[i, :] = HM[:, 1]
            os.remove(_getpathlocal() + '/AarhusINV/input' + idx + '/model1' + str(i * 2 + 1).zfill(5) + '.fwr')
        Gates_HM = HM[:, 0]

        # load LM once to see how many gates we have
        LM = np.loadtxt(_getpathlocal() + '/AarhusINV/input' + idx + '/model1' + str((i + 1) * 2).zfill(5) + '.fwr',
                        skiprows=17)
        dataholder_LM = np.empty((nmodels, len(LM[:, 1])))

        for i in range(nmodels):
            LM = np.loadtxt(_getpathlocal() + '/AarhusINV/input' + idx + '/model1' + str((i + 1) * 2).zfill(5) + '.fwr',
                            skiprows=17)
            dataholder_LM[i, :] = LM[:, 1]
            os.remove(_getpathlocal() + '/AarhusINV/input' + idx + '/model1' + str((i + 1) * 2).zfill(5) + '.fwr')
        Gates_LM = LM[:, 0]

        return dataholder_HM, Gates_HM, dataholder_LM, Gates_LM

    init_folder(idx)
    nmodels = len(models1D)
    with open(os.path.join(_getpathlocal(), 'AarhusINV','input'+idx,'model1.mod'), 'w') as f:
        writeHeaders(nmodels, f)
        for i in range(nmodels):
            # HM
            writeModel(models1D[i][1], models1D[i][0], len(models1D[i][0]), f)
            # LM
            writeModel(models1D[i][1], models1D[i][0], len(models1D[i][0]), f)

    if os.name == 'nt':
        a = subprocess.run('"'+_getpathlocal() + '\\AarhusINV\\AarhusInv64.exe" "' + _getpathlocal() + '\\AarhusINV\\input' + idx + '\\model1.mod" "' + _getpathlocal() + '\\AarhusINV\\input' + idx + '\\AarhusInv.con"',check=True)
    if os.name == 'posix':
        command = "numactl --physcpubind=" + str(cpuID) + " wine64 AarhusInv64.exe input" + idx + "/model1.mod input" + idx + "/AarhusInv.con"
        a = subprocess.run(command,check=False, cwd=_getpathlocal()+'/AarhusINV',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    data = loadModel(nmodels, idx)
    rmtree(os.path.join(_getpathlocal(), 'AarhusINV', 'input' + idx))

    return data


def walkTEM(models1D):
    '''This section of the code is a implementation of WalkTEM simulation using EMpiMOD. The code is an adaptation of
    their exemple code available on empymod.emsig.xyz'''
    nmodels = len(models1D)
    with open(_getpathlocal() + '/walkTEM/walkTEM.params') as f:
        data = json.load(f)

    hm_off_time = np.array(data['hm_off_time'])
    lm_off_time = np.array(data['lm_off_time'])

    lm_waveform_times = np.r_[data['lm_waveform_times']]
    lm_waveform_current = np.r_[data['lm_waveform_current']]

    hm_waveform_times = np.r_[data['hm_waveform_times']]
    hm_waveform_current = np.r_[data['hm_waveform_current']]

    def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
        """Apply a source waveform to the signal.

        Parameters
        ----------
        times : ndarray
            Times of computed input response; should start before and end after
            `times_wanted`.

        resp : ndarray
            EM-response corresponding to `times`.

        times_wanted : ndarray
            Wanted times.

        wave_time : ndarray
            Time steps of the wave.

        wave_amp : ndarray
            Amplitudes of the wave corresponding to `wave_time`, usually
            in the range of [0, 1].

        nquad : int
            Number of Gauss-Legendre points for the integration. Default is 3.

        Returns
        -------
        resp_wanted : ndarray
            EM field for `times_wanted`.

        """

        # Interpolate on log.
        PP = iuSpline(np.log10(times), resp)

        # Wave time steps.
        dt = np.diff(wave_time)
        dI = np.diff(wave_amp)
        dIdt = dI / dt

        # Gauss-Legendre Quadrature; 3 is generally good enough.
        # (Roots/weights could be cached.)
        g_x, g_w = roots_legendre(nquad)

        # Pre-allocate output.
        resp_wanted = np.zeros_like(times_wanted)

        # Loop over wave segments.
        for i, cdIdt in enumerate(dIdt):

            # We only have to consider segments with a change of current.
            if cdIdt == 0.0:
                continue

            # If wanted time is before a wave element, ignore it.
            ind_a = wave_time[i] < times_wanted
            if ind_a.sum() == 0:
                continue

            # If wanted time is within a wave element, we cut the element.
            ind_b = wave_time[i + 1] > times_wanted[ind_a]

            # Start and end for this wave-segment for all times.
            ta = times_wanted[ind_a] - wave_time[i]
            tb = times_wanted[ind_a] - wave_time[i + 1]
            tb[ind_b] = 0.0  # Cut elements

            # Gauss-Legendre for this wave segment. See
            # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
            # for the change of interval, which makes this a bit more complex.
            logt = np.log10(np.outer((tb - ta) / 2, g_x) + (ta + tb)[:, None] / 2)
            fact = (tb - ta) / 2 * cdIdt
            resp_wanted[ind_a] += fact * np.sum(np.array(PP(logt) * g_w), axis=1)

        return resp_wanted

    def get_time(time, r_time):
        """Additional time for ramp.

        Because of the arbitrary waveform, we need to compute some times before and
        after the actually wanted times for interpolation of the waveform.

        Some implementation details: The actual times here don't really matter. We
        create a vector of time.size+2, so it is similar to the input times and
        accounts that it will require a bit earlier and a bit later times. Really
        important are only the minimum and maximum times. The Fourier DLF, with
        `pts_per_dec=-1`, computes times from minimum to at least the maximum,
        where the actual spacing is defined by the filter spacing. It subsequently
        interpolates to the wanted times. Afterwards, we interpolate those again to
        compute the actual waveform response.

        Note: We could first call `waveform`, and get the actually required times
              from there. This would make this function obsolete. It would also
              avoid the double interpolation, first in `empymod.model.time` for the
              Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
              Probably not or marginally faster. And the code would become much
              less readable.

        Parameters
        ----------
        time : ndarray
            Desired times

        r_time : ndarray
            Waveform times

        Returns
        -------
        time_req : ndarray
            Required times
        """
        tmin = np.log10(max(time.min() - r_time.max(), 1e-10))
        tmax = np.log10(time.max() - r_time.min())
        return np.logspace(tmin, tmax, time.size + 2)

    def walktem(moment, depth, res):
        """Custom wrapper of empymod.model.bipole.

        Here, we compute WalkTEM data using the ``empymod.model.bipole`` routine as
        an example. We could achieve the same using ``empymod.model.dipole`` or
        ``empymod.model.loop``.

        We model the big source square loop by computing only half of one side of
        the electric square loop and approximating the finite length dipole with 3
        point dipole sources. The result is then multiplied by 8, to account for
        all eight half-sides of the square loop.

        The implementation here assumes a central loop configuration, where the
        receiver (1 m2 area) is at the origin, and the source is a 40x40 m electric
        loop, centered around the origin.

        Note: This approximation of only using half of one of the four sides
              obviously only works for central, horizontal square loops. If your
              loop is arbitrary rotated, then you have to model all four sides of
              the loop and sum it up.


        Parameters
        ----------
        moment : str {'lm', 'hm'}
            Moment. If 'lm', above defined ``lm_off_time``, ``lm_waveform_times``,
            and ``lm_waveform_current`` are used. Else, the corresponding
            ``hm_``-parameters.

        depth : ndarray
            Depths of the resistivity model (see ``empymod.model.bipole`` for more
            info.)

        res : ndarray
            Resistivities of the resistivity model (see ``empymod.model.bipole``
            for more info.)

        Returns
        -------
        WalkTEM : EMArray
            WalkTEM response (dB/dt).

        """

        # Get the measurement time and the waveform corresponding to the provided
        # moment.
        if moment == 'lm':
            off_time = lm_off_time
            waveform_times = lm_waveform_times
            waveform_current = lm_waveform_current
        elif moment == 'hm':
            off_time = hm_off_time
            waveform_times = hm_waveform_times
            waveform_current = hm_waveform_current
        else:
            raise ValueError("Moment must be either 'lm' or 'hm'!")

        # === GET REQUIRED TIMES ===
        time = get_time(off_time, waveform_times)

        # === GET REQUIRED FREQUENCIES ===
        time, freq, ft, ftarg = empymod.utils.check_time(
            time=time,  # Required times
            signal=1,  # Switch-on response
            ft='dlf',  # Use DLF
            ftarg={'dlf': 'key_81_CosSin_2009'},  # Short, fast filter; if you
            verb=1,  # need higher accuracy choose a longer filter.
        )

        # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
        # We only define a few parameters here. You could extend this for any
        # parameter possible to provide to empymod.model.bipole.
        EM = empymod.model.bipole(
            src=[20, 20, 0, 20, 0, 0],  # El. bipole source; half of one side.
            rec=[0, 0, 0, 0, 90],  # Receiver at the origin, vertical.
            depth=np.r_[0, depth],  # Depth-model, adding air-interface.
            res=np.r_[2e14, res],  # Provided resistivity model, adding air.
            # aniso=aniso,                # Here you could implement anisotropy...
            #                             # ...or any parameter accepted by bipole.
            freqtime=freq,  # Required frequencies.
            mrec=True,  # It is an el. source, but a magn. rec.
            strength=8,  # To account for 4 sides of square loop.
            srcpts=3,  # Approx. the finite dip. with 3 points.
            htarg={'dlf': 'key_101_2009'},
            verb=1  # Short filter, so fast.
        )

        # Multiply the frequecny-domain result with
        # \mu for H->B, and i\omega for B->dB/dt.
        EM *= 2j * np.pi * freq * 4e-7 * np.pi

        # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
        # Note: Here we just apply one filter. But it seems that WalkTEM can apply
        #       two filters, one before and one after the so-called front gate
        #       (which might be related to ``delay_rst``, I am not sure about that
        #       part.)
        cutofffreq = 4.5e5  # As stated in the WalkTEM manual
        h = (1 + 1j * freq / cutofffreq) ** -1  # First order type
        h *= (1 + 1j * freq / 3e5) ** -1
        EM *= h

        # === CONVERT TO TIME DOMAIN ===
        delay_rst = 1.8e-7  # As stated in the WalkTEM manual
        EM, _ = empymod.model.tem(EM[:, None], np.array([1]),
                                  freq, time + delay_rst, 1, ft, ftarg)
        EM = np.squeeze(EM)

        # === APPLY WAVEFORM ===
        return waveform(time, EM, off_time, waveform_times, waveform_current)

    dataholder_HM = np.empty((nmodels, len(hm_off_time)))
    dataholder_LM = np.empty((nmodels, len(lm_off_time)))

    for i in range(nmodels):
        depth = np.cumsum(models1D[i][1])
        res = models1D[i][0]
        dataholder_HM[i, :] = walktem('hm', depth=depth, res=res)
        dataholder_LM[i, :] = walktem('lm', depth=depth, res=res)

    return dataholder_HM, hm_off_time, dataholder_LM, lm_off_time


def forward_hydro(Arch_Table, Hydromodel, updated_prop='K', stratIndex=None, faciesIndex=None, propIndex=None):
    """Function to simulate forward of a given hydromodel"""

    # We want to work with array, so in case a int is passed, we make a 1D array of it

    if type(stratIndex) is int:
        stratIndex = np.array([stratIndex])
    if type(faciesIndex) is int:
        faciesIndex = np.array([faciesIndex])
    if type(propIndex) is int:
        propIndex = np.array([propIndex])

    if type(stratIndex) is type(None):
        stratIndex = np.arange(Arch_table.nreal_units)
    if type(faciesIndex) is type(None):
        faciesIndex = np.arange(Arch_table.nreal_fa)
    if type(propIndex) is type(None):
        propIndex = np.arange(Arch_table.nreal_prop)
    # Identify wich method is asked for):
    dataholder = []
    for strat in stratIndex:
        for facies in faciesIndex:
            for prop in propIndex:
                if updated_prop == 'K':
                    new_K = Arch_Table.getprop('K')[strat, facies, prop]
                    Hydromodel.update_K(new_K)
                    Hydromodel.run_sim()
                    dataholder.append(Hydromodel.get_heads())
    DataReal = DataRealisation('Hydro1 ' + updated_prop, dataholder, None, stratIndex, faciesIndex, propIndex)
    return DataReal
