import numpy as np
import subprocess
import multiprocessing
from ArchPy.base import *
from ArchPy.databox import *
import os
import inspect
import gc
from IPython import display

def residual(DataRealisation,TrueData):
    residual = []
    for i in range(DataRealisation.nmodel):
        residual.append(TrueData.misfit(DataRealisation.datas[i]))
    return residual

def likelihood(DataRealisation,TrueData):
    likelihood = []
    for i in range(DataRealisation.nmodel):
        likelihood.append(TrueData.likelihood(DataRealisation.datas[i]))
    return likelihood

def likelihood_rhoa(DataRealisation,TrueData):
    likelihood = []
    for i in range(DataRealisation.nmodel):
        likelihood.append(TrueData.likelihood_rhoa(DataRealisation.datas[i]))
    return likelihood

def metropolis_inversion(Arch_table, trueData, n_iter, n_Fake, plot = True, export_plot=False, True_model = None):
    all_residual = [1000]
    allmodels = []
    repro = 0
    fig = plt.figure()
    for i in range(n_iter):
        if repro:
            x,y = random_int(400,Arch_table)
            x,y = np.unique(np.array([x,y]),axis=1)
            fake_bh = make_fake_bh(Arch_table,x,y,stratIndex=0, faciesIndex=0)
            Arch_table.list_fake_bhs = []
            Arch_table.add_fake_bh(fake_bh)

        Arch_table.reprocess()
        Arch_table.compute_surf(1,fl_top = True)
        Arch_table.compute_facies(nreal = 1)
        Arch_table.compute_prop(1)

        forward2 = Arch_table.physicsforward('tTEM',trueData.positions,0,0,0)
        forward_hydro = forward.forward_hydro(P2, Hydromodel, updated_prop = 'K', stratIndex=0, faciesIndex=0, propIndex=0)

        fac = Arch_table.get_facies()[0,0,:,25,:]
        fac[fac == 0] = np.nan
        if True_model == None:
            fac2 = fac.copy()
            fac2 = fac2 * 0
        else:
            fac2 = True_model.get_facies()[0,0,:,25,:]
        if plot:
            plt.close(fig)
            display.clear_output(wait=True)
            fig, axs = plt.subplots(5,figsize=(10,12))
            fig.suptitle('Iteration' + str(i))
            axs[0].scatter(trueData.rhoa()[0],forward2.datas[0].rhoa()[0])

            axs[1].semilogy(abs(trueData.rhoa()[0].flatten()-forward2.datas[0].rhoa()[0].flatten()))

            all_residual.append(likelihood_rhoa(forward2,trueData)[0])

            axs[2].semilogy(all_residual[1:],'o-')
            axs[2].title.set_text(str(all_residual[-1]))


            axs[3].imshow(fac,aspect = 'auto', origin='lower',extent=[0,50,0,50])
            axs[4].imshow(fac2,aspect = 'auto', origin='lower',extent=[0,50,0,50])
            display.display(plt.gcf())
            if export_plot:
                plt.savefig('figs/fig'+str(i)+'.png')
        alpha = min(1,all_residual[-1]/all_residual[-2])
        repro = int(np.random.random(1)[0] < alpha)
        gc.collect()
        allmodels.append(Arch_table.get_facies()[0,0,:,:,:])
    return allmodels,all_residual[1:]
#def invert(Arch_table,max_iter=10, misfit = 1):
def random_int(n, Arch_table):
    x = np.random.uniform(Arch_table.ox,Arch_table.ox+Arch_table.nx*Arch_table.sx,n)
    y = np.random.uniform(Arch_table.oy,Arch_table.oy+Arch_table.ny*Arch_table.sy,n)
    return x,y

def make_fake_bh(Arch_table,positions_x,positions_y,stratIndex=0, faciesIndex=0):
    surfaces, unit_names = Arch_table.get_surface()
    n_surf = surfaces.shape[0]
    surfaces = surfaces[:,stratIndex,:,:]
    facies = Arch_table.get_facies()[stratIndex,faciesIndex]
    fake_bh = []
    if type(positions_x) is not np.ndarray:
        positions_x = np.array([positions_x])
    if type(positions_y) is not np.ndarray:
        positions_y = np.array([positions_y])
    for x,y in zip(positions_x,positions_y):
        surf = []
        cell_x,cell_y, z = Arch_table.pointToIndex(x,y,1)
        for i in range(n_surf):
            surf.append((Arch_table.get_unit(unit_names[i]),surfaces[i,cell_y,cell_x]))

        facies_idx = facies[:,cell_y,cell_x]

        facies_log = []
        #If model has no free space above, we need to add the transition Surface - first layer
        if facies_idx[-1] != 0:
            facies_log.append((Arch_table.get_facies_obj(ID = facies_idx[-1], type='ID'),Arch_table.zg[-1]))

        for index_trans in reversed(np.where(np.diff(facies_idx) != 0)[0]):
            facies_log.append((Arch_table.get_facies_obj(ID =facies_idx[index_trans], type='ID'),Arch_table.zg[index_trans+1]))
        fake_bh.append(borehole("fake","fake",x=x,y=y,z=Arch_table.top[cell_y,cell_x],depth =-Arch_table.bot[cell_y,cell_x],log_strati=surf,log_facies=facies_log))
    return fake_bh
