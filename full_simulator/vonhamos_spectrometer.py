# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append(r"/Users/aminelamouchi/opt/anaconda3/lib/python3.8/site-packages")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

import matplotlib
matplotlib.use('Agg')

dxCrystal=100.
dyCrystal=25.

energy_range = np.linspace(9000-100, 9000+100, 100)

def build_beamline(energy=9000, beamH = 0.02/2.35, beamV = 0.02/2.35):
    beamLine = raycing.BeamLine()         

    # --- beamline parameters ---------------------------------------------
    E0 = energy    # eV
    Rs  = 250.0     # sagittal radius (mm)
    Rm  = 1e9    # meridional radius (mm)
    crystal_mat = rmats.CrystalSi(hkl=[4,4,4], t=25) # silicon crystal with 25 mm thickness
    theta  = crystal_mat.get_Bragg_angle(E0)
    
    # --- geometry ----------------------------------------------------
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sin2Theta = np.sin(2 * theta)
    p = Rs / sinTheta
    yDet = p * 2 * cosTheta**2
    zDet = p * sin2Theta

    # --- source ----------------------------------------------------
    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        center=(0., 0., 0.),
        dx=beamH, # horizontal beam size
        dz=beamV, # vertical beam size
        energies=energy_range,
        distxprime='flat',
        distzprime='flat',
        dxprime = 1.1 * dxCrystal / p, # angular divergence to cover crystal, 1.1 is a safety margin
        dzprime = dyCrystal * sinTheta / p, # vertical divergence
        nrays=1e6)

    # --- analyzer --------------------------------------------------
    beamLine.vonHamos01 = roes.JohannToroid(
        bl=beamLine, Rm=Rm, Rs=Rs,
        limPhysX=[-50.,  50.],
        limPhysY=[-12.5, 12.5],
        material=crystal_mat, shape='rect')

    beamLine.vonHamos01.center = 0, p, 0
    beamLine.vonHamos01.pitch = theta

    beamLine.screen01 = rscreens.Screen(bl=beamLine)
    beamLine.screen01.center = 0, yDet, zDet
    beamLine.screen01.x = 1, 0, 0
    beamLine.screen01.z = 0, cosTheta, sinTheta

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    vonHamos01beamGlobal01, vonHamos01beamLocal01 = beamLine.vonHamos01.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=vonHamos01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'vonHamos01beamGlobal01':        vonHamos01beamGlobal01,
        'vonHamos01beamLocal01':         vonHamos01beamLocal01,
        'screen01beamLocal01':           screen01beamLocal01}
    
    beamLine.prepare_flow()

    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []
    plots.append(xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(label='x'),
        yaxis=xrtplot.XYCAxis(label='z'),
        caxis=xrtplot.XYCAxis(label='energy', unit='eV'),
        title='detector', saveName='detector.png'))
    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots()
    #beamLine.glow(scale=[0.3, 0.3, 0.3])  # uncomment for a 3‑D viewer
    xrtrun.run_ray_tracing(plots=plots, backend='raycing', beamLine=beamLine, repeats=5)


if __name__ == '__main__':
    main()
