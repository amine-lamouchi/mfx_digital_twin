from lcls_beamline_toolbox.models import mfx
import vonhamos_spectrometer as vh

def main():
    # 1) run Matt's simulation and get IP profile
    mfx_sim = mfx.MFX(E0=9000, N=256)
    mfx_sim.propagate()
    mfx_ip = mfx_sim.beamline.MFX_IP

    wx = mfx_ip.wx * 1000 / 2.355   # convert from m to mm then FWHM to sigma
    wy = mfx_ip.wy * 1000 / 2.355

    # 2) build von Hamos spectrometer and attach source
    beamLine = vh.build_beamline(energy=mfx_sim.E0, beamH=wx, beamV=wy)

    # 3) run ray tracing
    vh.rrun.run_process = vh.run_process
    plots = vh.define_plots()
    vh.xrtrun.run_ray_tracing(plots=plots, beamLine=beamLine, backend="raycing")

if __name__ == "__main__":
    main()
