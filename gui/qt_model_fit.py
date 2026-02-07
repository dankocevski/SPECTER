# qt_model_fit.py

from matplotlib.gridspec import GridSpec
from gdt.core.plot.model import ModelFit as BaseModelFit  # adjust import path if needed
from gdt.core.plot.plot import ModelData
from gdt.core.plot.plot import PlotElementCollection as Collection
from gdt.core.plot.defaults import PLOTFONTSIZE

import numpy as np
import matplotlib.pyplot as plt

class QtModelFit(BaseModelFit):
    """
    A ModelFit variant that can:
    - Attach to an existing Axes (e.g. embedded in a PyQt6 canvas)
    - Hide/show residuals by toggling visibility & GridSpec, not deleting axes
    """

    def __init__(self, fitter=None, canvas=None, ax=None,
                 view='counts', resid=False, interactive=False):
        """
        Parameters:
            fitter : SpectralFitter, optional
            canvas : ignored (kept for API compatibility)
            ax     : matplotlib Axes, optional
                If provided, use this as the top panel and create a residual
                axis beneath it in the same Figure.
            view   : str, optional
            resid  : bool, optional
            interactive : bool, optional
                Only used in the standalone (no-ax) case, like the original.
        """

        # Case 1: No existing Axes provided -> behave like the original class
        if ax is None:
            super().__init__(fitter=fitter,
                             canvas=canvas,
                             view=view,
                             resid=resid,
                             interactive=interactive)
            return
        
        # Case 2: Existing Axes provided -> customize initialization
        self._view = view
        self._fitter = None
        self._count_models = Collection()
        self._count_data = Collection()
        self._resids = Collection()
        self._spectrum_model = Collection()

        # Now override the figure/axes to hook into the existing canvas Axes
        self._ax = ax
        self._figure = self._ax.figure

        # Build a 2-row GridSpec in this figure: top = main, bottom = residuals
        gs = GridSpec(2, 1, figure=self._figure, height_ratios=[3, 1])

        # Attach the existing axes to the top slot of the GridSpec
        self._ax.set_subplotspec(gs[0])
        self._ax.set_position(gs[0].get_position(self._figure))

        # Create the residual axis at the bottom, sharing x with the main axis
        self._resid_ax = self._figure.add_subplot(gs[1], sharex=self._ax)

        # Remove vertical space between panels
        self._figure.subplots_adjust(hspace=0)

        # Reset view/fitter state like the original
        self._view = view
        self._fitter = None

        # Plot data if fitter provided
        if fitter is not None:
            self.set_fit(fitter, resid=resid)

        # self._resid_ax_hidden = False

    # ------------------------------------------------------------------
    # Override hide_residuals / show_residuals with your custom versions
    # ------------------------------------------------------------------
    def hide_residuals(self):
        """Hide the residuals panel but do not delete the axis."""
        if not hasattr(self, "_resid_ax"):
            return

        # Make the residual axes invisible
        self._resid_ax.set_visible(False)

        # Shrink the bottom GridSpec row to effectively zero height
        gs = self._ax.get_gridspec()
        gs.set_height_ratios([1, 0.0001])   # top panel full, bottom nearly zero

        # Give the main axis the x-label and show x-ticks
        self._ax.set_xlabel("Energy (keV)", fontsize=PLOTFONTSIZE)
        self._ax.xaxis.set_tick_params(labelbottom=True)

        # Expand the main axis to fill the area
        self._ax.set_position([0.125, 0.11, 0.775, 0.77])

        # self._resid_ax_hidden = True

    def show_residuals(self, sigma=True):
        """Show the residuals panel and restore height, then plot residuals."""
        if not hasattr(self, "_resid_ax"):
            return

        # If for some reason only one axes exists, re-add the residual axis
        if len(self._figure.axes) == 1:
            self._figure.add_axes(self._resid_ax)

        # Make residual axis visible again
        self._resid_ax.set_visible(True)

        # Restore height ratios (3:1 main:residual)
        gs = self._ax.get_gridspec()
        gs.set_height_ratios([3, 1])

        # Clear any existing residual artists before replotting
        self._resid_ax.cla()

        # --- Original residual plotting logic ---
        energy, chanwidths, resid, resid_err = self._fitter.residuals(sigma=sigma)

        ymin, ymax = ([], [])
        for i in range(self._fitter.num_sets):
            det = self._fitter.detectors[i]
            self._resids.include(
                ModelData(energy[i], resid[i], chanwidths[i],
                          resid_err[i], self._resid_ax,
                          color=self.colors[i], alpha=0.7,
                          linewidth=0.9),
                name=det
            )
            ymin.append((resid[i] - resid_err[i]).min())
            ymax.append((resid[i] + resid_err[i]).max())

        # Zero line
        self._resid_ax.axhline(0.0, color='black')
        self._resid_ax.set_xlabel('Energy [kev]', fontsize=PLOTFONTSIZE)

        if sigma:
            self._resid_ax.set_ylabel('Residuals [sigma]', fontsize=PLOTFONTSIZE)
        else:
            self._resid_ax.set_ylabel('Residuals [counts]', fontsize=PLOTFONTSIZE)

        # Manual y-limits (as in original)
        ymin = min(ymin)
        ymax = max(ymax)
        self._resid_ax.set_ylim((1.0 - np.sign(ymin) * 0.1) * ymin,
                                (1.0 + np.sign(ymax) * 0.1) * ymax)

        self._ax.set_position([0.125, 0.3025, 0.775, 0.5775])

        # # Contract the main axis to fill just the top panel
        # if self._resid_ax_hidden == True:
        #     self._ax.set_position([0.125, 0.11, 0.775, 0.88])
        #     self._resid_ax_hidden = False
