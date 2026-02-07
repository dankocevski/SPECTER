import glob
import pickle
import os
import warnings
import sys
import numpy as np
import logging
import json

from pathlib import Path
from platformdirs import user_data_dir

from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.binning.binned import (
    combine_by_factor,
    rebin_by_time,
    rebin_by_snr,
    combine_into_one,
)
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.data_primitives import TimeBins
from gdt.core.spectra import functions
from gdt.core.spectra.fitting import (
    SpectralFitterPgstat,
    SpectralFitterCstat,
    SpectralFitterChisq,
)
from gdt.core.plot.model import ModelFit
from gdt.core.plot.spectrum import Spectrum
from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.plot.earthplot import EarthPlot, EarthPoints
from gdt.core.plot.sky import EquatorialPlot
from gdt.core import data_path

from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.missions.fermi.gbm.finders import TriggerFtp, ContinuousFtp
from gdt.missions.fermi.gbm.response import GbmRsp2
from gdt.missions.fermi.gbm.trigdat import Trigdat
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.phaii import GbmPhaii
from gdt.missions.fermi.plot import FermiEarthPlot
from gdt.missions.fermi.gbm.saa import GbmSaa
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.localization import GbmHealPix

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from scipy.signal import correlate, correlation_lags
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.widgets import RectangleSelector

from gui.phaviewer import PhaViewer
from gui.fitplotter import FitPlotter
from gui.range_selector import RangeSelector
from gui.filemanager import FileManager
from gui.qt_model_fit import QtModelFit

import analysis.bayesian_blocks as bayesian_blocks
import analysis.structure_function as sf
import analysis.energetics as energetics
import analysis.leahy as leahy

# Set the app name for platformdirs
APP_NAME = "SPECTER"

# Hide warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="astropy.coordinates")

# Get the default data directory
def default_data_dir() -> Path:
    return Path(user_data_dir(appname=APP_NAME))

# Get the data directory, with possible overrides
def get_data_dir() -> Path:

    # 1) env var override
    env = os.getenv("SPECTER_DATA_DIR")
    if env:
        print("\nUsing data directory: %s" % Path(env).expanduser())
        return Path(env).expanduser()

    # 2) config file override (optional)
    cfg = Path.home() / ".config" / "specter" / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        if "data_dir" in data:
            print("\nUsing data directory: %s" % Path(data["data_dir"]).expanduser())
            return Path(data["data_dir"]).expanduser()

    # 3) default
    data_dir_path = default_data_dir()
    print("\nUsing data directory: %s" % data_dir_path)
    return data_dir_path  

# Get the home directory
DATA_DIR = get_data_dir()

class GRB(object):
    """GRB
    ---
    A high-level helper class for the Gamma-ray Data Tools.

    This class aggregates data I/O, lightcurve and spectral processing, plotting,
    and several analysis utilities commonly used in GRB (gamma-ray burst)
    workflows. It is intended to coordinate operations across multiple detector
    products (NaI and BGO), manage background fits and instrument responses, and
    wrap common analysis tasks such as T90, Bayesian blocks, lag/period searches,
    hardness ratios, spectral model fitting, and energetics (Eiso/Liso) calculations.

    Primary attributes (representative)
    - name (str): burst name/identifier (e.g., "130427A").
    - met (float): mission-elapsed time (MET) for the trigger (property with setter).
    - time (Time-like): astropy Time wrapper set via met.
    - home_dir, data_dir (path-like): local filesystem storage locations.
    - data (GbmDetectorCollection): loaded detector data (TTE binned -> Phaii or Phaii).
    - detectors (list[str]): detector identifiers in `data` (e.g., ["n0","n1","b0"]).
    - data_type (str): "tte", "cspec", or "ctime".
    - data_filenames, rsp2_filenames (list[str]): filenames for loaded products.
    - rsp, rsp2 (GbmDetectorCollection or list): instrument responses per detector.
    - backgrounds (GbmDetectorCollection): interpolated backgrounds for the analysis interval.
    - specfitter (SpectralFitter*): the active fitter instance after fit_spectra().
    - fit_model: instantiated composite spectral model used for the last fit.
    - photon_flux, energy_flux: flux values calculated after a spectral fit.
    - src_range, bkgd_range, view_range: time selection tuples used by plotting/fitting.
    - energy_range_nai, energy_range_bgo: energy selection tuples for NaI and BGO detectors.
    - bblocks_results, t90, specfitters: analysis results produced by various routines.
    - _pha_viewers, _fit_plotter, _file_manager: GUI windows (if used).

    Behavioral and error considerations
    - Many methods assume presence of supporting domain objects (GbmTte, GbmPhaii, GbmRsp2,
        GbmDetectorCollection, BackgroundFitter, SpectralFitter*, bayesian_blocks, sf, energetics).
        Missing or incompatible implementations for these classes will raise exceptions.
    - Interactive selection methods (select_backgrounds, select_source) rely on GUI widgets
        and block until the user completes the selection.
    - The class prints diagnostic messages for most operations rather than raising for
        recoverable issues (e.g., missing response files or non-convergent fits). Callers
        should inspect instance attributes (specfitter.success, fit_model, backgrounds, etc.)
        to programmatically detect failures.
    - Several plotting methods call plt.show() (blocking depending on backend) and may
        attempt to set window titles or use Qt-specific functionality; those calls are
        guarded in places but some backends may ignore or raise errors if GUI frameworks
        are absent.

    Example usage:
            import specter
            grb = specter.GRB("160509374")
            grb.load_tte()
            grb.src_range=(14,17)
            grb.bkgd_range=[[(-150, -25), (100, 200)]]
            grb.fit_backgrounds(order=1)
            grb.fit_spectra(models=["band"], stat="PG-Stat")
            grb.calc_t90()
            eiso = grb.calc_eiso(redshift=1.23)
    """

    def __init__(self, name, silent=False):

        # print("Gamma-ray Data Analysis Tool v1.0")
        # print("A GDT Application")

        self.name = name
        self.trigdat_header = None
        self.trigger_detectors = None
        self.detectors = None
        self.fsw_location = None
        self.fsw_classification = None
        self.fsw_spectrum = None
        self.spec_rebin_factor = None
        self.temporal_resolution = None
        # self.home_dir = "/Users/dkocevsk/Research/Data/GBM"
        self.home_dir = DATA_DIR
        self.data_dir = data_path.joinpath(self.home_dir, self.name)
        self.filename = None
        self.data_collection = None
        self.backgrounds = None
        self.rsp2 = None
        self.models = None
        self.fit_model = None
        self.data = None
        self.data_filenames = []
        self.rsp_filenames = []
        self.rsp2_filenames = []
        self.specfitter = None
        self.model_plot = None
        self.bblocks_results = None
        self.t90_range = None
        self._met = None
        self.time = None
        self.continuous_data = False
        self._pha_viewers = []
        self._open_dialogs = []
        self._fit_plotter = None
        self._file_manager = None
        self.data_type = None
        self.poshist_file = None
        self.healpix_file = None
        self.poshist = None
        self.healpix = None
        self.redshift = None

        self._src_range = None
        self.bkgd_range = None
        self.view_range = None
        # self.analysis_range = (None, None)
        self.analysis_range = None

        self._energy_range_nai = (10, 1000)
        self._energy_range_bgo = (325, 35000)

        self.all_detectors = [
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "b0",
            "b1",
        ]

        if silent == False:
            try:
                self.get_info()
            except:
                print(
                    "\nWarning: Could not retrieve trigger information for this burst."
                )
                self.trigger_detectors = self.all_detectors

    @property
    def met(self):
        return self._met

    @met.setter
    def met(self, value):
        self._met = value
        self.time = Time(value, format="fermi")

    @property
    def src_range(self):
        """Current source time range."""
        return getattr(self, "_src_range", None)

    @src_range.setter
    def src_range(self, value):
        # Check for actual change → don’t spam GUI updates
        if value == self._src_range:
            return

        self._src_range = value
        self.update_gui_windows()

    @property
    def energy_range_nai(self):
        """Current energy range."""
        return getattr(self, "_energy_range_nai", None)

    @energy_range_nai.setter
    def energy_range_nai(self, value):
        # Check for actual change → don’t spam GUI updates
        if value == self._energy_range_nai:
            return

        self._energy_range_nai = value
        self.update_gui_windows(reset_ylim=True)

    @property
    def energy_range_bgo(self):
        """Current energy range."""
        return getattr(self, "_energy_range_bgo", None)

    @energy_range_bgo.setter
    def energy_range_bgo(self, value):
        # Check for actual change → don’t spam GUI updates
        if value == self._energy_range_bgo:
            return

        self._energy_range_bgo = value
        self.update_gui_windows(reset_ylim=True)

    def load_file(
        self, filename, resolution=0.064, data_type=None, bin=True, names=None
    ):
        """
        Load a single GBM data file (TTE, CSPEC, or CTIME) and incorporate it into the
        current object's detector data collection.
        This method opens the FITS file, extracts relevant header metadata, detects the
        data type (unless explicitly provided), and then loads and optionally bins the
        data before adding it to the object's internal GbmDetectorCollection. It
        updates several attributes on the object (met, name, data, detectors,
        data_filenames) as a side effect.
        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the FITS file to load.
        resolution : float, optional
            Time resolution, in seconds, used when binning TTE (time-tagged event)
            data. Default is 0.064 s.
        data_type : str or None, optional
            If provided, should be one of 'tte', 'cspec', or 'ctime'. If None (default),
            the data type is read from the FITS header (DATATYPE). The detected value
            will override this parameter if header information is present.
        bin : bool, optional
            Whether to bin TTE data after loading. If True (default), TTE data are
            converted to binned pha/ii format using the configured binning method and
            resolution. Ignored for CSPEC/CTIME files.
        names : sequence or None, optional
            Optional list of names to assign when creating a new detector collection.
            (Note: in the current implementation this parameter is not used when
            appending to an existing collection.)
        Returns
        -------
        None
            The method does not return a value. It mutates the object's state:
            - self.met is set from the FITS TRIGTIME header if not already set.
            - self.name is set from the FITS OBJECT header (with 'GRB ' stripped)
              if not already set.
            - self.data is created or extended with the newly loaded detector data.
            - self.detectors is appended with the detector identifier parsed from the
              FITS DETNAM header.
            - self.data_filenames is appended with filename.
            - A warning is printed if mixed data types are loaded into the same object.
        Examples
        --------
        # Load and bin a TTE file with 0.128 s resolution
        obj.load_file("detector_tte.fits", resolution=0.128, bin=True)
        # Load a CSPEC file (no binning)
        obj.load_file("detector_cspec.fits")
        """

        print("\nProcessing %s" % filename)

        hdu = fits.open(filename)
        trigger_time = hdu[0].header.get("TRIGTIME")
        name = hdu[0].header.get("OBJECT").replace("GRB ", "")
        detector = (
            hdu[0].header.get("DETNAM").replace("NAI_0", "n").replace("BGO_0", "b")
        )
        data_type = hdu[0].header.get("DATATYPE").lower()

        if self.met is None:
            self.met = trigger_time

        if self.name is None:
            self.name = name

        if data_type is None:
            data_type = hdu[0].header.get("DATATYPE").lower()
            print("Data type detected: %s" % data_type)

        if data_type == "tte":
            # tte = GbmTte.open(data_path.joinpath(self.data_dir, filename))
            tte = GbmTte.open(filename)

            method = "bin_by_time"
            time_ref = 0.0

            tte_data = []
            if bin == True:
                print(
                    "Binning TTE data using method: %s with resolution: %.3f s"
                    % (method, resolution)
                )
                if "bin_by_time" in method:
                    tte_binned = tte.to_phaii(
                        bin_by_time, resolution, time_ref=time_ref
                    )
                    tte_data.append(tte_binned)
            else:
                tte_data.append(tte)

            if self.data is not None:
                print(
                    "\nAdding data for detector %s to existing data collection"
                    % detector
                )
                self.data.include(tte_data[0], det=detector, name=detector)
            else:
                tte_data = GbmDetectorCollection.from_list(tte_data, names=[detector])
                self.data = tte_data

        if data_type == "cspec" or data_type == "ctime":
            pha = GbmPhaii.open(filename)

            pha_data = []
            pha_data.append(pha)

            if self.data is not None:
                print(
                    "\nAdding data for detector %s to existing data collection"
                    % detector
                )
                self.data.include(pha_data[0], det=detector, name=detector)
            else:
                pha_data = GbmDetectorCollection.from_list(pha_data, names=[detector])
                self.data = pha_data

        if self.detectors is None:
            self.detectors = []
        if self.data_filenames is None:
            self.data_filenames = []

        if self.data_type is not None and self.data_type != data_type:
            print(
                "\nWarning: mixing data types (%s and %s)."
                % (self.data_type, data_type)
            )

        self.detectors.append(detector)
        self.data_filenames.append(filename)

    def get_trigger_data(
        self, name=None, data_type="tte", detectors=None, clobber=False
    ):
        """
        Download trigger data for the current GRB using TriggerFtp.

        Parameters
        ----------
        name : str, optional
            GRB name to use. If None (default), the method will use self.name.
            Note: with the current implementation, passing a non-None value causes
            the method to print an error message and return None; TriggerFtp is
            constructed using self.name regardless of this argument.
        data_type : str, optional
            Type of data to download. Supported values:
            "tte", "cspec", "ctime", "trigdat", "rsp", "rsp2". Default is "tte".
        detectors : sequence or None, optional
            Detectors to request. If None (default), self.trigger_detectors is used.
        clobber : bool, optional
            Whether to overwrite existing files. Present for API compatibility but
            currently unused by this implementation.

        Returns
        -------
        str or None
            The filepath (or filepaths) returned by the underlying TriggerFtp
            get_* method corresponding to the requested data_type. Returns None if
            the method exits early due to an unexpected 'name' argument.

        Raises
        ------
        Exception
            Any exceptions raised by TriggerFtp or underlying I/O/network operations
            are propagated (e.g., download failures, permission errors).

        Notes
        -----
        - The function prints progress/status messages and a message indicating the
          directory where data are saved (self.data_dir).
        - The implementation currently always constructs TriggerFtp with self.name,
          so to request data for a different GRB the object’s name attribute must be
          updated prior to calling this method.
        """
        if name is None:
            name = self.name
        else:
            print("\n A GRB name is required.")
            return

        if detectors is None:
            detectors = self.trigger_detectors

        data_finder = TriggerFtp(self.name)

        print("\nDownloading %s data..." % data_type)
        if data_type == "tte":
            filepath = data_finder.get_tte(download_dir=self.data_dir, dets=detectors)
        elif data_type == "cspec":
            filepath = data_finder.get_cspec(download_dir=self.data_dir, dets=detectors)
        elif data_type == "ctime":
            filepath = data_finder.get_ctime(download_dir=self.data_dir, dets=detectors)
        elif data_type == "trigdat":
            filepath = data_finder.get_trigdat(download_dir=self.data_dir)
        elif data_type == "rsp":
            filepath = data_finder.get_rsp(
                download_dir=self.data_dir, cspec=True, ctime=False, dets=detectors
            )
        elif data_type == "rsp2":
            filepath = data_finder.get_rsp2(
                download_dir=self.data_dir, cspec=True, ctime=False, dets=detectors
            )

        print("\nData saved to: %s" % self.data_dir)
        return filepath

    def get_continuous_data(
        self, name=None, data_type="tte", detectors=None, clobber=False
        ):
        """Download continuous (e.g., TTE) data for the object's configured MET.

        This method locates and downloads continuous GBM data corresponding to self.time
        using the ContinuousFtp helper and saves it to self.data_dir. It prints a
        confirmation of the save location and returns the path(s) reported by the
        underlying downloader.

        Parameters
        ----------
        name : str or None, optional
            Optional name or identifier for the data request. Currently unused by the
            implementation and reserved for future use.
        data_type : str, optional
            Type of continuous data to retrieve. Default is "tte" (Time-Tagged Events).
            The current implementation calls ContinuousFtp.get_tte, so other values may
            not be supported.
        detectors : sequence or None, optional
            If provided, a sequence of detector identifiers (e.g. names or indices)
            to restrict the download to those detectors. If None, data for all
            detectors will be requested.
        clobber : bool, optional
            If True, request overwriting of existing files. This parameter is present
            for API compatibility but is not used by the current implementation.

        Returns
        -------
        str or list or None
            The file path or list of file paths returned by the underlying
            ContinuousFtp.get_tte call. Returns None if self.time is not set.

        Raises
        ------
        Exception
            Propagates exceptions raised by the ContinuousFtp downloader (e.g.
            network/IO errors). A check is performed for self.time and the method
            returns None (after printing a message) rather than raising if no time
            is configured.

        Notes
        -----
        - Files are saved to self.data_dir as reported by the method's printed
          confirmation.
        - Ensure that self.time is set before calling this method.
        - Behavior for data_type values other than "tte" and for clobber handling
          may be extended in future revisions.

        Examples
        --------
        # request all-detector TTE data
        filepath = obj.get_continuous_data(data_type="tte")

        # request specific detectors
        filepath = obj.get_continuous_data(detectors=["n0", "n1"])
        """

        if self.time is None:
            print("\n An MET is required.")
            return

        # Create an instance of the continuous data finder
        data_finder = ContinuousFtp(self.time)

        if detectors is None:
            filepath = data_finder.get_tte(download_dir=self.data_dir)
        else:
            filepath = data_finder.get_tte(download_dir=self.data_dir, dets=detectors)

        print("\nData saved to: %s" % self.data_dir)
        return filepath

    def get_poshist(self):
        """
        Retrieve and download spacecraft position history (poshist) data.

        This method fetches position history data for the spacecraft at the specified time
        using the ContinuousFtp data finder. The downloaded file path is stored in the
        instance variable self.poshist_file.

        Returns
        -------
        None
            
        Notes
        -----
        - Requires self.time to be set before calling this method
        - Downloads data to the directory specified by self.data_dir

        """

        if self.time is None:
            print("\n An time is required.")
            return

        # Create an instance of the continuous data finder
        data_finder = ContinuousFtp(self.time)
        self.poshist_file = data_finder.get_poshist(download_dir=self.data_dir)[0]

        print("\nPosthist data saved to: %s" % self.data_dir)

        return

    def get_healpix(self):
        """
        Download and store the HEALPix probability sky map for the GRB.

        This method retrieves the HEALPix file associated with the GRB from the trigger FTP server
        and saves it to the specified data directory. The HEALPix file contains the localization
        probability map for the gamma-ray burst.

        Returns:
            None

        Notes:
            - Requires self.name to be set (GRB name/identifier)
            - If multiple HEALPix files are found, only the first one is used
            - If no HEALPix file is available, self.healpix_file is set to None
        """

        if self.name is None:
            print("\n A GRB name is required.")
            return

        data_finder = TriggerFtp(self.name)
        self.healpix_file = data_finder.get_healpix(download_dir=self.data_dir)

        if len(self.healpix_file) == 0:
            print("\nHealpix file unavailable for GRB %s\n" % self.name)
            self.healpix_file = None
            return
        else:

            self.healpix_file = self.healpix_file[0]
            print("\nHealpix data saved to: %s" % self.data_dir)

        return

    def get_info(self, name=None):
        """
        Retrieves and displays trigger information for a Gamma-Ray Burst (GRB) event.

        This method fetches the TRIGDAT file for the specified GRB, extracts key information
        from the trigger data headers, and prints a summary of the event details.

        Parameters
        ----------
        name : str, optional
            The name/identifier of the GRB event. If None, uses the instance's name attribute.
            If the instance has no name set, prints an error message and returns.

        Returns
        -------
        None
            This method does not return a value; it sets instance attributes and prints information.

        Attributes Set
        --------------
        trigdat_header : fits.Header
            The primary header from the TRIGDAT file.
        met : float
            Mission Elapsed Time of the trigger.
        trig_sig : float
            Trigger significance value.
        trigger_detectors : list of str
            List of detector names that triggered on the event.
        fsw_location : tuple
            Flight Software localization (RA, Dec, Error) in degrees.
        fsw_classification : tuple
            Top classification from Flight Software (classification_type, confidence).
        fsw_spectrum : str
            Spectrum classification from Flight Software.

        Notes
        -----
        - The method requires a valid GRB name to be provided or set in the instance.
        - Downloads TRIGDAT file to the specified data directory if not already present.
        - Prints detailed trigger information including location, classification, and spectrum.
        - Uses the last (most recent) FSW location from the available locations.

        """

        if name is None:
            name = self.name
        else:
            print("\n A GRB name is required.")
            return

        print("\nRetrieving trigger information...")
        data_finder = TriggerFtp(self.name)
        filepath = data_finder.get_trigdat(self.data_dir)

        trigdat = Trigdat.open(filepath[0])

        header = trigdat.headers[0]
        self.trigdat_header = header

        # self.MET = trigdat.trigtime
        self.met = trigdat.trigtime
        self.trig_sig = header["TRIG_SIG"]
        self.trigger_detectors = trigdat.triggered_detectors
        fsw_locations = trigdat.fsw_locations
        fsw_location_best = fsw_locations[-1]
        self.fsw_location = fsw_location_best.location
        self.fsw_classification = fsw_location_best.top_classification
        self.fsw_spectrum = fsw_location_best.spectrum

        print("\nName: GRB %s" % self.name)
        print("MET: %s" % self.met)
        print("Triggered Detectors: %s" % " ".join(self.trigger_detectors))
        print("Trigger Significance: %s" % self.trig_sig)
        print(
            "Classification: %s %s%%"
            % (
                fsw_location_best.top_classification[0],
                round(fsw_location_best.top_classification[1] * 100, 2),
            )
        )
        print("Spectrum: %s" % fsw_location_best.spectrum)

        print(
            "\nFight Software Localization:\nRA = %s, Dec = %s, Err = %s"
            % (self.fsw_location[0], self.fsw_location[1], self.fsw_location[2])
        )

        return

    def load_tte(
        self,
        detectors=None,
        bin=True,
        resolution=0.064,
        method="bin_by_time",
        time_ref=0.0,
        load_rsp2=True,
        ):
        """
        Load Time-Tagged Event (TTE) data for one or more detectors, optionally binning
        the TTE events into PHA-II (binned) format and loading corresponding RSP2
        response files.
        This method:
        - Determines which detectors to load (uses provided `detectors` or falls back
            to the instance's `trigger_detectors`).
        - Searches for local TTE FITS files matching the pattern
            "glg_tte_{detector}_*.fit" inside `self.data_dir`. If none are found, it
            attempts to download the TTE file via self.get_trigger_data(...).
        - Opens each TTE file with GbmTte.open(...) and, if requested, bins the TTE
            into PHA-II using the selected binning method.
        - Collects the per-detector results into a GbmDetectorCollection and assigns
            it to `self.data`. Also records `self.data_filenames` and sets
            `self.data_type = "tte"`.
        - Optionally attempts to load RSP2 response files with self.load_rsp2(...).
            Failures to load RSP2 are caught and reported as a warning.
        Parameters
        ----------
        detectors : sequence or None, optional
                Iterable of detector identifiers (e.g. ["na1", "nb2"]). If None, the
                instance attribute `self.detectors` will be set to `self.trigger_detectors`
                (unless `self.detectors` was already set). Default is None.
        bin : bool, optional
                If True (default), bin TTE events into binned spectra (PHA-II) according
                to the chosen `method`. If False, raw GbmTte objects are kept in the
                returned collection.
        resolution : float, optional
                Time resolution (in seconds) used for binning when the chosen method
                requires a fixed time resolution. Default is 0.064 (s).
        method : str, optional
                Binning method identifier. The implementation checks whether the string
                "bin_by_time" appears in `method` to select the corresponding binning
                routine (i.e. calls `tte.to_phaii(bin_by_time, resolution, time_ref=...)`).
                Default is "bin_by_time".
        time_ref : float, optional
                Reference time (in the same units as `resolution`, typically seconds)
                passed to the binning routine. Default is 0.0.
        load_rsp2 : bool, optional
                If True (default), attempt to load RSP2 response files for the selected
                detectors by calling `self.load_rsp2(detectors=detectors)`. Errors in
                loading RSP2 are caught and result in a printed warning (no exception is
                raised).
        Returns
        -------
        None
            The method does not return a value. It mutates the object's state:

        Examples
        --------
        # Load and bin TTE for detectors "na1" and "na2", using the default settings
        self.load_tte(detectors=["na1", "na2"])
        # Load raw TTE objects (no binning)
        self.load_tte(bin=False)
        # Load with different time resolution and a nonzero reference time
        self.load_tte(resolution=0.128, time_ref=5.0)
        """

        if detectors is None:
            if self.detectors is None:
                self.detectors = self.trigger_detectors
        else:
            self.detectors = detectors

        tte_data = []
        self.data_filenames = []

        print("")
        for detector in self.detectors:

            # Find matching files for this detector
            pattern = str(self.data_dir) + "/" + f"glg_tte_{detector}_*.fit"
            matches = glob.glob(pattern)

            # If none found, try to download the file and re-check
            if not matches:
                filename = self.get_trigger_data(data_type="tte", 
                detectors=[detector])[0]
            else:
                filename = matches[0]
                self.data_filenames.append(filename)

            print("Processing %s" % filename)
            tte = GbmTte.open(data_path.joinpath(self.data_dir, filename))

            if bin == True:
                if "bin_by_time" in method:
                    tte_binned = tte.to_phaii(
                        bin_by_time, resolution, time_ref=time_ref
                    )
                    tte_data.append(tte_binned)
            else:
                tte_data.append(tte)
    
        tte_data = GbmDetectorCollection.from_list(tte_data, names=self.detectors)

        self.data = tte_data

        if load_rsp2 == True:
            self.load_rsp2(detectors=detectors)

        # Register that we've opened tte data
        self.data_type = "tte"

        print("\nDone.")
        return

    def load_cspec(
        self,
        detectors=None,
        bin=True,
        resolution=0.064,
        method="bin_by_time",
        time_ref=0.0,
        load_rsp2=True,
    ):

        if detectors is None:
            self.detectors = self.trigger_detectors
        else:
            self.detectors = detectors

        cspec_data = []
        self.data_filenames = []

        print("")
        for detector in self.detectors:

            filename = glob.glob(
                str(self.data_dir) + "/" + "glg_cspec_%s_*.pha" % detector
            )

            if len(filename) == 0:
                filepath = self.get_data(data_type="cspec", detectors=detector)
                filename = str(filepath[0])
            else:
                filename = filename[0]

            self.data_filenames.append(filename)

            print("Processing %s" % filename)
            cspec = GbmPhaii.open(data_path.joinpath(self.data_dir, filename))

            cspec_data.append(cspec)

        cspec_data = GbmDetectorCollection.from_list(cspec_data, names=self.detectors)

        self.data = cspec_data

        if load_rsp2:
            self.load_rsp2(detectors=detectors)

        # Register that we've opened cspec data
        self.data_type = "cspec"

        print("\nDone.")
        return

    def load_ctime(
        self,
        detectors=None,
        bin=True,
        resolution=0.064,
        method="bin_by_time",
        time_ref=0.0,
        load_rsp2=True,
    ):

        if detectors is None:
            self.detectors = self.trigger_detectors
        else:
            self.detectors = detectors

        ctime_data = []
        self.data_filenames = []

        print("")
        for detector in self.detectors:

            filename = glob.glob(
                str(self.data_dir) + "/" + "glg_ctime_%s_*.pha" % detector
            )

            if len(filename) == 0:
                # filepath = self.get_data(data_type='ctime', detectors=detector)
                filepath = self.get_trigger_data(data_type="ctime", detectors=detector)
                filename = str(filepath[0])
            else:
                filename = filename[0]

            self.data_filenames.append(filename)

            print("Processing %s" % filename)
            ctime = GbmPhaii.open(data_path.joinpath(self.data_dir, filename))

            ctime_data.append(ctime)

        ctime_data = GbmDetectorCollection.from_list(ctime_data, names=self.detectors)

        self.data = ctime_data


        if load_rsp2:
            self.load_rsp2(detectors=detectors)

        # Register that we've opened cspec data
        self.data_type = "ctime"

        print("\nDone.")
        return

    def load_rsp2(self, detectors=None):

        self.detectors = detectors or self.detectors

        rsp2_data = []

        print("")
        for detector in self.detectors:

            filename = glob.glob(
                str(self.data_dir) + "/" + "glg_cspec_%s_*.rsp2" % detector
            )

            if len(filename) == 0:
                filepath = self.get_trigger_data(data_type="rsp2", detectors=detector)
                filename = str(filepath[0])
            else:
                filename = filename[0]

            self.rsp2_filenames.append(filename)

            print("Processing %s" % filename)
            rsp2 = GbmRsp2.open(filename)
            rsp2_data.append(rsp2)

        rsp2_data = GbmDetectorCollection.from_list(rsp2_data)

        self.rsp2 = rsp2_data

        return

    def load_poshist(self):

        if self.poshist_file is None:
            self.get_poshist()

        print("\nLoading poshist file %s" % self.poshist_file)
        self.poshist = GbmPosHist.open(self.poshist_file)

        return

    def plot_data(
        self,
        detectors=None,
        view_range=None,
        src_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
    ):

        self.detectors = detectors or self.detectors

        if self.data is None:
            self.load_tte()

        # self.energy_range_nai = energy_range_nai or self.energy_range_nai
        # self.energy_range_bgo = energy_range_bgo or self.energy_range_bgo
        # self.view_range = view_range or self.view_range
        # self.src_range = src_range or self.src_range

        data_lcs = self.data.to_lightcurve(
            nai_kwargs={"energy_range": self.energy_range_nai},
            bgo_kwargs={"energy_range": self.energy_range_bgo},
        )
        data_specs = self.data.to_spectrum(time_range=self.src_range)

        if self.backgrounds is not None:
            bkgd_lcs = self.backgrounds.integrate_energy(
                nai_args=self.energy_range_nai, bgo_args=self.energy_range_bgo
            )
            bkgd_specs = self.backgrounds.integrate_time(*self.src_range)
            lcplots = [
                Lightcurve(data=data_lc, background=bkgd_lc, title=detector)
                for data_lc, bkgd_lc, detector in zip(
                    data_lcs, bkgd_lcs, self.detectors
                )
            ]
            specplots = [
                Spectrum(data=data_spec, background=bkgd_spec, title=detector)
                for data_spec, bkgd_spec, detector in zip(
                    data_specs, bkgd_specs, self.detectors
                )
            ]
        else:
            lcplots = [
                Lightcurve(data=data_lc, title=detector)
                for data_lc, detector in zip(data_lcs, self.detectors)
            ]
            specplots = [
                Spectrum(data=data_spec, title=detector)
                for data_spec, detector in zip(data_specs, self.detectors)
            ]

        if self.src_range is not None:
            src_lcs = self.data.to_lightcurve(
                time_range=self.src_range,
                nai_kwargs={"energy_range": self.energy_range_nai},
                bgo_kwargs={"energy_range": self.energy_range_bgo},
            )
            src_specs = self.data.to_spectrum(
                time_range=self.src_range,
                nai_kwargs={"energy_range": self.energy_range_nai},
                bgo_kwargs={"energy_range": self.energy_range_bgo},
            )
            _ = [
                lcplot.add_selection(src_lc) for lcplot, src_lc in zip(lcplots, src_lcs)
            ]
            _ = [
                specplot.add_selection(src_spec)
                for specplot, src_spec in zip(specplots, src_specs)
            ]

        for lcplot in lcplots:
            lcplot.xlim = self.view_range
            fig = lcplot.ax.figure
            fig.set_size_inches(12, 5)

            if self.view_range is not None:
                lcplot.xlim = self.view_range

            lcplot.ax.ticklabel_format(useOffset=False, style="plain", axis="x")
            lcplot.ax.minorticks_on()  # enable minor ticks
            lcplot.ax.xaxis.set_minor_locator(
                ticker.AutoMinorLocator(4)
            )  # 4 per major interval

            if self.continuous_data is True:
                lcplot.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x - self.met:.1f}")
                )

            # # place a shared x-label closer to the bottom; adjust the y-value (0.02) as needed
            # if self.continuous_data is True:
            #     fig.text(0.5, 0.05, 'Time - %.1f (s)' % self.met, ha='center', va='bottom', fontsize=12)
            # else:
            #     fig.text(0.5, 0.05, 'Time (s)', ha='center', va='bottom', fontsize=12)

        plt.show()

        return

    def plot_lightcurves(
        self,
        data=None,
        detectors=None,
        view_range=None,
        src_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        interactive=False,
        time_ref=0.0,
        subtract_bkgd=False,
        summed=False,
        ax=None,
        plot_bkgd=True,
        show_bblocks=False,
    ):

        detectors = detectors or self.detectors
        data = data or self.data

        # Ensure that the data is a collection
        if not isinstance(data, GbmDetectorCollection):
            detectors = [data.detector]
            ax = [ax]
            data = GbmDetectorCollection.from_list([data], names=[data.detector])
            ax_provided = True
        else:
            # Ensure ax is a list with the same length as data
            if ax is None:
                ax_provided = False
                ax = [None] * len(data)

        if data is None:
            self.load_tte(time_ref=time_ref)

        energy_range_nai = energy_range_nai or self.energy_range_nai
        energy_range_bgo = energy_range_bgo or self.energy_range_bgo
        view_range = view_range or self.view_range
        src_range = src_range or self.src_range

        data_lcs = data.to_lightcurve(
            nai_kwargs={"energy_range": energy_range_nai},
            bgo_kwargs={"energy_range": energy_range_bgo},
        )

        # Subtract background
        if subtract_bkgd == True and self.backgrounds is not None:
            bkgd_lcs = self.backgrounds.integrate_energy(
                nai_args=energy_range_nai, bgo_args=energy_range_bgo
            )

            bkgd_subtracted_lcs = []
            total_counts = np.zeros_like(data_lcs[0].counts, dtype=np.float64)
            for index in range(len(data_lcs)):
                counts = data_lcs[index].counts - bkgd_lcs[index].counts.squeeze()
                total_counts += counts
                bkgd_subtracted_lc = TimeBins(
                    counts,
                    bkgd_lcs[index].tstart,
                    bkgd_lcs[index].tstop,
                    bkgd_lcs[index].exposure,
                )
                bkgd_subtracted_lcs.append(bkgd_subtracted_lc)

            data_lcs = bkgd_subtracted_lcs

        # Create the light curves
        if self.backgrounds is not None:
            bkgd_lcs = self.backgrounds.integrate_energy(
                nai_args=energy_range_nai, bgo_args=energy_range_bgo
            )
            lcplots = [
                Lightcurve(data=data_lc, background=bkgd_lc, title=detector, ax=_ax)
                for data_lc, bkgd_lc, detector, _ax in zip(
                    data_lcs, bkgd_lcs, self.detectors, ax
                )
            ]
        else:
            lcplots = [
                Lightcurve(data=data_lc, title=None, interactive=interactive, ax=_ax)
                for data_lc, detector, _ax in zip(data_lcs, detectors, ax)
            ]

        # Add the source intervals
        if src_range is not None:
            src_lcs = data.to_lightcurve(
                time_range=src_range,
                nai_kwargs={"energy_range": energy_range_nai},
                bgo_kwargs={"energy_range": energy_range_bgo},
            )

            if subtract_bkgd == True and self.backgrounds is not None:
                bkgds_slice = self.backgrounds.slice_time(src_range[0], src_range[1])

                bkgd_subtracted_src_lcs = []
                total_counts = np.zeros_like(src_lcs[0].counts, dtype=np.float64)
                for index in range(len(src_lcs)):
                    bkgd_slice_lc = bkgds_slice[index].integrate_energy(
                        emin=energy_range_nai[0], emax=energy_range_nai[1]
                    )
                    counts = src_lcs[index].counts - bkgd_slice_lc.counts.squeeze()
                    total_counts += counts
                    bkgd_subtracted_src_lc = TimeBins(
                        counts,
                        bkgd_slice_lc.tstart,
                        bkgd_slice_lc.tstop,
                        bkgd_slice_lc.exposure,
                    )
                    bkgd_subtracted_src_lcs.append(bkgd_subtracted_src_lc)
                src_lcs = bkgd_subtracted_src_lcs

            _ = [
                lcplot.add_selection(src_lc) for lcplot, src_lc in zip(lcplots, src_lcs)
            ]

        # Create the figure for each plot
        for lcplot, detector in zip(lcplots, detectors):
            if ax_provided is False:
                fig = lcplot.ax.figure
                fig.set_size_inches(12, 5)

            if subtract_bkgd is True:
                lcplot.remove_errorbars()

            if view_range is not None:
                lcplot.xlim = view_range

            # Add a detector label
            lcplot.ax.text(
                0.02,
                0.95,
                detector,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            # Adding an energy label
            energy_label = ""
            if detector.startswith("n"):
                if energy_range_nai is not None:
                    energy_label = (
                        f"{energy_range_nai[0]:.1f} - {energy_range_nai[1]:.1f} keV"
                    )
            else:
                if energy_range_bgo is not None:
                    energy_label = (
                        f"{energy_range_bgo[0]:.1f} - {energy_range_bgo[1]:.1f} keV"
                    )

            x_pos = max(0.01, 1.0 - 0.0115 * len(energy_label))
            lcplot.ax.text(
                x_pos,
                0.95,
                energy_label,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            lcplot.ax.ticklabel_format(useOffset=False, style="plain", axis="x")
            lcplot.ax.minorticks_on()  # enable minor ticks
            lcplot.ax.xaxis.set_minor_locator(
                ticker.AutoMinorLocator(4)
            )  # 4 per major interval

            if self.continuous_data is True:
                lcplot.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x - self.met:.1f}")
                )
                lcplot.ax.set_xlabel(f"Time - {self.met:.1f} (s)")

        if show_bblocks is True:

            results = self.bblocks_results

            block_edges = results["block_edges"]
            block_rates = results["block_rates"]
            block_counts = results["block_counts"]

            lcplot.ax.step(block_edges[:-1], block_rates, where="post", color="orange")

        if ax_provided is False:
            plt.tight_layout()
            plt.show()

        return lcplots

    def plot_summed_lightcurve(
        self,
        data=None,
        detectors=None,
        view_range=None,
        src_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        interactive=False,
        time_ref=0.0,
    ):

        detectors = detectors or self.detectors
        data = data or self.data

        energy_range_nai = energy_range_nai or self.energy_range_nai
        energy_range_bgo = energy_range_bgo or self.energy_range_bgo
        view_range = view_range or self.view_range
        src_range = src_range or self.src_range

        data_lcs = data.to_lightcurve(
            nai_kwargs={"energy_range": energy_range_nai},
            bgo_kwargs={"energy_range": energy_range_bgo},
        )

        if self.backgrounds is not None:

            # Subtract background
            bkgd_lcs = self.backgrounds.integrate_energy(
                nai_args=energy_range_nai, bgo_args=energy_range_bgo
            )

            bkgd_subtracted_lcs = []
            total_counts = np.zeros_like(data_lcs[0].counts, dtype=np.float64)
            for index in range(len(data_lcs)):
                counts = data_lcs[index].counts - bkgd_lcs[index].counts.squeeze()
                total_counts += counts
                bkgd_subtracted_lc = TimeBins(
                    counts,
                    bkgd_lcs[index].tstart,
                    bkgd_lcs[index].tstop,
                    bkgd_lcs[index].exposure,
                )
                bkgd_subtracted_lcs.append(bkgd_subtracted_lc)

            data_lcs = bkgd_subtracted_lcs
        
        else:
            print("\nNo backgrounds available; Exiting.\n")
            return
            # total_counts = np.zeros_like(data_lcs[0].counts, dtype=np.float64)
            # for index in range(len(data_lcs)):
            #     counts = data_lcs[index].counts
            #     total_counts += counts

        # Sum the lightcurves
        summed_lc = TimeBins(
            # total_counts, bkgd_lcs[0].tstart, bkgd_lcs[0].tstop, bkgd_lcs[0].exposure
            total_counts, data_lcs[0].tstart, data_lcs[0].tstop, data_lcs[0].exposure
        )
        data_lcs = [summed_lc]
        
        detectors = (
            " + ".join(self.detectors)
            if isinstance(self.detectors, (list, tuple))
            else str(self.detectors)
        )
        detectors = [detectors]

        # Create the light curves
        lcplots = [
            Lightcurve(
                data=data_lc,
                title="Summed Lightcurve - GRB %s" % self.name,
                interactive=interactive,
            )
            for data_lc, detector in zip(data_lcs, detectors)
        ]

        lcplot = lcplots[0]
        detector = detectors[0]

        # Create the figure for each plot
        fig = lcplot.ax.figure
        fig.set_size_inches(12, 5)

        # Remove the error bars
        lcplot.remove_errorbars()

        # Set the view range
        if view_range is not None:
            lcplot.xlim = view_range

        # Add a detector label
        lcplot.ax.text(
            0.0175,
            0.95,
            detector,
            transform=lcplot.ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        # Adding an energy label
        energy_label = ""
        if detector.startswith("n"):
            if energy_range_nai is not None:
                energy_label = (
                    f"{energy_range_nai[0]:.1f} - {energy_range_nai[1]:.1f} keV"
                )
        else:
            if energy_range_nai is not None:
                energy_label = (
                    f"{energy_range_bgo[0]:.1f} - {energy_range_bgo[1]:.1f} keV"
                )

        x_pos = max(0.01, 0.85)
        lcplot.ax.text(
            x_pos,
            0.95,
            energy_label,
            transform=lcplot.ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        lcplot.ax.ticklabel_format(useOffset=False, style="plain", axis="x")
        lcplot.ax.minorticks_on()  # enable minor ticks
        lcplot.ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator(4)
        )  # 4 per major interval

        # Draw a background line
        lcplot.ax.plot(
            [bkgd_lcs[0].tstart, bkgd_lcs[0].tstop],
            [0, 0],
            color="red",
            linestyle="dashed",
            linewidth=1,
        )

        if self.continuous_data is True:
            lcplot.ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x - self.met:.1f}")
            )
            lcplot.ax.set_xlabel(f"Time - {self.met:.1f} (s)")

        lcplot.ax.grid(alpha=0.3)

        plt.draw()  # Forces a redraw of the current figure
        plt.pause(0.1)

        return

    def plot_stacked_lightcurves(
        self,
        detectors=None,
        view_range=None,
        src_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        interactive=False,
        time_ref=0.0,
        relative_time=True,
        hide_src=False,
        hide_bkgd=False,
    ):

        self.detectors = detectors or self.detectors

        if self.data is None:
            self.load_tte(time_ref=time_ref)

        energy_range_nai = energy_range_nai or self.energy_range_nai
        energy_range_bgo = energy_range_bgo or self.energy_range_bgo
        view_range = view_range or self.view_range
        src_range = src_range or self.src_range

        data_lcs = self.data.to_lightcurve(
            nai_kwargs={"energy_range": energy_range_nai},
            bgo_kwargs={"energy_range": energy_range_bgo},
        )

        if self.backgrounds is not None:
            bkgd_lcs = self.backgrounds.integrate_energy(
                nai_args=energy_range_nai, bgo_args=energy_range_bgo
            )

        detectors = self.data.detector()

        nrows = len(data_lcs)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=True,
            figsize=(12, 10),
            gridspec_kw={"hspace": 0.1},
        )

        # Ensure axes is iterable when nrows == 1
        if nrows == 1:
            axes = [axes]

        lc_plots = []

        # for data_lc, detector, ax in zip(data_lcs, detectors, axes):
        for index in range(len(data_lcs)):

            data_lc = data_lcs[index]
            detector = detectors[index]
            ax = axes[index]

            if self.backgrounds is not None and hide_bkgd == False:
                bkgd_lc = bkgd_lcs[index]
            else:
                bkgd_lc = None

            lcplot = Lightcurve(
                data=data_lc, background=bkgd_lc, interactive=False, ax=ax
            )
            lcplot.ax.set_ylabel("")
            lcplot.ax.set_xlabel("")

            if self.view_range is not None:
                lcplot.xlim = self.view_range

            lcplot.ax.ticklabel_format(useOffset=False, style="plain", axis="x")
            lcplot.ax.minorticks_on()  # enable minor ticks
            lcplot.ax.xaxis.set_minor_locator(
                ticker.AutoMinorLocator(4)
            )  # 4 per major interval

            # Add a detector label
            lcplot.ax.text(
                0.01,
                0.95,
                detector,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            # Adding an energy label
            energy_label = ""
            if detector.startswith("n"):
                if energy_range_nai is not None:
                    energy_label = (
                        f"{energy_range_nai[0]:.1f} - {energy_range_nai[1]:.1f} keV"
                    )
            else:
                if energy_range_nai is not None:
                    energy_label = (
                        f"{energy_range_bgo[0]:.1f} - {energy_range_bgo[1]:.1f} keV"
                    )

            x_pos = max(0.01, 1.0 - 0.009 * len(energy_label))
            lcplot.ax.text(
                x_pos,
                0.95,
                energy_label,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
            lcplot.ax.grid(alpha=0.3)

            if self.continuous_data is True:
                lcplot.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x - self.met:.1f}")
                )

            lc_plots.append(lcplot)

        if self.src_range is not None and hide_src == False:
            src_lcs = self.data.to_lightcurve(
                time_range=src_range,
                nai_kwargs={"energy_range": energy_range_nai},
                bgo_kwargs={"energy_range": energy_range_bgo},
            )
            _ = [
                lcplot.add_selection(src_lc)
                for lcplot, src_lc in zip(lc_plots, src_lcs)
            ]

        # Reserve space on the left and add a single shared y-axis label
        fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
        fig.supylabel("Count Rate (count/s)")

        # place a shared x-label closer to the bottom; adjust the y-value (0.02) as needed
        if self.continuous_data is True:
            fig.text(0.5, 0.05, "Time - %.1f (s)" % self.met, ha="center", va="bottom")
        else:
            fig.text(0.5, 0.05, "Time (s)", ha="center", va="bottom")

        plt.tight_layout()
        plt.draw()  # Forces a redraw of the current figure
        plt.pause(0.1)

        return

    def rebin_lightcurve(self, value, method="rebin_by_time"):

        if method == "rebin_by_time":
            self.temporal_resolution = value
            self.data = self.data.rebin_time(rebin_by_time, self.temporal_resolution)
            self.data = GbmDetectorCollection.from_list(self.data, names=self.detectors)

        if method == "combine_by_factor":
            self.data = self.data.rebin_time(combine_by_factor, value)
            self.data = GbmDetectorCollection.from_list(self.data, names=self.detectors)

        if method == "rebin_by_snr":
            self.data = self.data.rebin_time(rebin_by_snr, value)
            self.data = GbmDetectorCollection.from_list(self.data, names=self.detectors)

        if method == "combine_into_one":
            self.data = self.data.rebin_time(combine_into_one)
            self.data = GbmDetectorCollection.from_list(self.data, names=self.detectors)

        self.update_gui_windows(update_data=True)

        return

    def rebin_spectra(self, value, method="combine_by_factor"):

        if method == "combine_by_factor":

            self.data = self.data.rebin_energy(combine_by_factor, value)
            self.data = GbmDetectorCollection.from_list(self.data, names=self.detectors)

        if method == "combine_into_one":

            self.data = self.data.rebin_energy(combine_into_one)
            self.data = GbmDetectorCollection.from_list(self.data, names=self.detectors)

        self.update_gui_windows(update_data=True)

        return

    def reset_binning(self):

        print("\nResetting data binning...")

        if self.data_type == "tte":
            if self.continuous_data is True:
                self.load_tte(resolution=0.064, time_ref=self.met)
            else:
                self.load_tte()
        elif self.data_type == "cspec":
            self.load_cspec()
        elif self.data_type == "ctime":
            self.load_ctime()

        # print("Done.")

        self.update_gui_windows(update_data=True, reset_ylim=True)

    def plot_spectra(self, data=None, detectors=None, src_range=None, ax=None):

        if self.data is None:
            self.get_data()
            self.load_tte()

        src_range = src_range or self.src_range
        detectors = detectors or self.detectors
        data = data or self.data

        # Ensure that the data is a collection
        if not isinstance(data, GbmDetectorCollection):
            detectors = [data.detector]
            ax = [ax]
            data = GbmDetectorCollection.from_list([data], names=[data.detector])
        else:
            # Ensure ax is a list with the same length as data
            if ax is None:
                ax = [None] * len(data)

        data_specs = data.to_spectrum(time_range=src_range)
        specplots = [
            Spectrum(data=data_spec, title=detector, ax=_ax)
            for data_spec, detector, _ax in zip(data_specs, detectors, ax)
        ]

        src_specs = data.to_spectrum(
            time_range=src_range,
            nai_kwargs={"energy_range": self.energy_range_nai},
            bgo_kwargs={"energy_range": self.energy_range_bgo},
        )
        _ = [
            specplot.add_selection(src_spec)
            for specplot, src_spec in zip(specplots, src_specs)
        ]

        for specplot, detector, _ax in zip(specplots, detectors, ax):

            # Add a detector label
            specplot.ax.text(
                0.01,
                0.95,
                detector,
                transform=specplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            # Adding an energy label
            energy_label = ""
            if detector.startswith("n"):
                if self.energy_range_nai is not None:
                    energy_label = f"{self.energy_range_nai[0]:.1f} - {self.energy_range_nai[1]:.1f} keV"
            else:
                if self.energy_range_bgo is not None:
                    energy_label = f"{self.energy_range_bgo[0]:.1f} - {self.energy_range_bgo[1]:.1f} keV"

            x_pos = max(0.01, 1.0 - 0.012 * len(energy_label))
            specplot.ax.text(
                x_pos,
                0.95,
                energy_label,
                transform=specplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

        plt.show()

        return specplots

    def fit_backgrounds(
        self,
        order=1,
        bkgd_range=None,
        src_range=None,
        view_range=None,
        plot_residuals=True,
        plot_fit=True,
    ):

        data_lcs = self.data.to_lightcurve(
            nai_kwargs={"energy_range": self.energy_range_nai},
            bgo_kwargs={"energy_range": self.energy_range_bgo},
        )

        if self.src_range is None:
            print("\nA source range is required to fit the background. Exiting.")
            return

        view_range = view_range or self.view_range
        src_range = src_range or self.src_range
        bkgd_range = bkgd_range or self.bkgd_range
        bkgd_range_list = self.bkgd_range

        # Ensure we have a per-detector list: if None, or a single-element list, expand to match self.data
        if bkgd_range_list is None:
            self.select_backgrounds(view_range=view_range)
            bkgd_range_list = bkgd_range
        elif isinstance(bkgd_range_list, list) and len(bkgd_range_list) == 1:
            bkgd_range_list = [bkgd_range_list[0]] * len(self.data)

        backfitters = [
            BackgroundFitter.from_phaii(data, Polynomial, time_ranges=bkgd_range)
            for data, bkgd_range in zip(self.data, bkgd_range_list)
        ]
        backfitters = GbmDetectorCollection.from_list(
            backfitters, dets=self.detectors, names=self.detectors
        )

        print("\nFitting background...")
        print("\nBackground selection:")
        for det, rngs in zip(self.detectors, bkgd_range_list):
            print(f"  {det}: {rngs}")

        backfitters.fit(order=order)
        for statistic, dof, detector in zip(
            backfitters.statistic(), backfitters.dof(), self.detectors
        ):
            chisq_dof = statistic / dof

            print("\nDetector: %s" % detector)
            print("chisq / dof:")
            print(chisq_dof)

            if plot_residuals is True:
                # Use a figure window title (the OS window) rather than the axes title
                if "b0" in detector or "b1" in detector:
                    fig = plt.gcf()
                    try:
                        fig.canvas.manager.set_window_title("Fit Residuals")
                    except Exception:
                        # Older/newer backends may not support set_window_title — ignore safely
                        pass
                    ax = plt.gca()
                    ax.set_title(detector)
                    ax.set_xlabel("Energy (keV)")
                    ax.set_ylabel("Chisq / D.O.F")
                    bgfit = ax.step(
                        np.linspace(
                            self.energy_range_bgo[0],
                            self.energy_range_bgo[1],
                            len(chisq_dof),
                        ),
                        chisq_dof,
                    )
                else:
                    fig, ax = plt.subplots()
                    try:
                        fig.canvas.manager.set_window_title("Fit Residuals")
                    except Exception:
                        pass
                    ax.set_title(detector)
                    ax.set_xlabel("Energy (keV)")
                    ax.set_ylabel("Chisq / D.O.F")
                    bgfit = ax.step(
                        np.linspace(
                            self.energy_range_nai[0],
                            self.energy_range_nai[1],
                            len(chisq_dof),
                        ),
                        chisq_dof,
                    )

                try:
                    fig.show()
                except Exception:
                    # Fallback to pyplot show if fig.show() is not supported
                    plt.show()

        print("\nInterpolating background fits...")
        bkgds = backfitters.interpolate_bins(
            self.data.data()[0].tstart, self.data.data()[0].tstop
        )

        bkgds = GbmDetectorCollection.from_list(
            bkgds, names=self.detectors, dets=self.detectors
        )
        print("Done.")

        self.backgrounds = bkgds

        if plot_fit is True:
            self.plot_data()

        return

    def coerce_ranges(self, r):
        """Return a list of (tmin,tmax) tuples from r (tuple or list-of-tuples)."""
        if r is None:
            return None
        # single tuple -> list
        if (
            isinstance(r, (tuple, list))
            and len(r) == 2
            and all(isinstance(x, (int, float)) for x in r)
        ):
            a, b = r
            return [(min(a, b), max(a, b))]
        # iterable of pairs
        out = []
        try:
            for it in r:
                a, b = it
                out.append((min(a, b), max(a, b)))
        except Exception:
            return None
        return out if out else None

    def select_backgrounds(self, view_range=None):

        # 1) Build lightcurves (one per detector)
        data_lcs = self.data.to_lightcurve(
            nai_kwargs={"energy_range": self.energy_range_nai},
            bgo_kwargs={"energy_range": self.energy_range_bgo},
        )

        # 2) Iterate detectors one by one, gather multi-interval background selections
        bkgd_ranges = []  # [ [(tmin, tmax), ...], ... ] — aligned with self.detectors

        default_ranges = self.coerce_ranges(bkgd_ranges) or self.coerce_ranges(
            self.bkgd_range
        )

        for data, detector, lc in zip(self.data, self.detectors, data_lcs):

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(12, 4))

            # Draw the light curve
            lcplot = Lightcurve(data=lc, interactive=True, ax=ax)

            if self.view_range is not None:
                print("Setting view range to: %s" % str(self.view_range))
                lcplot.xlim = self.view_range

            lcplot.ax.ticklabel_format(useOffset=False, style="plain", axis="x")
            lcplot.ax.minorticks_on()  # enable minor ticks
            lcplot.ax.xaxis.set_minor_locator(
                ticker.AutoMinorLocator(4)
            )  # 4 per major interval

            # Add a detector label
            lcplot.ax.text(
                0.01,
                0.95,
                detector,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            # Adding an energy label
            energy_label = ""
            if detector.startswith("n"):
                if self.energy_range_nai is not None:
                    energy_label = (
                        f"{self.energy_range_nai[0]} - {self.energy_range_nai[1]} keV"
                    )
            else:
                if self.energy_range_bgo is not None:
                    energy_label = (
                        f"{self.energy_range_bgo[0]} - {self.energy_range_bgo[1]} keV"
                    )

            x_pos = max(0.01, 1.0 - 0.008 * len(energy_label))
            lcplot.ax.text(
                x_pos,
                0.95,
                energy_label,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            if self.continuous_data is True:
                lcplot.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x - self.met:.1f}")
                )
                lcplot.ax.set_xlabel(f"Time - {self.met:.1f} (s)")

            selector = RangeSelector(fig, ax)
            plt.tight_layout()
            plt.show()  # blocks until user clicks outside axes to finish

            picked_ranges = selector.get_ranges() or (
                default_ranges if default_ranges is not None else []
            )
            selector.disconnect()

            # Ensure list of tuples
            picked_ranges = self.coerce_ranges(picked_ranges) or []
            bkgd_ranges.append(picked_ranges)

        # Keep for reference/use
        self.bkgd_range = bkgd_ranges

        print("\nCollected background ranges (per detector):")
        for det, rngs in zip(self.detectors, bkgd_ranges):
            print(f"  {det}: {rngs}")

        return

    def select_source(self, src_detector, view_range=None):

        # 1) Build lightcurves (one per detector)
        data_lcs = self.data.to_lightcurve(
            nai_kwargs={"energy_range": self.energy_range_nai},
            bgo_kwargs={"energy_range": self.energy_range_bgo},
        )

        # 2) Iterate detectors one by one, gather multi-interval background selections
        src_ranges = []  # [ [(tmin, tmax), ...], ... ] — aligned with self.detectors

        for data, detector, lc in zip(self.data, self.detectors, data_lcs):

            if detector is not src_detector:
                print("detector = %s" % detector)
                print("Detector is not src_detector!")
                continue
            else:
                print("Found %s!" % detector)

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(12, 4))

            # Draw the light curve
            lcplot = Lightcurve(data=lc, interactive=True, ax=ax)

            if self.view_range is not None:
                print("Setting view range to: %s" % str(self.view_range))
                lcplot.xlim = self.view_range

            lcplot.ax.ticklabel_format(useOffset=False, style="plain", axis="x")
            lcplot.ax.minorticks_on()  # enable minor ticks
            lcplot.ax.xaxis.set_minor_locator(
                ticker.AutoMinorLocator(4)
            )  # 4 per major interval

            # Add a detector label
            lcplot.ax.text(
                0.01,
                0.95,
                detector,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            # Adding an energy label
            energy_label = ""
            if detector.startswith("n"):
                if self.energy_range_nai is not None:
                    self.energy_label = f"{self.energy_range_nai[0]:.1f} - {self.energy_range_nai[1]:.1f} keV"
            else:
                if self.energy_range_nai is not None:
                    self.energy_label = f"{self.energy_range_bgo[0]:.1f} - {self.energy_range_bgo[1]:.1f} keV"

            x_pos = max(0.01, 1.0 - 0.008 * len(energy_label))
            lcplot.ax.text(
                x_pos,
                0.95,
                energy_label,
                transform=lcplot.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            if self.continuous_data is True:
                lcplot.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x - self.met:.1f}")
                )
                lcplot.ax.set_xlabel(f"Time - {self.met:.1f} (s)")

            selector = RangeSelector(fig, ax, src_selection=True)
            plt.tight_layout()
            plt.show()  # blocks until user clicks outside axes to finish

            # picked_ranges = selector.get_ranges() or (default_ranges if default_ranges is not None else [])
            src_ranges = selector.get_ranges()
            selector.disconnect()

        print("\nCollected source range:")
        print(f"  {src_detector}: {src_ranges}")

        self.src_range = src_ranges[0]

        return

    def fit_spectra(
        self,
        models=["band"],
        src_range=None,
        view_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        plot_selections=False,
        plot_fit=False,
        stat="PG-Stat",
        default_values=None,
        use_previous_fit=False,
        show_gui=False,
        free=None,
        ):
        """
        Fit spectral models to detector data and update the GRB object's fit state.

        This method prepares data and responses, builds a composite spectral model,
        performs a fit with a chosen statistic, computes fluxes and asymmetric
        parameter errors, and updates several attributes on the instance. It prints
        a summary table of fitted parameters and basic fit diagnostics. Optionally
        it can plot the fit and show/update a GUI fit plotter.

        Parameters
        ----------
        models : list or sequence, optional
            Sequence of model identifiers (strings) or model descriptions to fit.
            If falsy, the instance attribute self.models is used. Default is ["band"].
        src_range : object or sequence, optional
            Time selection for spectral extraction. NOTE: current implementation
            uses instance attribute self.src_range when calling self.data.to_pha.
            Accepted format depends on self.data.to_pha. Default is None.
        view_range : object, optional
            Unused in the current implementation; reserved for future use. Default None.
        energy_range_nai : tuple(float, float), optional
            Energy selection for NaI detectors (keV). NOTE: current implementation
            uses self.energy_range_nai from the instance when creating pha. Default None.
        energy_range_bgo : tuple(float, float), optional
            Energy selection for BGO detectors (keV). NOTE: current implementation
            uses self.energy_range_bgo from the instance when creating pha. Default None.
        plot_selections : bool, optional
            If True, show selection plots (not implemented in this routine). Default False.
        plot_fit : bool, optional
            If True, call self.plot_spectral_fit(show_data=True) after a successful fit.
            Default False.
        stat : str, optional
            Statistic to use for fitting. Supported exact values in this implementation:
            "PG-Stat", "C-Stat", "Chi2". Default is "PG-Stat".
        default_values : sequence, optional
            Sequence of initial/default parameter values for the composite model.
            If provided, assigned to fit_model.default_values before fitting.
        use_previous_fit : bool, optional
            If True, use the last fit's optimized parameters as initial guesses for
            the free parameters of the new model (reads self.specfitter.parameters).
            Default False.
        show_gui : bool, optional
            If True, create or update a FitPlotter GUI (requires class FitPlotter
            and GUI dependencies). Default False.
        free : sequence of bool, optional
            Sequence indicating which model parameters are free (True) or fixed (False).
            If provided, assigned to fit_model.free before fitting.

        Behavior and side-effects
        -------------------------
        - Updates self.models to the provided models list if non-empty.
        - Calls self.data.to_pha(...) to build PHA objects (uses instance attributes
          for time/energy ranges in the current code).
        - Prepares instrument responses: if self.rsp2 is present it interpolates them
          to PHA times and builds self.rsp as a GbmDetectorCollection; otherwise uses
          self.rsp assumed to be set already.
        - Optionally rebins responses if self.spec_rebin_factor is set.
        - Instantiates one of SpectralFitterPgstat, SpectralFitterCstat, or
          SpectralFitterChisq according to `stat` (method="TNC").
        - Builds a composite model by resolving model names via self.resolve_spectral_model.
        - Applies provided default_values and free masks to the model if supplied.
        - If use_previous_fit is True, copies previous fit parameter values into the
          default values for free parameters.
        - Performs the fit via self.specfitter.fit(fit_model, options={"maxiter": 10000}).
        - Computes asymmetric parameter errors with cl=0.9 and integrates the fitted
          model over 10-1000 keV to obtain:
            - self.photon_flux (photons s^-1 cm^-2)
            - self.energy_flux (erg s^-1 cm^-2)
        - Prints a formatted parameter/error table and basic fit statistics.
        - Optionally plots the fit and/or creates/updates a FitPlotter via self._fit_plotter.
        - Stores the fitted model in self.fit_model and keeps self.specfitter updated.

        Return
        ------
        float or None
            The fit statistic (self.specfitter.statistic) if a fit was performed and the
            method completes normally. Returns None if the specified statistic or any
            of the requested models are unrecognized and the method exits early.

        Notes
        -----
        - The method currently expects certain instance attributes to exist:
          self.data, self.rsp or self.rsp2, self.backgrounds, self.rsp, self.detectors,
          self.spec_rebin_factor, self._fit_plotter, and methods/classes:
          SpectralFitterPgstat, SpectralFitterCstat, SpectralFitterChisq,
          GbmDetectorCollection.from_list, self.resolve_spectral_model,
          fit_model.integrate, and fit_model.param_list.
        - The exact formats and requirements for models, default_values, and free
          depend on the model/resolver implementation used by resolve_spectral_model.
        - The code prints messages rather than raising exceptions for some error
          conditions (e.g., unrecognized statistic or model); callers should inspect
          stdout or the instance state to detect early exits.
        - The method uses asymmetric_errors(cl=0.9) to compute uncertainties.
        - Example usage (conceptual):
            grb.fit_spectra(models=["band", "pl"], stat="PG-Stat",
                            default_values=[...], free=[True, True, False, ...],
                            plot_fit=True)
        """

        self.models = models or self.models

        # Get the data
        data_phas = self.data.to_pha(
            time_ranges=self.src_range,
            nai_kwargs={"energy_range": self.energy_range_nai},
            bgo_kwargs={"energy_range": self.energy_range_bgo},
        )

        # Get the responses
        if self.rsp2 is not None:
            self.rsp = [
                rsp2.interpolate(pha.tcent) for rsp2, pha in zip(self.rsp2, data_phas)
            ]
            self.rsp = GbmDetectorCollection.from_list(self.rsp, names=self.detectors)

        if self.spec_rebin_factor is not None:
            rsp = self.rsp.rebin(self.spec_rebin_factor)
        else:
            rsp = self.rsp.to_list()

        if use_previous_fit == False:
            if stat == "PG-Stat":
                self.specfitter = SpectralFitterPgstat(
                    data_phas, self.backgrounds.to_list(), rsp, method="TNC"
                )
            elif stat == "C-Stat":
                self.specfitter = SpectralFitterCstat(
                    data_phas, self.backgrounds.to_list(), rsp, method="TNC"
                )
            elif stat == "Chi2":
                self.specfitter = SpectralFitterChisq(
                    data_phas, self.backgrounds.to_list(), rsp, method="TNC"
                )
            else:
                print(
                    f"\nStatistic {stat} not recognized. Available statistics are: pgstat, cstat, chisq."
                )
                return

        fit_model = self.resolve_spectral_model(self.models[0])
        if fit_model is None:
            print(f"\nModel {self.models[0]} not recognized.")
            return

        for model in self.models[1:]:
            next_model = self.resolve_spectral_model(model)
            if next_model is None:
                print(f"\nModel {model} not recognized.")
                return
            fit_model += next_model

        # Use the user defined default values for the fit parameters
        if default_values is not None:
            print("\nUsing user-defined default parameter values:\n%s" % default_values)
            fit_model.default_values = default_values

        if free is not None:
            fit_model.free = free

        # Use previous fit parameters as initial guesses for the next fit
        # if use_previous_fit == True and self.specfitter.success == True:
        if use_previous_fit == True:
            print("\nUsing previous fit parameters as initial guesses...")
            param_index = 0
            for i, is_free in enumerate(fit_model.free):
                if is_free:
                    fit_model.default_values[i] = self.specfitter.parameters[
                        param_index
                    ]
                    print(
                        f"Parameter {fit_model.param_list[i][0]}: {fit_model.default_values[i]}"
                    )
                    param_index += 1

        print("\nFitting %s model..." % fit_model.name)
        self.specfitter.fit(fit_model, options={"maxiter": 10000})
        # print(specfitter.message)
        if self.specfitter.success == False:
            print(
                "Fit did not converge. Try changing the model or the time/energy selections."
            )
        else:
            print("Fit converged.\n")

        asymmetric_errors = self.specfitter.asymmetric_errors(cl=0.9)

        # flux over 10-1000 keV
        self.photon_flux = fit_model.integrate(
            self.specfitter.parameters, (10.0, 1000.0)
        )  # photons/s/cm^2
        self.energy_flux = fit_model.integrate(
            self.specfitter.parameters, (10.0, 1000.0), energy=True
        )  # erg/s/cm^2

        # Header
        print("")
        print(
            f"{'Parameter':20s}\t{'Description':15s}\t{'Unit':12s}\t{'Value':12s}\t{'-Error':12s}\t{'+Error':12s}"
        )
        print("-" * 100)

        # Iterate through parameters, match free ones with errors
        error_index = 0
        for i, is_free in enumerate(fit_model.free):
            if is_free:
                name, unit, desc = fit_model.param_list[i]

                if "Low-Energy Photon index" in desc:
                    desc = "Low index"
                if "High-Energy Photon index" in desc:
                    desc = "High index"

                value = self.specfitter.parameters[error_index]
                err_lo, err_hi = asymmetric_errors[error_index]
                print(
                    f"{name:20s}\t{desc:15s}\t{unit:12s}\t{value:.3e}\t{err_lo:.3e}\t{err_hi:.3e}"
                )
                error_index += 1

        print(
            f"\nStat/DoF: {self.specfitter.statistic:.2f}/{self.specfitter.dof:.2f}\n"
        )

        print("Photon Flux (10-1000 keV) = %.2e photons/s/cm^2" % self.photon_flux)
        print("Energy Flux (10-1000 keV) = %.2e erg/s/cm^2" % self.energy_flux)
        print("")

        if plot_fit == True:
            self.plot_spectral_fit(show_data=True)

        self.fit_model = fit_model
        self.specfitter = self.specfitter

        # if self._fit_plotter is not None:
        #     self.update_gui_windows()

        if show_gui == True:
            if self._fit_plotter is None:
                self._fit_plotter = FitPlotter(grb=self)
                self._fit_plotter.show()
            else:
                self._fit_plotter.update_plot()
                
        return self.specfitter.statistic

    def plot_spectral_fit(
        self, view="counts", ax=None, show_data=True, show_residuals=True
    ):

        if ax is None:
            self.model_plot = ModelFit(
                fitter=self.specfitter, view=view, interactive=False
            )
        else:
            self.model_plot = QtModelFit(
                fitter=self.specfitter, ax=ax, view=view, interactive=False
            )

        if show_residuals == True:
            QtModelFit.show_residuals(self.model_plot)
        else:
            QtModelFit.hide_residuals(self.model_plot)

        if ax is None:
            plt.show()
        else:
            ax.figure.canvas.draw()

        return self.model_plot

    def find_best_model(
        self,
        src_range=None,
        view_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        plot_selections=False,
    ):

        models = [
            "band",
            "comp",
            "powerlaw",
            "bb",
            ["band", "bb"],
            ["band", "powerlaw"],
            ["comp", "bb"],
            ["comp", "powerlaw"],
            ["powerlaw", "bb"],
        ]

        statistics = []
        dofs = []

        for model in models:
            print(model)
            type(model)
            if not isinstance(model, list):
                model = [model]
            self.fit_spectra(
                models=model,
                src_range=src_range,
                view_range=view_range,
                energy_range_nai=energy_range_nai,
                energy_range_bgo=energy_range_bgo,
                plot_selections=plot_selections,
            )

            statistics.append(self.specfitter.statistic)
            dofs.append(self.specfitter.dof)

        print(f"{'Model':15s}\t{'Stat':12s}\t{'DoF':12s}\t{'Stat/DoF':12s}")
        print("-" * 60)

        for model, stat, dof in zip(models, statistics, dofs):

            # Convert lists like ['band', 'bb'] into a readable string "band+bb"
            if isinstance(model, list):
                model_name = "+".join(model)
            else:
                model_name = model

            chisq_dof = stat / dof
            print(f"{model_name:15s}\t{stat:.3e}\t{dof:.3e}\t{chisq_dof:.3e}")

        return

    def batch_fit_spectra(
        self,
        models=["band"],
        time_selections=[(0, 1), (1, 2)],
        src_range=None,
        view_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        plot_selections=False,
        plot_fit=False,
        use_previous_fit=True,
        stat="PG-Stat",
        default_values=None,
        free=None,
        ):

        # Record the full source range to restore later
        original_src_range = self.src_range

        # Extract the time bins
        tstart = self.data.data()[0].tstart
        tstop = self.data.data()[0].tstop

        # Create a mask where the time intervals overlap the desired range
        src_range = src_range or self.src_range
        mask = (tstop >= src_range[0]) & (tstart <= src_range[1])

        # Apply the mask to extract the subset
        tstart_in_range = tstart[mask]
        tstop_in_range = tstop[mask]

        specfitters = []

        print("\nBatch fitting %d time bins..." % (len(tstart_in_range)))

        for tstart_bin, tstop_bin in zip(tstart_in_range, tstop_in_range):

            if (
                use_previous_fit is True
                and self.specfitter is not None
                and self.specfitter.success == True
            ):
                print("_use_previous_fit = True")
                _use_previous_fit = True
            else:
                print("_use_previous_fit = False")
                _use_previous_fit = False

            print(f"\nFitting time range: ({tstart_bin}, {tstop_bin})")
            self.fit_spectra(
                models=models,
                src_range=(tstart_bin, tstop_bin),
                view_range=view_range,
                energy_range_nai=energy_range_nai,
                energy_range_bgo=energy_range_bgo,
                plot_selections=plot_selections,
                plot_fit=plot_fit,
                use_previous_fit=_use_previous_fit,
            )

            specfitters.append(self.specfitter)

        self.specfitters = specfitters

        number_failed = sum(1 for sf in self.specfitters if not sf.success)
        print(
            f"\nBatch fitting complete. {number_failed} out of {len(self.specfitters)} fits failed to converge.\n"
        )

        print(
            f"{'Start':>10s} {'Stop':>10s} {'DoF':>10s} {'Stat/DoF':>15s} {'Converged':>10s}"
        )
        print("-" * 60)

        for tstart_bin, tstop_bin, specfitter in zip(
            tstart_in_range, tstop_in_range, self.specfitters
        ):
            print(
                f"{tstart_bin:10.3f} {tstop_bin:10.3f} {specfitter.dof:10d} {specfitter.statistic/specfitter.dof:15.3e} {str(specfitter.success):>10}"
            )

        self.src_range = original_src_range

        # Determine the free parameter names (from the first fit's model)
        fit_model = self.fit_model
        free_mask = fit_model.free
        param_info = fit_model.param_list  # list of (name, unit, desc)
        free_names = [
            param_info[i][0] for i, is_free in enumerate(free_mask) if is_free
        ]

        # Print header (now includes Stop column)
        print("")  # leading blank line to match your style
        header = f"{'Start':>10s} {'Stop':>10s}" + "".join(
            f" {n:>14s}" for n in free_names
        )
        print(header)
        print("-" * (10 + 1 + 10 + 1 + 15 * len(free_names)))

        # Print rows
        for tstart_bin, tstop_bin, specfitter in zip(
            tstart_in_range, tstop_in_range, self.specfitters
        ):
            vals = specfitter.parameters
            row = f"{tstart_bin:10.3f} {tstop_bin:10.3f}" + "".join(
                f" {v:14.6g}" for v in vals
            )
            print(row)

        return

    def plot_batch_fit_results(self, param_name, log=False):

        if not hasattr(self, "specfitters") or self.specfitters is None:
            print("No batch fit results found. Please run batch_fit_spectra() first.")
            return

        # Extract the time bins
        tstart = self.data.data()[0].tstart
        tstop = self.data.data()[0].tstop

        # Create a mask where the time intervals overlap the desired range
        src_range = self.src_range
        mask = (tstop >= src_range[0]) & (tstart <= src_range[1])

        # Apply the mask to extract the subset
        tstart_in_range = tstart[mask]
        tstop_in_range = tstop[mask]

        fit_model = self.fit_model
        free_mask = fit_model.free
        param_info = fit_model.param_list  # list of (name, unit, desc)
        free_names = [
            param_info[i][0] for i, is_free in enumerate(free_mask) if is_free
        ]

        if param_name not in free_names:
            print(
                f"Parameter '{param_name}' not found among free parameters: {free_names}"
            )
            return

        param_index = free_names.index(param_name)
        param_values = [
            sf.parameters[param_index] if sf.success else np.nan
            for sf in self.specfitters
        ]
        param_errors = [
            sf.asymmetric_errors()[param_index] if sf.success else (np.nan, np.nan)
            for sf in self.specfitters
        ]
        err_lo = [
            abs(err[0]) if not np.isnan(val) else np.nan
            for val, err in zip(param_values, param_errors)
        ]
        err_hi = [
            abs(err[1]) if not np.isnan(val) else np.nan
            for val, err in zip(param_values, param_errors)
        ]
        mid_times = [
            (start + stop) / 2 for start, stop in zip(tstart_in_range, tstop_in_range)
        ]

        plt.figure(figsize=(10, 5))
        plt.errorbar(mid_times, param_values, yerr=[err_lo, err_hi], fmt="o", capsize=3)
        plt.xlabel("Time (s)")
        plt.ylabel(param_name)
        # plt.title(f'Time Evolution of {param_name}')
        plt.xlim(self.view_range)
        plt.grid(alpha=0.3)

        if log == True:
            plt.yscale("log")

        plt.show()

        return

    def line_search(
        self,
        models=["comp"],
        line_energy_start=10,
        line_width=5.0,
        src_range=None,
        view_range=None,
        energy_range_nai=None,
        energy_range_bgo=None,
        stat="PG-Stat",
        default_values=None,
        free=None,
        plot_results=False,
        ):
        """
        Perform a line search over a range of energies to detect spectral features.

        This method systematically searches for spectral lines by fitting a Gaussian line
        component at different energies and comparing the fit statistics to a baseline
        model without the line.

        Parameters
        ----------
        models : list of str, optional
            List of model names to use for the baseline fit. Default is ["comp"].
        line_energy_start : float, optional
            Starting energy (in keV) for the line search. Default is 10.
        line_width : float, optional
            Width of the Gaussian line (in keV) and the step size for the energy grid.
            Default is 5.0.
        src_range : tuple or None, optional
            Source time range for spectral fitting. Default is None.
        view_range : tuple or None, optional
            Viewing angle range for spectral fitting. Default is None.
        energy_range_nai : tuple or None, optional
            Energy range (in keV) for NaI detectors. Default is None.
        energy_range_bgo : tuple or None, optional
            Energy range (in keV) for BGO detectors. Default is None.
        stat : str, optional
            Statistical method to use for fitting. Default is "PG-Stat".
        default_values : array-like or None, optional
            Default parameter values for the model. Default is None.
        free : array-like or None, optional
            Boolean array indicating which parameters are free. Default is None.
        plot_results : bool, optional
            If True, plot the delta statistic vs line energy. Default is False.

        Returns
        -------
        None
            Results are printed to console and optionally plotted.

        Notes
        -----
        The method performs the following steps:
        1. Fits a baseline model without a line component
        2. Adds a Gaussian line component at each energy in the search grid
        3. Compares the fit statistic to the baseline using delta statistic
        4. Reports results in a table format
        5. Optionally plots delta statistic vs energy with 3σ and 5σ thresholds

        The delta statistic can be interpreted as a measure of line significance,
        with values > 9 corresponding to ~3σ and > 25 to ~5σ detection.
        """

        self.models = models or self.models

        # Perform an initial fit without the line to get baseline statistics
        self.fit_spectra(
            models=models,
            src_range=src_range,
            view_range=view_range,
            energy_range_nai=energy_range_nai,
            energy_range_bgo=energy_range_bgo,
            stat=stat,
            default_values=default_values,
            free=free,
        )

        baseline_specfitter = self.specfitter
        baseline_statistic = baseline_specfitter.statistic
        baseline_dof = baseline_specfitter.dof

        if self.specfitter.success == False:
            print("\nWarning: Baseline fit did not converge.")
            default_values = self.fit_model.default_values
            free = self.fit_model.free
        else:
            # Saving the best fit parameters to use as default values for line fits
            fit_values = baseline_specfitter.parameters
            default_values = self.fit_model.default_values
            free = self.fit_model.free

            # Convert defaults to an array so we can index it
            default_values = np.array(default_values, dtype=float)

            # Insert specfitter parameters into the free positions
            default_values[np.array(free, dtype=bool)] = fit_values

        # Adding the line model
        combined_model = self.models.copy()
        combined_model.append("GaussLine")

        # Adding the line parameters
        default_values = np.concatenate(
            [default_values, np.array([1e-2, line_energy_start, line_width])]
        )
        free = np.concatenate([free, np.array([True, False, False], dtype=bool)])

        print(
            f"\nBaseline fit statistic: {baseline_statistic:.2f} with DoF: {baseline_dof:.2f}"
        )

        line_energies = np.arange(
            line_energy_start, self.energy_range_nai[1], line_width
        )
        delta_stats = []
        converged = []

        for energy in line_energies:
            print(f"\nAdding line at {energy} keV...")

            default_values[5] = energy

            # Perform the spectral fit
            new_statistic = self.fit_spectra(
                models=combined_model,
                src_range=src_range,
                view_range=view_range,
                energy_range_nai=energy_range_nai,
                energy_range_bgo=energy_range_bgo,
                stat=stat,
                default_values=default_values,
                free=free,
            )

            # Get the delta statistic
            # new_statistic = self.specfitter.statistic
            delta_stat = baseline_statistic - new_statistic
            delta_stats.append(delta_stat)
            converged.append(self.specfitter.success)

        # Print results as a table
        header = f"{'Line Energy (keV)':>18s} {'ΔStat':>12s} {'Converged':>10s}"
        print("\n" + header)
        print("-" * (18 + 1 + 12 + 1 + 10))
        for energy, delta_stat, success in zip(line_energies, delta_stats, converged):
            print(f"{energy:18.2f} {delta_stat:12.2f} {str(success):>10s}")

        if plot_results == True:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.step(line_energies, delta_stats)
            ax.plot([1, 1000], [25, 25], "r--", lw=0.5, label="5-σ")
            ax.plot([1, 1000], [9, 9], "r:", lw=0.5, label="3-σ")
            ax.set_xlabel("Line Energy (keV)")
            ax.set_ylabel("Delta Statistic")
            ax.set_title("Line Search Results - GRB %s" % self.name)
            ax.set_ylim(0, 30)
            ax.set_xlim(min(line_energies), max(line_energies))
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")

            fig.tight_layout()
            fig.show()

    def rebin_by_time(self, resolution, time_range=None):

        if time_range is None:
            time_range = self.src_range

        rebinned_data = self.data.rebin_time(
            rebin_by_time, resolution, time_range=time_range
        )
        self.data = GbmDetectorCollection.from_list(
            rebinned_data, names=self.data.detector()
        )

        return

    def combine_by_factor(self, factor, time_range=None):

        if time_range is None:
            time_range = self.src_range

        self.data = self.data.rebin_energy(combine_by_factor, factor, time_range=range)

    def rebin_by_snr(self, snr, time_range=None):

        if time_range is None:
            time_range = self.src_range

        # rebinned_data = self.data.rebin_time(rebin_by_time, resolution, time_range=time_range)
        rebinned_data = self.data.rebin_time(
            rebin_by_snr, self.backgrounds.counts, snr, time_range=time_range
        )

        rebinned_data_list = []
        for data, background_counts in zip(self.data.data(), self.backgrounds.counts()):
            rebinned_data = data.rebin_time(rebin_by_snr, background_counts, 5)
            rebinned_data_list.append(rebinned_data)

        self.data = GbmDetectorCollection.from_list(
            rebinned_data, names=self.data.detector()
        )

        return

    def save(self, filename=None):
        """
        Save the current state of the object to a file using pickle.

        This method serializes the object's dictionary and saves it to a pickle file.
        If no filename is provided, it generates a default filename based on the
        object's name and data directory.

        Parameters
        ----------
        filename : str or Path, optional
            The path where the object state should be saved. If not provided,
            uses the instance's existing filename. If that's also None, generates
            a default filename in the format 'GRB{name}.pickle' in the data directory.

        Returns
        -------
        None
            This method doesn't return a value but prints a confirmation message
            with the file path where the object was saved.

        Notes
        -----
        The method saves the entire `__dict__` attribute of the object, which contains
        all instance variables. The file is saved in binary pickle format.

        Examples
        --------
        >>> obj.save()  # Uses default or existing filename
        >>> obj.save('custom_path.pickle')  # Uses specified filename
        """

        self.filename = filename or self.filename

        if self.filename == None:
            self.filename = data_path.joinpath(
                self.data_dir, "GRB" + self.name + ".pickle"
            )

        with open(self.filename, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"\nObject state saved to:\n{self.filename}")

        return

    def load(self, filename=None):
        """
        Load the object state from a pickle file.

        This method deserializes and restores the object's state from a previously
        saved pickle file. If no filename is provided, it constructs a default
        filename based on the object's name and data directory.

        Parameters
        ----------
        filename : str, optional
            Path to the pickle file to load. If None, uses the instance's filename
            attribute or constructs a default path using the pattern:
            {data_dir}/GRB{name}.pickle

        Returns
        -------
        None
            The method updates the object's __dict__ in-place with the loaded data.

        Notes
        -----
        The pickle file should contain a dictionary that was previously saved from
        an instance of this class. All keys in the loaded dictionary will overwrite
        corresponding attributes in the current instance.

        Examples
        --------
        >>> obj.load()  # Load from default location
        Object state loaded from:
        /path/to/data/GRBXXXXXX.pickle

        >>> obj.load('/custom/path/data.pickle')  # Load from custom location
        Object state loaded from:
        /custom/path/data.pickle
        """

        self.filename = filename or self.filename

        if self.filename == None:
            self.filename = data_path.joinpath(
                self.data_dir, "GRB" + self.name + ".pickle"
            )

        with open(self.filename, "rb") as f:
            loaded_data = pickle.load(f)  # Use pickle to load the data
        self.__dict__.update(loaded_data)
        print(f"\nObject state loaded from:\n{self.filename}")

        return

    def bayesian_blocks(
        self,
        detector=None,
        p0=0.05,
        factor_thresh=1.5,
        sigma_thresh=None,
        show_plot=True,
        show_episodes=True,
        color_episodes=True,
        analysis_range=None,
        merge_gap=0,
        background=None,
        min_block_duration=0.064,
        ):

        # # Determine which detector to use
        # if detector is None:
        #     detector_index = 0
        # else:
        #     if detector in self.detectors:
        #         detector_index = self.detectors.index(detector)
        #     else:
        #         print(f"\nDetector {detector} not found in loaded detectors: {self.detectors}")
        #         return

        # # Get the background
        # if self.backgrounds is not None:
        #     background = self.backgrounds.to_list()[0]

        # Determine which detector to use
        detector = detector or self.detectors[0]

        # Get the data
        if self.data_type != "tte":
            print(
                "\nWarning: Bayesian blocks analysis is only implemented for evenly binned data."
            )

        # detector = self.detectors[detector_index]

        if analysis_range is not None:
            data = self.data.get_item(detector)
            data = data.data.slice_time(analysis_range[0], analysis_range[1])

            background = self.backgrounds.get_item(detector)

        else:
            data = self.data.get_item(detector).data

        if self.view_range is not None:
            xmin = self.view_range[0]
            xmax = self.view_range[1]
        else:
            xmin = None
            xmax = None

        if detector.startswith("n"):
            data = data.slice_energy(self.energy_range_nai[0], self.energy_range_nai[1])
            emin = self.energy_range_nai[0]
            emax = self.energy_range_nai[1]
        else:
            data = data.slice_energy(self.energy_range_bgo[0], self.energy_range_bgo[1])
            emin = self.energy_range_bgo[0]
            emax = self.energy_range_bgo[1]

        energy_label = f"{emin:.1f} - {emax:.1f} keV"

        # Extract bin information
        bin_centers = data.time_centroids
        bin_counts = data.integrate_energy().counts
        bin_width = data.time_widths[0]
        bin_rates = bin_counts / bin_width

        print("\nPerforming Analysis on Detector %s" % detector)

        # Calculate the Bayesian blocks
        results = bayesian_blocks.get_bayesian_blocks(
            t=bin_centers,
            counts=bin_counts,
            dt=bin_width,
            p0=p0,
            sigma_thresh=sigma_thresh,
            factor_thresh=factor_thresh,
            merge_gap=merge_gap,
            background=background,
            emin=emin,
            emax=emax,
        )

        # Plot the results
        if show_plot == True:
            bayesian_blocks.plot_bayesian_blocks(
                results,
                t=bin_centers,
                counts=bin_counts,
                detector_label=detector,
                show_episodes=show_episodes,
                color_episodes=color_episodes,
                background=None,
                xmin=xmin,
                xmax=xmax,
                energy_label=energy_label,
            )

        self.bblocks_results = results

        # return results
        return

    def calc_t90(
        self,
        analysis_range=None,
        detectors=None,
        plo=0.05,
        phi=0.95,
        show_plot=True,
        shade=True,
        energy_range=None,
        ):
        """
        Calculate the T90 interval (and related timestamps) for a light curve.

        This method computes the interval between the times at which the cumulative
        background-subtracted counts reach the plo and phi fractions of the total
        counts (by default 0.05 and 0.95 respectively), commonly referred to as
        T05, T95 and T90 = T95 - T05. The procedure uses binned light-curve counts
        returned by self.get_lightcurve_counts(...), constructs bin edges from the
        returned times and dt, forms a cumulative sum of counts, and linearly
        interpolates within the bin containing a given fractional threshold.

        Parameters
        ----------
        analysis_range : tuple or list of two floats, optional
            (tmin, tmax) time range in seconds to use for the analysis. If None,
            the method will use self.view_range or self.src_range (in that order).
        detectors : str or sequence of str, optional
            Detector id or list of detector ids to include. If None, the function
            defaults to self.detectors but excludes detectors whose id contains
            'b' or 'B' (i.e., typically BGO detectors). If a single string is
            provided, it will be converted to a 1-element list.
        plo : float, optional
            Lower fractional threshold for start time (default 0.05).
        phi : float, optional
            Upper fractional threshold for stop time (default 0.95).
        show_plot : bool, optional
            If True (default), a two-panel matplotlib figure is produced showing
            the cumulative fraction (top) and counts per bin (bottom) with vertical
            lines at T05 and T95 and an optional shaded T90 region.
        shade : bool, optional
            If True (default) shade the T90 region on the plots.
        energy_range : tuple or list of two floats, optional
            (emin, emax) energy range in keV to integrate for the light curve.
            If None, the routine uses self._energy_range_nai.

        Returns
        -------
        dict
            A dictionary containing:
              - "T05": float
                  Time (s) when the cumulative counts first reach plo * total_counts.
              - "T95": float
                  Time (s) when the cumulative counts first reach phi * total_counts.
              - "T90": float
                  t95 - t05 in seconds.
              - "N_counts": float
                  Total background-subtracted counts used in the T90 computation.

        Example
        -------
        Assuming `spec` is an instance that implements get_lightcurve_counts:
            result = spec.calc_t90(analysis_range=(0, 20), detectors=['n0','n1'],
                                   plo=0.05, phi=0.95, show_plot=True,
                                   energy_range=(50.0, 300.0))
            # result -> {"T05": ..., "T95": ..., "T90": ..., "N_counts": ...}

        """

        if self.backgrounds is None:
            print("\nError: Backgrounds must be defined to calculate T90.")
            return

        print("\nCalculating T90...\n")

        # Set the temporal and spectral ranges
        analysis_range = analysis_range or self.view_range or self.data.time_range()[0]
        energy_range = energy_range or self._energy_range_nai

        print(
            "Using Temporal Range: %s to %s sec"
            % (analysis_range[0], analysis_range[1])
        )
        print("Using Energy Range: %s to %s keV\n" % (energy_range[0], energy_range[1]))

        # Set which detectors to use
        if detectors is None:
            # Use self.detectors, but exclude any BGO detectors
            detectors = [d for d in self.detectors if "b" not in d.lower()]
        else:
            # Ensure detectors is a list-like container
            if not isinstance(detectors, (list, tuple)):
                detectors = [detectors]

        detector_label = (
            " + ".join(detectors) if len(self.detectors) > 1 else detectors[0]
        )
        energy_label = f"{energy_range[0]:.1f} - {energy_range[1]:.1f} keV"

        times, dt, counts = self.get_lightcurve_counts(
            detectors=detectors,
            energy_range=energy_range,
            time_range=analysis_range,
            subtract_bkgd=True,
        )

        tstart = np.asarray(times - 0.5 * dt, dtype=float)
        tstop = np.asarray(times + 0.5 * dt, dtype=float)

        if len(tstart) != len(tstop):
            raise ValueError("tstart and tstop must have the same length.")
        tedges = np.concatenate(([tstart[0]], tstop))

        counts = np.asarray(counts, dtype=float)
        if len(tedges) != len(counts) + 1:
            raise ValueError("tedges must have one more element than counts.")

        # --- Compute cumulative counts ---
        # counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        cumulative_counts = np.cumsum(counts)
        total_counts = cumulative_counts[-1]
        if total_counts <= 0:
            raise ValueError("Total counts are zero; cannot compute T90.")
            return

        # --- Determine times at fractional thresholds ---
        def time_at_threshold(threshold):
            k = np.searchsorted(cumulative_counts, threshold)
            if k == 0:
                return tedges[0]
            if k >= len(counts):
                return tedges[-1]
            c_prev = cumulative_counts[k - 1]
            frac = (threshold - c_prev) / counts[k]
            return tedges[k] + frac * (tedges[k + 1] - tedges[k])

        C05, C95 = plo * total_counts, phi * total_counts
        t05, t95 = time_at_threshold(C05), time_at_threshold(C95)
        t90 = t95 - t05

        frac = cumulative_counts / total_counts
        tmid = 0.5 * (tedges[:-1] + tedges[1:])  # midpoints for plotting counts

        # --- Plot cumulative distribution ---
        if show_plot == True:

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0},
            )

            # --- Top: cumulative fraction ---
            ax1.step(
                tedges[1:], frac, where="post", lw=1.0, label="Cumulative Fraction"
            )
            ax1.axvline(t05, color="green", ls="--", lw=1, label=f"T05 = {t05:.3f} sec")
            ax1.axvline(t95, color="green", ls="--", lw=1, label=f"T95 = {t95:.3f} sec")
            if shade:
                ax1.axvspan(
                    t05, t95, alpha=0.1, color="green", label=f"T90 = {t90:.3f} sec"
                )

            ax1.set_ylabel("Cumulative Fraction")
            ax1.set_ylim(-0.02, 1.02)
            ax1.grid(alpha=0.3)
            ax1.legend(loc="lower right")

            # --- Bottom: counts per bin ---
            ax2.step(tedges[1:], counts, where="post", lw=1.0, label="Counts per bin")
            ax2.axvline(t05, color="green", lw=1, ls="--")
            ax2.axvline(t95, color="green", lw=1, ls="--")
            if shade:
                ax2.axvspan(t05, t95, alpha=0.1, color="green")

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Counts")
            ax2.grid(alpha=0.3)
            # ax2.legend(loc="upper right")

            # Add a detector label
            ax2.text(
                0.0175,
                0.95,
                detector_label,
                transform=ax2.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
            ax2.text(
                max(0.01, 1.0 - 0.00925 * len(energy_label)),
                0.95,
                energy_label,
                transform=ax2.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

            fig.tight_layout()
            fig.show()

        print(f"T05: {t05:.3f} s")
        print(f"T95: {t95:.3f} s")
        print(f"T90: {t90:.3f} s\n")

        self.t90 = t90

        return {"T05": t05, "T95": t95, "T90": t90, "N_counts": total_counts}

    def rayleigh_scan(self, detectors=None, unbinned=True, sum_counts=False, analysis_range=None):
        """
        Perform a Rayleigh scan analysis to detect quasi-periodic oscillations (QPOs) in detector data.

        This method analyzes time-tagged event (TTE) data using the Rayleigh statistic to search for
        periodic signals in the specified frequency range. The analysis can be performed on individual
        detectors or on summed counts from multiple detectors.

        Parameters
        ----------
        detectors : str, list of str, or None, optional
            Detector(s) to analyze. If None, uses self.detectors. Can be a single detector name
            or a list of detector names (default: None).
        unbinned : bool, optional
            Whether to use unbinned data. Currently not implemented in the method body (default: True).
        sum_counts : bool, optional
            If True, sums counts from all specified detectors before performing the scan.
            If False, performs separate scans for each detector (default: False).
        analysis_range : tuple or None, optional
            Custom time range for analysis. Currently not implemented in the method body (default: None).

        Returns
        -------
        results : object
            Results from the Rayleigh scan analysis, returned by rayleigh_scan.run().
            The exact structure depends on the rayleigh_scan module implementation.

        Notes
        -----
        - The method automatically loads TTE data without binning.
        - For NaI detectors (names starting with 'n'), uses self.energy_range_nai for energy filtering.
        - For BGO detectors, uses self.energy_range_bgo for energy filtering.
        - Time slicing is performed using self.src_range.
        - The Rayleigh scan searches frequencies from 0.5 to 800 Hz with 4000 frequency bins.
        - Progress information is printed to stdout during execution.

        Examples
        --------
        >>> specter_obj.rayleigh_scan(detectors=['n0', 'n1'], sum_counts=True)
        Performs a Rayleigh scan on the combined counts from detectors n0 and n1.

        >>> specter_obj.rayleigh_scan(detectors='n0', sum_counts=False)
        Performs a Rayleigh scan on detector n0 only.
        """

        print("detectors:", detectors)
        print("self.detectors:", self.detectors)
        detectors = detectors or self.detectors
        print("selected detectors:", detectors)

        # Ensure detectors is a list-like container
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]

        import rayleigh_scan

        print("\nFinding QPO using Rayleigh method...")

        self.load_tte(bin=False)

        if sum_counts == False:

            for index in range(len(self.detectors)):

                detector = self.detectors[index]

                data = self.data.data()[index]
                data_slice = data.time_slice(self.src_range[0], self.src_range[1])

                if detector.startswith("n"):
                    data_slice = data_slice.energy_slice(
                        self.energy_range_nai[0], self.energy_range_nai[1]
                    )
                else:
                    data_slice = data_slice.energy_slice(
                        self.energy_range_bgo[0], self.energy_range_bgo[1]
                    )

                times = data_slice.times

                print("\nDetector: %s" % detector)
                print(
                    "Time range: %.2f-%.2f s" % (self.src_range[0], self.src_range[1])
                )
                print(
                    "Energy range: %d-%d keV"
                    % (self.energy_range_nai[0], self.energy_range_nai[1])
                    if detector.startswith("n")
                    else "Energy range: %d-%d keV"
                    % (self.energy_range_bgo[0], self.energy_range_bgo[1])
                )
                print("Number of events: %d" % len(times))

                print("\nPerforming Rayleigh scan...")
                results = rayleigh_scan.run(
                    times, fmin=0.5, fmax=800, nfreq=4000, dt_plot=0.064
                )
                print("Done")

        elif sum_counts == True:

            times_all = np.array([])
            for index in range(len(self.detectors)):

                detector = self.detectors[index]
                data = self.data.data()[index]
                data_slice = data.time_slice(self.src_range[0], self.src_range[1])

                if detector.startswith("n"):
                    data_slice = data_slice.energy_slice(
                        self.energy_range_nai[0], self.energy_range_nai[1]
                    )
                else:
                    data_slice = data_slice.energy_slice(
                        self.energy_range_bgo[0], self.energy_range_bgo[1]
                    )

                times = data_slice.times
                times_all = np.concatenate((times_all, times))

            detector = (
                " + ".join(self.detectors)
                if isinstance(self.detectors, (list, tuple))
                else str(self.detectors)
            )

            print("\nDetector: %s" % detector)
            print("Time range: %.2f-%.2f s" % (self.src_range[0], self.src_range[1]))
            print(
                "Energy range: %d-%d keV"
                % (self.energy_range_nai[0], self.energy_range_nai[1])
                if detector.startswith("n")
                else "Energy range: %d-%d keV"
                % (self.energy_range_bgo[0], self.energy_range_bgo[1])
            )
            print("Number of events: %d" % len(times_all))

            print("\nPerforming Rayleigh scan...")
            results = rayleigh_scan.run(
                times_all, fmin=0.5, fmax=800, nfreq=4000, dt_plot=0.064
            )

    def leahy_scan(self, detectors=None, sum_counts=False, analysis_range=None, energy_range=None):
        """
        Perform a Leahy periodogram scan for one or more detectors.

        This method slices and integrates the detector count data over a specified
        time and energy interval, accumulates counts across the requested detectors,
        and calls leahy.search(...) to perform the Leahy periodogram scan.

        Parameters
        ----------
        detectors : str or list of str, optional
            Detector name or iterable of detector names to include (e.g. "n0", "b0").
            If None, uses self.detectors. A single detector string will be accepted
            and wrapped into a list. Detector names beginning with "n" are treated
            as NaI detectors for default energy ranges; all others use the BGO default.
        sum_counts : bool, optional
            Present for API compatibility. Not used by the current implementation.
        analysis_range : tuple(float, float), optional
            Time interval (t_start, t_stop) in seconds used to slice the data. If
            omitted the method uses self.view_range or, if that is None, self.src_range.
            The chosen analysis_range is stored back to self.analysis_range.
        energy_range : tuple(float, float), optional
            Energy interval (emin, emax) in keV used for per-detector energy integration.
            If omitted for a given detector, defaults to self.energy_range_nai for
            detectors whose name starts with "n", otherwise defaults to
            self.energy_range_bgo. Note: when multiple detectors are processed and
            energy_range is None, the last detector's default is the one used for the
            printed energy label and the leahy.search call.

        Returns
        -------
        None

        Examples
        --------
        # Scan a single detector
        self.leahy_scan(detectors='n0', analysis_range=(0.0, 50.0), energy_range=(8.0, 200.0))

        # Scan multiple detectors (counts will be summed across detectors)
        self.leahy_scan(detectors=['n0', 'n1', 'b0'])
        """

        print("\nCalculating Leahy Periodogram...")

        detectors = detectors or self.detectors

        # Ensure detectors is a list-like container
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]

        # Set the temporal and spectral ranges
        self.analysis_range = analysis_range or self.view_range or self.src_range

        counts_all = None
        for detector in detectors:

            print("Processing detector:", detector)

            if detector.startswith("n"):
                energy_range = energy_range or self.energy_range_nai
            else:
                energy_range = energy_range or self.energy_range_bgo

            # Slice the data
            data = self.data.get_item(detector)
            data_slice = data.data.slice_time(
                self.analysis_range[0], self.analysis_range[1]
            )
            data_slice = data_slice.integrate_energy(
                emin=energy_range[0], emax=energy_range[1]
            )

            dt = data_slice.widths[0]
            times = data_slice.centroids

            if counts_all is None:
                counts_all = np.array(
                    data_slice.counts, dtype=float
                )  # allocate once, ensure float
            else:
                np.add(
                    counts_all, data_slice.counts, out=counts_all
                )  # in-place accumulation

        detector_label = " + ".join(detectors) if len(detectors) > 0 else detectors
        energy_label = f"{energy_range[0]:.1f} - {energy_range[1]:.1f} keV"

        print("\nDetector: %s" % detector_label)
        print(
            "Time range: %.2f-%.2f s" % (self.analysis_range[0], self.analysis_range[1])
        )
        print("Energy range: %d-%d keV" % (energy_range[0], energy_range[1]))

        print("\nPerforming Leahy scan...")
        leahy.search(
            times,
            counts_all,
            dt,
            fmin=0.1,
            fmax=10,
            detector_label=detector_label,
            energy_label=energy_label,
        )
        print("\nDone.")

    def show_gui(self, detector=None):
        """
        Open and display the GUI windows associated with this GRB object.
        This method ensures a single QApplication exists (creating one if necessary),
        then creates and shows the primary file manager window and one or more PHA
        spectrum viewers. If a specific detector is supplied, only a viewer for that
        detector is opened; otherwise a viewer is opened for each detector listed in
        self.detectors. Existing open viewers for a detector are detected and not
        duplicated. If a spectral fitter (self.specfitter) is present, a FitPlotter
        window is also created and shown.
        Parameters
        ----------
        detector : optional
            Identifier or object representing a single detector to open a viewer for.
            If None (the default), viewers are opened for all detectors in
            self.detectors.
        Returns
        -------
        None
            This method performs GUI side effects and does not return a value.

        """

        # Ensure exactly one QApplication exists
        app = QApplication.instance()

        if app is None:
            app = QApplication(sys.argv or [])

        # self._file_manager = FileManager(grb=self)
        # self._file_manager.show()

        # If a specific detector is requested, just use that;
        # otherwise, open viewers for all detectors.
        detectors = [detector] if detector is not None else self.detectors

        for det in detectors:

            # Skip if a viewer for this detector already exists
            if any(viewer.detector == det for viewer in self._pha_viewers):
                continue

            # Create and show a new viewer
            viewer = PhaViewer(grb=self, detector=det)
            viewer.show()

            # Add the viewer to the list of open windows
            self._pha_viewers.append(viewer)

        # Create the fit display
        if self.specfitter is not None:
            self._fit_plotter = FitPlotter(grb=self)
            self._fit_plotter.show()

        return

    def update_gui_windows(self, reset_ylim=False, update_data=False):
        """
        Update GUI windows for PHA viewers and the fit plotter.

        This method iterates over any PHA viewer windows tracked on the instance
        and triggers a redraw of their plots. It can optionally reset the
        vertical axis limits or refresh viewer data from the instance data store
        before requesting the redraw. If a fit-plotter is present, it will also
        be asked to update its window.

        Parameters
        ----------
        reset_ylim : bool, optional
            If True, reset the stored previous y-axis limits for each PHA viewer
            (by setting pha_viewer._prev_ylim to None) so that the next draw will
            recompute axis limits. Default is False.
        update_data : bool, optional
            If True, refresh each viewer's data by assigning
            pha_viewer.data = self.data.get_item(pha_viewer.detector) before
            updating the window. Default is False.

        Returns
        -------
        None


        """

        # Loop through each gui window and tell it to redraw its plot
        if len(self._pha_viewers) > 0:
            for pha_viewer in self._pha_viewers:

                if reset_ylim is True:
                    pha_viewer._prev_ylim = None
                if update_data is True:
                    pha_viewer.data = self.data.get_item(pha_viewer.detector)

                pha_viewer.update_window()

        if self._fit_plotter is not None:
            self._fit_plotter.update_window()

    def resolve_spectral_model(self, model_name):
        """Return an instantiated spectral model matching the provided name."""

        if not model_name:
            return None

        raw_name = str(model_name).strip()
        if not raw_name:
            return None

        # Direct class name match (case-sensitive)
        if hasattr(functions, raw_name):
            try:
                return getattr(functions, raw_name)()
            except Exception:
                return None

        normalized = raw_name.lower()

        # Common aliases to canonical class names
        alias_map = {
            "band": "Band",
            "bandfunction": "Band",
            "comp": "Comptonized",
            "comptonized": "Comptonized",
            "powerlaw": "PowerLaw",
            "pl": "PowerLaw",
            "bb": "BlackBody",
            "blackbody": "BlackBody",
        }
        target_name = alias_map.get(normalized)

        # Fallback: case-insensitive match against functions.__all__
        if target_name is None:
            for candidate in getattr(functions, "__all__", []):
                if candidate.lower() == normalized:
                    target_name = candidate
                    break

        if target_name and hasattr(functions, target_name):
            try:
                return getattr(functions, target_name)()
            except Exception:
                return None

        return None

    def plot_orbit(self, save=False):
        """
        Plot the spacecraft orbit around the Earth for a time window centered on the
        instance's reference time and optionally save the resulting figure.

        This method:
        - Ensures a position-history file is available by calling self.get_poshist()
            if self.poshist_file is None.
        - Opens the position-history with GbmPosHist.open(...) and obtains the
            spacecraft frame.
        - Creates a FermiEarthPlot (with a GbmSaa instance) and adds an orbit segment
            spanning 2700 seconds before to 2700 seconds after self.time (using the
            Time(...) format="fermi"), marking the trigger time.
        - Sets the plot title, adjusts the figure size to 9x4 inches, draws and flushes
            the canvas, and briefly pauses to ensure interactive display.
        - If save is True, saves the figure as "<self.data_dir>/orbit_plot_<met>.png"
            with dpi=150 and bbox_inches="tight". ValueError raised during saving is
            caught and printed; other exceptions are not explicitly handled.

        Parameters
        ----------
        save : bool, optional
                If True, attempt to save the generated orbit plot to disk. Default is False.

        Returns
        -------
        None

        Examples
        --------
        # Display the orbit only:
        self.plot_orbit()

        # Display and save the orbit plot:
        self.plot_orbit(save=True)
        """

        if self.poshist_file is None:
            self.get_poshist()

        poshist = GbmPosHist.open(self.poshist_file)
        frame = poshist.get_spacecraft_frame()

        earthplot = FermiEarthPlot(saa=GbmSaa())
        # orbit segment 2700 s prior to our time of interest and 2700 s after
        earthplot.add_spacecraft_frame(
            frame,
            tstart=Time(self.time.fermi - 2700, format="fermi"),
            tstop=Time(self.time.fermi + 2700, format="fermi"),
            trigtime=self.time,
        )
        earthplot.standard_title()

        # Change the plot size
        earthplot._m.figure.set_size_inches(9, 4, forward=True)

        # Show the plot
        earthplot._m.figure.canvas.draw()
        earthplot._m.figure.canvas.flush_events()
        plt.pause(0.001)

        if save:
            try:
                # Save the plot
                earthplot_filename = (
                    str(self.data_dir) + "/" + "orbit_plot_%.f.png" % self.met
                )
                earthplot._m.figure.savefig(
                    earthplot_filename, dpi=150, bbox_inches="tight"
                )
                earthplot._m.figure.canvas.flush_events()
                plt.pause(0.001)

                print("\nPlot saved to: %s" % earthplot_filename)

            except ValueError as err:
                print(err)

    def localization_info(self):
        """
        Display localization information for this GRB instance.

        This method ensures the required localization data are available (trigger
        metadata, position history, and HEALPix localization). If any of those are
        missing it will attempt to load them by calling the instance methods
        get_trigger_data(), get_poshist(), and get_healpix(). If get_healpix() is
        invoked, the resulting HEALPix object is opened and assigned to
        self.healpix.

        Using the HEALPix localization it computes:
        - centroid (best-fit RA/Dec)
        - SkyCoord objects for the best localization, the Sun location, and the
            geocenter location
        - angular separations between the best localization and the Sun / geocenter

        Side effects
        - May call and depend on get_trigger_data(), get_poshist(), and get_healpix().
        - May set self.healpix when the HEALPix file is opened.
        - Writes a human-readable summary to standard output (prints):
            - FSW location (RA, Dec, error)
            - Best location (RA, Dec)
            - 90% confidence area in square degrees
            - Sun location (RA, Dec) and angular distance from the best location
            - Geocenter location (RA, Dec) and angular distance from the best location
            - Fraction of the localization probability that lies on Earth

        Attributes used
        - self.name: identifier used in the printed header
        - self.fsw_location: tuple-like (RA, Dec, error) expected for FSW location
        - self.poshist_file: path or indicator of position-history data
        - self.healpix_file: path to HEALPix localization file
        - self.healpix: object providing centroid, sun_location, geo_location,
            area(confidence), and geo_probability

        Return
        - None

        """

        if self.fsw_location is None:
            self.get_trigger_data()

        if self.poshist_file is None:
            self.get_poshist()

        if self.healpix_file is None:
            self.get_healpix()
            self.healpix = GbmHealPix.open(self.healpix_file)

        centroid = self.healpix.centroid
        best_localization = SkyCoord(
            ra=centroid[0] * u.degree, dec=centroid[1] * u.degree, frame="icrs"
        )
        sun_location = SkyCoord(
            ra=self.healpix.sun_location.ra.degree * u.degree,
            dec=self.healpix.sun_location.dec.degree * u.degree,
            frame="icrs",
        )
        geo_location = SkyCoord(
            ra=self.healpix.geo_location.ra.degree * u.degree,
            dec=self.healpix.geo_location.dec.degree * u.degree,
            frame="icrs",
        )
        sun_distance = best_localization.separation(sun_location)
        earth_distance = best_localization.separation(geo_location)

        print("\nLocalization information for GRB %s:\n" % self.name)
        print(
            "FSW location: RA = %.2f, Dec = %.2f, Err = %.2f"
            % (self.fsw_location[0], self.fsw_location[1], self.fsw_location[2])
        )
        print(
            "Best location: RA = %.2f, Dec = %.2f"
            % (self.healpix.centroid[0], self.healpix.centroid[1])
        )  # best location
        print(
            "90%% confidence area: %.2f (sq. deg)" % self.healpix.area(0.9)
        )  # 90% confidence in units of sq. degrees
        print(
            "Sun location: RA = %.2f, Dec = %.2f, distance = %.2f deg"
            % (
                self.healpix.sun_location.ra.degree,
                self.healpix.sun_location.dec.degree,
                sun_distance.degree,
            )
        )  # sun location
        print(
            "Geocenter location: RA = %.2f, Dec = %.2f, distance = %.2f deg"
            % (
                self.healpix.geo_location.ra.degree,
                self.healpix.geo_location.dec.degree,
                earth_distance.degree,
            )
        )  # geocenter location
        print(
            "Fraction of localization on Earth: %.2f" % self.healpix.geo_probability
        )  # Fraction of localization on Earth

    def calc_detector_angles(self, detectors=None):
        """
        Calculate and display detector angles relative to the GRB localization.

        This method computes the pointing directions of specified Fermi-GBM detectors
        and their angular separations from the best GRB localization. Results are
        printed in a table sorted by angular distance.

        Args:
            detectors (list of str, optional): List of detector names to calculate angles for.
                Valid detector names are 'n0'-'n9', 'na', 'nb', 'b0', 'b1'.
                If None, defaults to all 14 GBM detectors.

        Returns:
            None: Results are printed to stdout.

        Notes:
            - Requires valid poshist and healpix files to be available
            - If healpix_file is None after attempting to load, method returns early
            - Output table shows RA, Dec, and angular separation for each detector
            - Detectors are sorted by angular distance from the GRB localization

        Example:
            >>> grb.calc_detector_angles()  # All detectors
            >>> grb.calc_detector_angles(detectors=['n0', 'n1', 'b0'])  # Specific detectors
        """

        # Use specified detectors, or fall back to self.detectors or self.trigger_detectors
        # detectors = detectors or self.detectors or self.trigger_detectors
        if detectors is None:
            detectors = [
                "n0",
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "n6",
                "n7",
                "n8",
                "n9",
                "na",
                "nb",
                "b0",
                "b1",
            ]

        # Make sure detectors is a list
        if hasattr(detectors, "str"):
            detectors = [detectors]

        # Load poshist and healpix if not already loaded
        if self.poshist is None:
            self.get_poshist()
            self.poshist = GbmPosHist.open(self.poshist_file)
        if self.healpix is None:
            self.get_healpix()
            if self.healpix_file is None:
                return
            self.healpix = GbmHealPix.open(self.healpix_file)

        # Get best localization coordinate
        centroid = self.healpix.centroid
        best_localization = SkyCoord(
            ra=centroid[0] * u.degree, dec=centroid[1] * u.degree, frame="icrs"
        )

        # Get spacecraft frame
        frame = self.poshist.get_spacecraft_frame()

        # Loop through specified detectors and print their angles
        print("\nDetector angles for GRB %s:" % self.name)
        for det in detectors:
            function_name = det + "_pointing"
            angle = getattr(self.healpix, function_name)

            ra = angle.ra.degree
            dec = angle.dec.degree
            distance = angle.separation(best_localization)

            # Accumulate detector angles and print once sorted by distance at the end
            if not hasattr(self, "_det_angle_accum"):
                self._det_angle_accum = []
            self._det_angle_accum.append((det, ra, dec, distance.degree))

            # If this is the last detector, sort and print, then clear accumulator
            try:
                last_det = detectors[-1]
            except Exception:
                last_det = det

            if det == last_det:
                sorted_list = sorted(self._det_angle_accum, key=lambda x: x[3])
                print(
                    f"\n{'Detector':<10}{'RA (deg)':>12}{'Dec (deg)':>12}{'Distance (deg)':>18}"
                )
                print("-" * 52)
                for dname, ra_, dec_, dist in sorted_list:
                    print(f"{dname:<10}{ra_:12.2f}{dec_:12.2f}{dist:18.2f}")
                del self._det_angle_accum

    def calc_time_to_saa(self):
        """
        Calculate and print the time since the last SAA exit and the time until the next SAA entry
        relative to the instance trigger time (self.met).
        This method:
        - Ensures a position history (poshist) is available: if self.poshist_file is None it calls
            self.get_poshist() and opens the poshist file into self.poshist.
        - Reads spacecraft state history from self.poshist.get_spacecraft_states(), expecting a
            mapping containing 'time' and 'saa' entries where the arrays are accessible via
            .value (e.g. astropy Quantity or similar).
        - Interprets 'time' as spacecraft wall-clock or mission-elapsed times and 'saa' as a binary
            flag (1 inside SAA, 0 outside).
        - Locates the index of the time sample nearest to the trigger time self.met.
        - Finds SAA indices (where saa == 1) and determines:
                - the last SAA index before the trigger (SAA exit index), if any
                - the first SAA index after the trigger (SAA entrance index), if any
        - Computes:
                time_from_saa = time_at_last_exit - self.met   (seconds since last SAA exit)
                time_to_saa   = time_at_next_entry - self.met  (seconds until next SAA entry)
        - Prints human-readable messages describing the results, and prints warnings if there
            is no prior SAA exit or no upcoming SAA entry.

        Parameters
        ----------
        self : object
                The instance is expected to provide the following attributes/methods:
                - name (str): identifier used in printed output.
                - met (float): trigger time (same time units as the poshist 'time' array).
                - poshist_file (str or None): path to position history file (can be None).
                - poshist (object): an opened position-history object (used if poshist_file is set).
                - get_poshist() (callable): method to create/download/set self.poshist_file if it is None.
                The poshist object must implement get_spacecraft_states() and return a mapping containing
                keys 'time' and 'saa' with array-like .value attributes.

        Returns
        -------
        None

        """

        print("\nSAA information for GRB %s:" % self.name)

        if self.poshist_file is None:
            self.get_poshist()
            self.poshist = GbmPosHist.open(self.poshist_file)

        # Get spacecraft states
        spacecraft_states = self.poshist.get_spacecraft_states()

        # Extract time and SAA status
        time_met = spacecraft_states["time"].value
        saa = spacecraft_states["saa"].value

        # Find the index of the trigger time and the first saa entry after it
        trigger_index = np.abs(time_met - self.met).argmin()
        saa_index = np.where(saa == 1)
        saa_idx = np.where(saa == 1)[0]
        diff_index = saa_idx - trigger_index
        neg = np.where(diff_index < 0)[-1]
        pos = np.where(diff_index > 0)[0]
        if neg.size == 0:
            print("\nNo prior SAA exit found before trigger.")
        if pos.size == 0:
            print("\nNo upcoming SAA entry found after trigger.")
            return

        # replace saa_index with the first SAA index after the trigger
        saa_exit_index = saa_idx[neg[-1]]
        saa_entrance_index = saa_idx[pos[0]]

        time_from_saa = time_met[saa_exit_index] - self.met
        time_to_saa = time_met[saa_entrance_index] - self.met

        print("\nTime since SAA exit: %.2f sec" % time_from_saa)
        print("Time to next SAA entry: %.2f sec\n" % time_to_saa)

    def plot_localization(self):
        """
        Plot the GBM localization on an equatorial sky plot.

        This method prepares and displays an EquatorialPlot containing the spacecraft
        frame and the localization contours derived from the instance's healpix and
        poshist data.

        Behavior
        --------
        - If self.healpix_file is None, calls self.get_healpix() to obtain it.
        - If self.poshist_file is None, calls self.get_poshist() to obtain it.
        - Opens localization and position-history objects via GbmHealPix.open(...) and
            GbmPosHist.open(...).
        - Retrieves the spacecraft frame from the poshist object and adds it to the
            EquatorialPlot.
        - Attempts to overlay localization contours with skyplot.add_localization(...);
            exceptions raised during contour addition are suppressed.
        - Adjusts the figure size, forces a canvas redraw, flushes GUI events, and
            briefly pauses to allow interactive display.

        Parameters
        ----------
        None
                This method does not take any parameters.

        Returns
        -------
        None
                The function updates and displays the plot as a side effect and does not
                return a value.

        Example
        -------
        # Assuming this method is part of an instance with get_healpix/get_poshist:
        self.plot_localization()
        """

        if self.healpix_file is None:
            self.get_healpix()
            if self.healpix_file is None:
                return
        if self.poshist_file is None:
            self.get_poshist()
            if self.poshist_file is None:
                return

        self.healpix = GbmHealPix.open(self.healpix_file)

        poshist = GbmPosHist.open(self.poshist_file)
        frame = poshist.get_spacecraft_frame()

        skyplot = EquatorialPlot()
        skyplot.add_frame(frame[0])

        try:
            skyplot.add_localization(
                self.healpix, gradient=False, clevels=[0.5, 0.9, 0.95]
            )
            # skyplot.add_localization(loc, gradient=False)
        except:
            pass

        # Change the plot size
        skyplot._figure.set_size_inches(9, 4, forward=True)

        # Show the plot
        skyplot._figure.canvas.draw()
        skyplot._figure.canvas.flush_events()
        plt.pause(0.001)

        return

    def calc_structure_function(
        self,
        analysis_range=None,
        detector=None,
        min_dt=0.064,
        max_dt=1.0,
        snr=1,
        max_lag=25.0,
        mc_trials=400,
        mc_quantiles=(0.05, 0.995),
        mvt_rule="mc",
        min_pairs_per_lag=50,
        ):
        """
        Calculate the structure function (SF) of a time series (counts per bin) and estimate
        the minimum variability timescale (MVT) using Haar-wavelet based methods.

        This method:
        - Selects a detector and time/energy sub-range from the instance data/backgrounds.
        - Integrates counts over the selected energy range and subtracts background counts.
        - Propagates Poisson uncertainties (source and background) in quadrature.
        - Calls sf.calc_structure_function(...) to compute the SF, analytic Poisson floor,
            Monte Carlo noise bands and an MVT estimate.
        - Produces a two-panel matplotlib figure: (1) counts and background vs time, and
            (2) SF(τ) vs τ (log-log) with analytic noise floor, MC band (if available) and
            a vertical line at the estimated MVT.
        - Prints diagnostic messages (analysis range, MVT) to stdout.

        Parameters
        ----------
        analysis_range : tuple[float, float] or None, optional
                Time range (t_start, t_stop) to analyze. If None, uses self.view_range if set,
                otherwise the full available time range. Times are in the same units as the
                instance time centroids (seconds).
        detector : str or None, optional
                Detector identifier to use (must be present in self.detectors). If None, the
                first detector (index 0) is used. If the given detector is not found the method
                prints a message and returns immediately.
        min_dt : float, optional
                Minimum time resolution to consider (seconds). Present for API compatibility;
                not directly used in the current implementation.
        max_dt : float, optional
                Maximum time resolution to consider (seconds). Present for API compatibility;
                not directly used in the current implementation.
        snr : float, optional
                Signal-to-noise threshold parameter (present for API compatibility; not used
                in the current implementation).
        max_lag : float, optional
                Maximum lag τ (in seconds) to evaluate the structure function. Passed to the
                underlying SF routine.
        mc_trials : int, optional
                Number of Monte Carlo trials to estimate the noise band (if mvt_rule uses MC).
        mc_quantiles : tuple[float, float], optional
                Lower and upper quantiles to use for the Monte Carlo noise band (e.g. (0.05, 0.995)).
        mvt_rule : str, optional
                Rule used to select the MVT from the SF results. Typical values are "mc"
                (Monte Carlo based) or "analytic" (if supported by the SF routine).
        min_pairs_per_lag : int, optional
                Minimum number of pairs required per lag bin when computing the SF. Passed to
                the underlying SF routine.

        Returns
        -------
        None

        Notes
        -----
        - Background subtraction is performed per-bin (counts - bkgd_counts). Poisson
            uncertainties are propagated in quadrature between source and background
            components.
        """

        if self.backgrounds is None:
            print("\nError: No background must be defined to calculate structure function.")
            return

        range = analysis_range or self.view_range or None

        print("\nCalculating minimum variability timescale using Haar wavelets...")

        # Determine which detector to use
        if detector is None:
            detector_index = 0
        else:
            if detector in self.detectors:
                detector_index = self.detectors.index(detector)
            else:
                print(
                    f"\nDetector {detector} not found in loaded detectors: {self.detectors}"
                )
                return

        detector = self.detectors[detector_index]
        print("Processing detector:", detector)

        # from haar import haar_power_mod

        if range is not None:
            print('Checkpoint 1 - analysis range:', range)
            data = self.data.data()[detector_index]
            data = data.slice_time(range[0], range[1])
        else:
            print('Checkpoint 2 - analysis range:', range)
            data = self.data.data()[detector_index]

        if detector.startswith("n"):
            data = data.slice_energy(self.energy_range_nai[0], self.energy_range_nai[1])
            emin = self.energy_range_nai[0]
            emax = self.energy_range_nai[1]
        else:
            data = data.slice_energy(self.energy_range_bgo[0], self.energy_range_bgo[1])
            emin = self.energy_range_bgo[0]
            emax = self.energy_range_bgo[1]

        bkgd = self.backgrounds.slice_time(range[0], range[1])
        bkgd = bkgd[detector_index].integrate_energy(emin=emin, emax=emax)
        bkgd_counts = bkgd.counts.squeeze()

        # Extract bin information
        counts = data.integrate_energy().counts
        bin_width = data.time_widths[0]
        src_rates = counts / bin_width

        # ensure 1-D arrays (shape will be (8066,) which is the standard 1-D shape)
        counts = np.ravel(counts)
        error = np.ravel(np.sqrt(counts))
        n_bins = counts.size  # or len(counts)

        counts_bkgd_subtracted = counts - bkgd_counts
        error = np.sqrt(error**2 + np.sqrt(bkgd_counts) ** 2)
        # rate = counts_bkgd_subtracted/bin_width
        # rate_error = np.ravel(np.sqrt(counts)/bin_width)
        t = data.time_centroids

        res = sf.calc_structure_function(
            counts=counts,
            bg_counts=bkgd_counts,
            dt=bin_width,
            max_lag=max_lag,
            mc_trials=mc_trials,
            mc_quantiles=mc_quantiles,
            mvt_rule=mvt_rule,
            # mvt_rule="analytic",
            # analytic_nsigma=10.0,
            min_pairs_per_lag=min_pairs_per_lag,
            # bg_var_counts=...  # optional if you have per-bin bg uncertainty
        )

        print("Estimated MVT (s):", res.mvt)

        # Create two independent subplots in the same figure (no shared axes)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=False, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]}
        )

        # Top: counts vs t
        ax1.step(t, counts, where="mid", color="tab:blue", label="counts")
        ax1.plot(
            t,
            bkgd_counts,
            color="tab:orange",
            label="background (counts/bin)",
            linestyle="--",
        )
        # ax1.axhline(bg_counts, color="tab:gray", linestyle="--", label="background (counts/bin)")
        ax1.set_ylabel("Counts per bin")
        ax1.legend(loc="upper right")
        ax1.grid(alpha=0.3)
        ax1.set_xscale("linear")

        ax2.loglog(res.tau, res.sf, marker="o", linestyle="-", label="SF (net rate)")
        ax2.loglog(
            res.tau, res.sf_noise, linestyle="--", label="Analytic Poisson floor"
        )
        if res.sf_mc_lo is not None:
            ax2.fill_between(
                res.tau, res.sf_mc_lo, res.sf_mc_hi, alpha=0.3, label="MC noise band"
            )
        if res.mvt is not None:
            ax2.axvline(
                res.mvt, linestyle=":", color="k", label=f"MVT ≈ {res.mvt:.3g} s"
            )
        ax2.set_xlabel("Lag τ (s)")
        ax2.set_ylabel("SF(τ) = <(x(t+τ)-x(t))^2>   [x in counts/s]")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig.show()

    def lag_analysis(
        self,
        detectors=None,
        low_energy=(25, 50),
        high_energy=(100, 300),
        analysis_range=None,
        lag_range=[-5, 5],
        poly_order=3,
        poly_fit=False,
        fit_range=[-0.25, 0.25],
        gaussian_fit=False,
        subtract_bkgd=False,
        ):
        """
        Perform lag analysis between two energy bands and visualize the light curves and
        their cross-correlation function (CCF).

        This method:
        - Extracts light curves for two energy bands using self.get_lightcurve_counts.
        - Converts counts to rates using the returned time bin widths.
        - Computes a (normalized) cross-correlation function between the two rate
            time-series and displays it together with the input light curves.
        - Identifies the maximum of the discrete CCF and can optionally refine the lag
            estimate using either a polynomial fit (np.polyfit) or a Gaussian fit
            (scipy.optimize.curve_fit) over a user-specified fit range.
        - Draws and annotates the results on a two-panel matplotlib figure and prints
            summary information to stdout. The method visualizes fitted models and marks
            the refined lags on the CCF plot.

        Parameters
        ----------
        detectors : str or sequence of str, optional
                Detector name or list of detector names to combine. If None, uses
                self.detectors. If a single string is provided it is treated as a single
                detector.
        low_energy : tuple(int, int), optional
                (min_keV, max_keV) for the low-energy band. Default is (25, 50).
        high_energy : tuple(int, int), optional
                (min_keV, max_keV) for the high-energy band. Default is (100, 300).
        analysis_range : tuple(float, float), optional
                (t_start, t_stop) time interval (seconds) over which to extract light
                curves. If None, falls back to self.view_range or self.src_range.
        lag_range : sequence or tuple, optional
                Requested lag plotting range in seconds, e.g. [-5, 5]. (Note: in current
                implementation the full computed lags are plotted; this parameter may be
                reserved for future clipping.)
        poly_order : int, optional
                Polynomial degree to use for the polynomial refinement. The degree is
                adaptively reduced if there are fewer points than requested. Default 3.
        poly_fit : bool, optional
                If True, perform a polynomial fit to the CCF within fit_range to obtain a
                refined lag estimate and plot the fitted polynomial on the CCF subplot.
        fit_range : sequence (float, float), optional
                (lag_min, lag_max) in seconds: the interval of lags over which polynomial
                and/or Gaussian fitting is performed. Default is [-0.25, 0.25].
        gaussian_fit : bool, optional
                If True, fit a Gaussian plus constant baseline to the CCF inside fit_range
                and plot the fitted Gaussian; the Gaussian mean is reported as a refined
                lag estimate.
        subtract_bkgd : bool, optional
                If True, use raw rates to compute the cross-correlation. If False (default),
                the median of each rate series is subtracted before computing the CCF to
                help with stationarity.

        Returns
        -------
        None

        Example
        -------
        # Typical call:
        obj.lag_analysis(detectors=['NaI0','NaI3'], low_energy=(25,50), high_energy=(100,300),
                                         analysis_range=(0,20), poly_fit=True, gaussian_fit=True, fit_range=(-0.25,0.25))

        """
        def gaussian(x, amplitude, mean, stddev, noise):
            return noise + amplitude * np.exp(-(((x - mean) / 4 / stddev) ** 2))

        if subtract_bkgd and (self.backgrounds is None):
            print(
                "\nWarning: Background data not available; cannot subtract background."
            )
            subtract_bkgd = False


        # Set the analysis range
        analysis_range = analysis_range or self.view_range or self.src_range

        print("Performing lag analysis...\n")

        print("Time range: %.2f-%.2f s" % (analysis_range[0], analysis_range[1]))
        print("Low energy range: %d-%d keV" % (low_energy[0], low_energy[1]))
        print("High energy range: %d-%d keV" % (high_energy[0], high_energy[1]))

        # Set the detectors to use
        detectors = detectors or self.detectors

        # Ensure detectors is a list-like container
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]

        times, dt, counts_low = self.get_lightcurve_counts(
            detectors=detectors,
            energy_range=low_energy,
            time_range=analysis_range,
            subtract_bkgd=subtract_bkgd,
        )

        times, dt, counts_high = self.get_lightcurve_counts(
            detectors=detectors,
            energy_range=high_energy,
            time_range=analysis_range,
            subtract_bkgd=subtract_bkgd,
        )

        rates_low = counts_low / dt
        rates_high = counts_high / dt

        # Generate plot label
        detector_label = " + ".join(detectors) if len(detectors) > 1 else detectors[0]

        # Create two independent subplots in the same figure (no shared axes)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=False, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]}
        )

        # Plot the light curves
        ax1.step(
            times,
            rates_low,
            where="mid",
            color="tab:green",
            lw=1,
            label="%d-%d keV" % (low_energy[0], low_energy[1]),
        )
        ax1.step(
            times,
            rates_high,
            where="mid",
            color="tab:orange",
            lw=1,
            label="%d-%d keV" % (high_energy[0], high_energy[1]),
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Counts/sec")
        ax1.legend(loc="upper right")
        ax1.grid(alpha=0.3)
        ax1.set_xscale("linear")
        ax1.set_title(f"Lag Analysis")
        ax1.text(
            0.0175,
            0.95,
            detector_label,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        fig.tight_layout()
        fig.show()

        # Cross-correlation (normalized)
        if subtract_bkgd == True:
            ccf = correlate(rates_low, rates_high, mode="full", method="direct")
        else:
            ccf = correlate(
                rates_low - np.median(rates_low),
                rates_high - np.median(rates_high),
                mode="full",
                method="direct",
            )

        lags = correlation_lags(len(rates_low), len(rates_high), mode="full") * dt[0]
        ccf /= np.std(rates_low) * np.std(rates_high) * len(rates_low)

        # Plot the cross-correlation function
        ax2.step(
            lags, ccf, where="mid", color="tab:blue", lw=1, label="Cross-correlation function"
        )
        ax2.set_xlabel("Lag (s)")
        ax2.set_ylabel("Cross-correlation")
        # ax2.grid(True, ls=":", alpha=0.5)
        ax2.grid(alpha=0.3)

        index_max = np.argmax(ccf)
        lag_max = lags[index_max]
        ax2.axvline(
            lag_max,
            color="red",
            ls="--",
            lw=0.75,
            label=f"Max CCF lag = {lag_max:.4f} s",
        )

        if poly_fit == True:

            # Fit a polynomial to the CCF inside the requested fit_range to get a refined lag
            fit_min, fit_max = fit_range[0], fit_range[1]
            mask = (lags >= fit_min) & (lags <= fit_max)
            n_points = np.count_nonzero(mask)

            refined_lag = None
            x_poly = None
            ccf_poly = None

            if n_points < 2:
                # Not enough points to fit: fallback to the peak of the discrete CCF
                idx_max = np.argmax(ccf)
                refined_lag = lags[idx_max]
                print(
                    f"Not enough points in fit_range {fit_range} for polynomial fit. Using discrete peak lag = {refined_lag:.6f} s"
                )
            else:
                # adapt polynomial degree if necessary
                deg = min(poly_order, max(1, n_points - 1))
                try:
                    z = np.polyfit(lags[mask], ccf[mask], deg)
                    p = np.poly1d(z)
                    x_poly = np.linspace(fit_min, fit_max, 2000)
                    ccf_poly = p(x_poly)
                    # take the lag at which the fitted polynomial is maximal
                    refined_lag = x_poly[np.nanargmax(ccf_poly)]
                    print(
                        f"Polynomial fit (deg={deg}) refined lag = {refined_lag:.6f} s"
                    )
                except Exception as e:
                    idx_max = np.argmax(ccf)
                    refined_lag = lags[idx_max]
                    print(
                        f"Polynomial fit failed ({e}). Using discrete peak lag = {refined_lag:.6f} s"
                    )

            # make results available for later plotting/annotation
            lag_max_poly = refined_lag
            lag_fit_x = x_poly
            lag_fit_model = ccf_poly

            # discrete peak
            index_max = np.argmax(ccf)
            lag_max = lags[index_max]
            ax2.axvline(
                lag_max,
                color="red",
                ls="--",
                lw=0.75,
                label=f"Max lag = {lag_max:.4f} s",
            )

            # plot polynomial fit if available
            if lag_fit_x is not None and lag_fit_model is not None:
                ax2.plot(
                    lag_fit_x,
                    lag_fit_model,
                    "--",
                    color="darkred",
                    lw=1.0,
                    label="Polynomial fit",
                )
                # mark the refined lag
                ax2.axvline(
                    lag_max_poly,
                    color="darkred",
                    ls="--",
                    lw=0.75,
                    label=f"Poly lag = {lag_max_poly:.4f} s",
                )

            # Gaussian fit to the same fit_range
        if gaussian_fit == True:

            fit_min, fit_max = fit_range[0], fit_range[1]
            mask = (lags >= fit_min) & (lags <= fit_max)
            n_points = np.count_nonzero(mask)

            x_mask = lags[mask]
            y_mask = ccf[mask]

            print(x_mask)
            print(y_mask)

            if x_mask.size >= 4:
                # initial guesses
                baseline_guess = np.median(y_mask)
                amp_guess = np.max(y_mask) - baseline_guess
                mean_guess = x_mask[np.argmax(y_mask)]
                std_guess = max((fit_max - fit_min) / 6.0, 1e-6)

                p0 = [amp_guess, mean_guess, std_guess, baseline_guess]
                # bounds: amplitude >= 0, mean within fit range, std > 0
                lower = [0.0, fit_min, 1e-8, -np.inf]
                upper = [np.inf, fit_max, np.inf, np.inf]

                popt, pcov = curve_fit(
                    gaussian, x_mask, y_mask, p0=p0, bounds=(lower, upper), maxfev=10000
                )
                gauss_x = np.linspace(fit_min, fit_max, 2000)
                gauss_y = gaussian(gauss_x, *popt)

                lag_max_gauss = popt[1]
                ax2.plot(
                    gauss_x, gauss_y, "--", color="olive", lw=1.0, label="Gaussian fit"
                )
                ax2.axvline(
                    lag_max_gauss,
                    color="olive",
                    ls="--",
                    lw=0.75,
                    label=f"Gauss lag = {lag_max_gauss:.4f} s",
                )
                print(
                    f"Gaussian fit refined lag = {lag_max_gauss:.6f} s (mean), sigma = {popt[2]:.6f} s"
                )

            else:
                print(
                    "Not enough points for Gaussian fit in fit_range; skipping Gaussian fit."
                )

        ax2.legend(loc="upper right")
        fig.show()

        print("\nMaximum CCF (lag): %.2f sec\n" % lag_max)

    def calc_hardness_ratio(
        self,
        detectors=None,
        low_energy=(25, 50),
        high_energy=(100, 300),
        analysis_range=None,
        subtract_bkgd=False,
        sum_counts=False,
        ):
        """
        Calculate and plot the hardness ratio between two energy bands.
        This method computes the hardness ratio HR = C_high / C_low (and an associated
        propagated uncertainty) for one or more detectors using count lightcurves
        returned by self.get_lightcurve_counts and then plots the result with
        self.plot_hardness_ratio.
        
        Parameters
        ----------
        detectors : None, str, or sequence of str, optional
            Detector name or list/tuple of detector names to analyze. If None, the
            method uses self.detectors. If a single string is provided it is treated
            as a single detector. Default: None.
        low_energy : tuple of two floats, optional
            Lower-energy band (inclusive) in keV as (E_min, E_max). Default: (25, 50).
        high_energy : tuple of two floats, optional
            Higher-energy band (inclusive) in keV as (E_min, E_max). Default: (100, 300).
        analysis_range : None or tuple, optional
            Time range (start, stop) to analyze. If None, falls back to self.view_range
            or self.src_range (in that order) as available. Default: None.
        subtract_bkgd : bool, optional
            If True, request that background be subtracted when obtaining lightcurves
            (passed through to self.get_lightcurve_counts). Default: False.
        sum_counts : bool, optional
            If False (default), compute and plot hardness ratio separately for each
            detector in `detectors`. If True, sum counts across all specified
            detectors first and compute/plot a single hardness ratio for the summed
            lightcurve.

        Example
        -------
        # Example usage (conceptual):
        # self.calc_hardness_ratio(detectors=['n0','n1'], low_energy=(25,50),
        #                          high_energy=(100,300), analysis_range=(t0,t1),
        #                          subtract_bkgd=True, sum_counts=False)
        """

        print("\nPlotting hardness ratio...\n")

        print("Low energy range: %d-%d keV" % (low_energy[0], low_energy[1]))
        print("High energy range: %d-%d keV\n" % (high_energy[0], high_energy[1]))

        # Set the analysis range
        analysis_range = analysis_range or self.view_range or self.src_range

        # Set the detectors to use
        detectors = detectors or self.detectors

        # Ensure detectors is a list-like container
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]

        total_counts_low = None
        total_counts_high = None

        if sum_counts == False:

            # Loop through each detector
            for detector in detectors:

                times, time_widths, counts_low = self.get_lightcurve_counts(
                    detectors=detector,
                    time_range=analysis_range,
                    energy_range=low_energy,
                    subtract_bkgd=subtract_bkgd,
                )

                times, time_widths, counts_high = self.get_lightcurve_counts(
                    detectors=detector,
                    time_range=analysis_range,
                    energy_range=high_energy,
                    subtract_bkgd=subtract_bkgd,
                )

                rates_low = counts_low / time_widths
                rates_high = counts_high / time_widths

                # Calculate hardness ratio for this detector individually
                hardness_ratio = counts_high / counts_low
                hardness_ratio_error = hardness_ratio * np.sqrt(
                    (np.sqrt(counts_high) / counts_high) ** 2
                    + (np.sqrt(counts_low) / counts_low) ** 2
                )

                # Generate plot label
                detector_label = detector

                # Plot hardness ratio for this detector
                self.plot_hardness_ratio(
                    times,
                    rates_low,
                    rates_high,
                    hardness_ratio,
                    hardness_ratio_error,
                    low_energy,
                    high_energy,
                    detector_label=detector_label,
                )

        if sum_counts == True:

            times, time_widths, counts_low = self.get_lightcurve_counts(
                detectors=detectors,
                time_range=analysis_range,
                energy_range=low_energy,
                subtract_bkgd=subtract_bkgd,
            )

            times, time_widths, counts_high = self.get_lightcurve_counts(
                detectors=detectors,
                time_range=analysis_range,
                energy_range=high_energy,
                subtract_bkgd=subtract_bkgd,
            )

            rates_low = counts_low / time_widths
            rates_high = counts_high / time_widths

            # Calculate hardness ratio for summed detectors
            hardness_ratio = counts_high / counts_low
            hardness_ratio_error = hardness_ratio * np.sqrt(
                (np.sqrt(counts_high) / counts_high) ** 2
                + (np.sqrt(counts_low) / counts_low) ** 2
            )

            # Generate plot label
            detector_label = (
                " + ".join(self.detectors)
                if len(self.detectors) > 1
                else self.detectors[0]
            )
            # plot_label = f'Summed Detectors {detector_label}'

            # Plot hardness ratio for summed detectors
            self.plot_hardness_ratio(
                times,
                rates_low,
                rates_high,
                hardness_ratio,
                hardness_ratio_error,
                low_energy,
                high_energy,
                detector_label=detector_label,
            )

    def plot_hardness_ratio(
        self,
        times,
        rates_low,
        rates_high,
        hardness_ratio,
        hardness_ratio_error,
        low_energy,
        high_energy,
        detector_label=None,
        ):
        """
        Plot the light curves for two energy bands and their hardness ratio.
        Creates a two-row figure (shared x-axis) where the top subplot shows the
        step light curves for a low and a high energy band, and the bottom subplot
        shows the hardness ratio (H/L) with error bars.
        Parameters
        ----------
        self : object
            Instance reference (method belongs to a class).
        times : array-like
            1-D array of time bin centers (seconds). All series must have the same length.
        rates_low : array-like
            Count rates for the low-energy band (counts/s), same length as `times`.
        rates_high : array-like
            Count rates for the high-energy band (counts/s), same length as `times`.
        hardness_ratio : array-like
            Hardness ratio values (H/L) to plot, same length as `times`.
        hardness_ratio_error : array-like
            1-D array of errors/uncertainties for `hardness_ratio`, same length as `times`.
        low_energy : sequence of two numbers
            Energy range for the low band, e.g. (low_keV_min, low_keV_max). Used to
            construct the legend label for the top panel.
        high_energy : sequence of two numbers
            Energy range for the high band, e.g. (high_keV_min, high_keV_max). Used to
            construct the legend label for the top panel.
        detector_label : str, optional
            Optional short label identifying the detector; placed in the top-left of
            the upper subplot. If None, no explicit detector name is shown.

        Returns
        -------
        matplotlib.figure.Figure
            The Matplotlib Figure object containing the two subplots. The function
            also calls tight_layout() and fig.show() before returning.

        Example
        -------
        fig = obj.plot_hardness_ratio(times, rates_low, rates_high,
                                      hardness_ratio, hardness_ratio_error,
                                      low_energy=(10, 50), high_energy=(50, 300),
                                      detector_label='NaI_0')
        """

        # Create two independent subplots in the same figure (no shared axes)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]}
        )

        # Plot the light curves
        ax1.step(
            times,
            rates_low,
            where="mid",
            color="#659b8a",
            lw=1.0,
            label=str(low_energy[0]) + "-" + str(low_energy[1]) + " keV",
        )
        ax1.step(
            times,
            rates_high,
            where="mid",
            color="#966db4",
            lw=1.0,
            label=str(high_energy[0]) + "-" + str(high_energy[1]) + " keV",
        )
        # ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Count Rate (counts/s)")
        ax1.legend(loc="upper right")
        # ax1.grid(True, ls=":", alpha=0.5)
        ax1.grid(alpha=0.3)
        ax1.set_xscale("linear")
        ax1.set_title(f"Hardness Ratio Analysis")
        # Add a detector label
        ax1.text(
            0.0175,
            0.95,
            detector_label,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        ax2.step(
            times, hardness_ratio, where="mid", lw=1.0, label="Hardness Ratio (H/L)"
        )
        ax2.errorbar(
            times,
            hardness_ratio,
            yerr=hardness_ratio_error,
            fmt="o",
            markersize=0,
            color="dimgrey",
            alpha=0.25,
            lw=1.0,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Hardness Ratio (H/L)")
        # ax2.grid(True, ls=":", alpha=0.5)
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.show()

        return fig

    def get_lightcurve_counts(
        self, detectors=None, time_range=None, energy_range=None, subtract_bkgd=False
        ):
        """
        Calculate and return light curve counts in a specified temporal and energy range for one or more detectors.

        The method sums (optionally background-subtracted) counts from the requested detectors over the requested time
        and energy ranges and returns the time bin centers, bin widths and the summed counts array.

        Parameters
        ----------
        detectors : str or list of str or None, optional
            Detector name or list/tuple of detector names to include (e.g. "n0", "b0"). If None, the instance's
            default detectors (self.detectors) are used.
        time_range : tuple(float, float) or None, optional
            (tstart, tstop) in the same time units used by the data objects. If provided, data (and backgrounds,
            when requested) are sliced to this interval. If None, the full time range of each detector's data is used.
        energy_range : tuple(float, float) or None, optional
            (emin, emax) in keV. If provided, the data (and backgrounds, when requested) are integrated over this
            energy interval. If None, per-detector defaults are used: self.energy_range_nai for detectors whose names
            start with 'n' and self.energy_range_bgo otherwise.
        subtract_bkgd : bool, optional
            If True, subtract the corresponding background (from self.backgrounds) after slicing/integrating it to
            the same time and energy intervals.

        Returns
        -------
        time_centroids : numpy.ndarray
            Array of time bin centers (centroids) corresponding to the returned counts. Note: this value is taken from
            the most recently processed detector in the provided list; callers should ensure detectors share a common
            time grid if they expect a meaningful combined result.
        time_widths : numpy.ndarray
            Array of time bin widths for the returned time bins (same caveat about common time grid as above).
        counts_summed : numpy.ndarray
            1-D array of summed counts across the requested detectors for each time bin. This is returned as a float
            numpy array. If subtract_bkgd is True, the background (bkgd.counts squeezed to 1-D) is subtracted
            before summation.

        Example
        -------
        # Return summed counts over detectors 'n0' and 'n1' between 10 and 20 seconds in 50-300 keV, with background subtraction:
        time_centroids, time_widths, counts = obj.get_lightcurve_counts(detectors=['n0', 'n1'],
                                                                        time_range=(10.0, 20.0),
                                                                        energy_range=(50.0, 300.0),
                                                                        subtract_bkgd=True)
        """

        # Set the detectors to use
        detectors = detectors or self.detectors

        # Ensure detectors is a list-like container
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]

        # Initiate counts accumulator
        counts_summed = None
        counts_list = []

        # Loop through each detector
        for detector in detectors:

            # print("Processing detector:", detector)

            # Get the data
            data = self.data.get_item(detector).data

            # Set the energy range
            if detector.startswith("n"):
                energy_range = energy_range or self.energy_range_nai
            else:
                energy_range = energy_range or self.energy_range_bgo

            # print("Time range: %.2f-%.2f s" % (time_range[0], time_range[1]) if time_range is not None else "Full time range    ")
            # print("Energy range: %d-%d keV" % (energy_range[0], energy_range[1]))

            # Slice the data in time
            if time_range is not None:
                data = data.slice_time(time_range[0], time_range[1])

            # Get the temporal info
            tstart = data.tstart
            tstop = data.tstop
            time_widths = data.time_widths
            time_centroids = data.time_centroids

            # Integrate the data in energy
            if energy_range is not None:
                data = data.integrate_energy(emin=energy_range[0], emax=energy_range[1])
            else:
                data = data.integrate_energy()

            if subtract_bkgd is True:
                bkgd = self.backgrounds.get_item(detector)

                if time_range is not None:
                    bkgd = bkgd.slice_time(tstart=time_range[0], tstop=time_range[1])

                if energy_range is not None:
                    bkgd = bkgd.integrate_energy(
                        emin=energy_range[0], emax=energy_range[1]
                    )
                else:
                    bkgd = bkgd.integrate_energy()

            # Subtract the backgrounds
            if subtract_bkgd is True:
                counts = data.counts - bkgd.counts.squeeze()
            else:
                counts = data.counts

            if counts_summed is None:
                counts_summed = np.array(
                    counts, dtype=float
                )  # allocate once, ensure float
            else:
                np.add(
                    counts_summed, counts, out=counts_summed
                )  # in-place accumulation

        return time_centroids, time_widths, counts_summed

    def calc_eiso(self, redshift, emin_rest=1.0, emax_rest=10000.0):
        """Calculate the isotropic-equivalent energy (E_iso) from a spectral fit and redshift.
        This method computes the k-corrected energy flux over a specified rest-frame
        energy band, converts it to a fluence using the burst duration (self.t90),
        and then converts that fluence to the isotropic-equivalent energy using the
        project's energetics.fluence_to_Eiso utility.

        Parameters
        ----------
        redshift : float
            Cosmological redshift of the source. Required to k-correct the observer-frame
            energy bounds and to compute the luminosity distance used by energetics.fluence_to_Eiso.
        emin_rest : float, optional
            Lower bound of the energy integration in the source rest frame (default: 1.0).
            Units are the same energy units expected by the spectral model (typical: keV).
        emax_rest : float, optional
            Upper bound of the energy integration in the source rest frame (default: 10000.0).
            Units are the same energy units expected by the spectral model (typical: keV).

        Returns
        -------
        float or None
            The isotropic-equivalent energy E_iso in erg, or None if the calculation
            could not be performed (e.g., missing spectral fit or redshift).
        Side effects / Notes

        Examples
        --------
        >>> # Assuming an instance `burst` with fitted spectral model and t90:
        >>> eiso = burst.calc_eiso(redshift=1.23, emin_rest=1.0, emax_rest=10000.0)
        """

        if self.fit_model is None:
            print(
                "Spectral fit results not available. Please perform spectral fitting first."
            )
            return

        if redshift is None:
            print("Redshift not provided. Cannot calculate Eiso without redshift.")
            return

        # Define energy bounds in the observer frames
        Emin_obs = emin_rest / (1 + redshift)
        Emax_obs = emax_rest / (1 + redshift)

        # Get the energy flux in the k-corrected energy range
        energy_flux = self.fit_model.integrate(
            self.specfitter.parameters, (Emin_obs, Emax_obs), energy=True
        )  # erg/s/cm^2

        # Calculate energy fluence from the spectral fit and t90
        fluence = energy_flux * self.t90

        print("\nCalculating Eiso...")
        print("\nUsing k-correction energy range: %.2f - %.2f keV (rest frame)" % (emin_rest, emax_rest))
        print("\nEnergy Flux = %.3e erg/s/cm^2" % energy_flux)
        print("T90 = %.3f sec" % self.t90)
        print("Fluence = %.3e erg/cm^2" % fluence)
        print("Redshift = %.3f" % redshift)

        # Calculate Eiso using the energetics module
        eiso = energetics.fluence_to_Eiso(fluence, redshift)

        # print(f"{'Eiso (erg)':20s}\t{eiso}\n")
        print("\nEiso (erg) = ", eiso, "\n")

        return eiso

    def calc_liso(self, redshift, emin_rest=1.0, emax_rest=10000.0):
        """
        Calculate the isotropic-equivalent luminosity (Liso) for the fitted spectral model.

        This method:
        - Converts the requested rest-frame energy bounds (emin_rest, emax_rest) to the observer frame
            using the provided redshift.
        - Integrates the fitted spectral model over the observer-frame energy range to obtain the
            energy flux (erg/s/cm^2).
        - Converts the energy flux to Liso (erg/s) using energetics.flux_to_Liso.

        Parameters
        ----------
        self
                Instance that must contain a fitted spectral model accessible as self.fit_model and
                spectral-fit parameters accessible as self.specfitter.parameters.
        redshift : float
                Source redshift. Required to convert rest-frame energy bounds to the observer frame.
        emin_rest : float, optional
                Lower bound of the energy integration in the source rest frame (keV). Default is 1.0 keV.
        emax_rest : float, optional
                Upper bound of the energy integration in the source rest frame (keV). Default is 10000.0 keV.

        Returns
        -------
        float or None
                The isotropic-equivalent luminosity Liso in erg/s if calculation succeeds.
                Returns None if no spectral fit results are available or if redshift is not provided.

        Notes
        -----
        - The observer-frame energy bounds are computed as:
            emin_obs = emin_rest / (1 + redshift)
            emax_obs = emax_rest / (1 + redshift)
        - The energy flux is obtained by calling:
                self.fit_model.integrate(self.specfitter.parameters, (emin_obs, emax_obs), energy=True)
            which is expected to return a flux in erg/s/cm^2.
        - The conversion from flux to luminosity uses energetics.flux_to_Liso(energy_flux, redshift).
        - This method prints status messages and diagnostic values (energy flux, redshift, Liso) to stdout.

        Example
        -------
        >>> # assuming `instance` has a completed spectral fit and a known redshift
        >>> liso = instance.calc_liso(redshift=0.5)
        >>> # liso is the returned isotropic-equivalent luminosity in erg/s
        """

        if self.fit_model is None:
            print(
                "Spectral fit results not available. Please perform spectral fitting first."
            )
            return

        if redshift is None:
            print("Redshift not provided. Cannot calculate Eiso without redshift.")
            return

        # Define energy bounds in the observer frames
        emin_obs = emin_rest / (1 + redshift)
        emax_obs = emax_rest / (1 + redshift)

        # Get the energy flux in the k-corrected energy range
        energy_flux = self.fit_model.integrate(
            self.specfitter.parameters, (emin_obs, emax_obs), energy=True
        )  # erg/s/cm^2

        print("\nCalculating Liso...")
        print("\nEnergy Flux = %.3e erg/s/cm^2" % energy_flux)
        print("Redshift = %.3f" % redshift)

        # Calculate Eiso using the energetics module
        liso = energetics.flux_to_Liso(energy_flux, redshift)

        print("\nLiso (erg/s): ", liso, "\n")

        return liso

    def localize_source(self, detectors=None, src_range=None, bkgd_range=None):
        """
        Localize a gamma-ray source using the Direction of Origin Localization (DoL) algorithm.

        This method performs source localization by analyzing time-tagged event (TTE) data from
        multiple detectors, calculating source and background counts, and applying the legacy DoL
        algorithm to determine the most likely source position in the sky.

        Parameters
        ----------
        detectors : list of str, optional
            List of detector names to use for localization. If None, uses all available detectors
            defined in self.all_detectors.
        src_range : tuple of float, optional
            Time range (start, stop) in seconds for source integration. If None, uses self.src_range.
        bkgd_range : tuple of float, optional
            Time range(s) for background estimation. If None, uses self.bkgd_range[0].

        Returns
        -------
        None
            Results are printed to console and stored in internal state.

        Notes
        -----
        - Loads TTE data for all specified detectors
        - Integrates source counts over the specified source time range
        - Fits polynomial background model and calculates background counts
        - Uses energy range of 50-300 keV for localization
        - Requires spacecraft position and attitude information from poshist file
        - Applies legacy DoL algorithm with scattering option enabled (scat_opt=1)
        - Prints source and background counts for each detector
        - Prints final localization coordinates in degrees

        The method assumes that self.data, self.met, self.all_detectors, and related attributes
        are properly initialized before calling.
        """

        from gdt.missions.fermi.gbm.localization.dol.legacy_dol import legacy_DoL

        src_time = src_range or self.src_range
        bg_times = bkgd_range or self.bkgd_range[0]
        loc_erange = (50.0, 300.0)

        src_counts = []
        src_exposure = []
        bg_counts = []
        bg_exposure = []
        bg_trigdet = []

        self.load_tte(detectors=self.all_detectors)

        print("\nCalculating source counts...")
        for det in self.all_detectors:
            data = self.data.get_item(det)
            bin = data.data.integrate_time(*src_time)
            src_counts.append(bin.counts.astype(np.int32))
            src_exposure.append(bin.exposure)
            print(f" - {det} {src_counts[-1]}")

        avg_src_exposure = np.sum(src_exposure) / np.array(src_exposure).size
        print(" Exposure %.3f sec" % avg_src_exposure)

        print("\nCalculating backgrounds counts...")
        for det in self.all_detectors:
            # phaii = trigdat.to_phaii(det, timescale=64)
            # fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bg_times)

            data = self.data.get_item(det)
            fitter = BackgroundFitter.from_phaii(data, Polynomial, time_ranges=bg_times)
            fitter.fit(order=1)
            bg_rates = fitter.interpolate_bins(data.data.tstart, data.data.tstop)
            bin = bg_rates.integrate_time(*src_time)
            bg_counts.append(bin.counts.astype(np.int32))
            bg_exposure.append(bin.exposure)
            print(f" - {det} {bg_counts[-1]}")
            if det in self.all_detectors:
                bg_trigdet.append(bg_rates.slice_energy(*loc_erange))

        avg_bg_exposure = np.sum(bg_exposure) / np.array(bg_exposure).size
        print(" Exposure %.3f sec" % avg_bg_exposure)


        energies = np.concatenate([data.data.emin, [data.data.emax[-1]]])
        crange = [np.digitize(e, energies, right=True) - 1 for e in loc_erange]

        tcenter = self.met + 0.5 * sum(src_time)

        if self.poshist_file is None:
            self.load_poshist()

        poshist = GbmPosHist.open(self.poshist_file)
        # frame = poshist.at(Time(tcenter, format='fermi'))
        frame = poshist.get_spacecraft_frame()
        frame = frame.at(Time(tcenter, format='fermi'))
        scpos = frame.obsgeoloc.xyz.to_value('km') # spacecraft position in km to Earth center
        quaternion = frame.quaternion.scalar_last # spacecraft rotation


        dol = legacy_DoL()
        loc = dol.eval(crange, np.array(src_counts), np.array(bg_counts), avg_src_exposure, avg_bg_exposure,
               scpos, quaternion, energies, 191.23910266, -38.14466349, int(tcenter),
               scat_opt=1)

        print(np.degrees(loc["best"]["ra"], loc["best"]["dec"]))

        pass