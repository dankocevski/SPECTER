import os
import copy
import logging
import tkinter as tk
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QMenuBar, QMenu, QFrame, QVBoxLayout,
    QFileDialog, QPushButton, QComboBox, QCheckBox, QLabel, QHBoxLayout, QWidget,
    QGridLayout, QToolBar, QSpacerItem, QSizePolicy, QMessageBox, QDialog, QVBoxLayout,
    QTextEdit
)

from PyQt6.QtGui import QAction, QKeySequence, QFont
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.spectra import functions

from .range_selector import RangeSelector
from .dialogs import (TextDisplayDialog, OptionDialog, PlotDialog, TextOptionDialog, 
                     ManualInputDialog, FitOptionsDialog, BayesianBlocksDialog,
                     StructureFunctionDialog, TemporalRebinDialog,
                     CombineByFactorDialog, RebinBySnrDialog, AnalysisRangeDialog, LagAnalysisDialog,
                     EnergeticsDialog
)

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class PhaViewer(QMainWindow):
    """Class for displaying PHA and PHAII files"""

    def __init__(self, grb, detector):

        super().__init__()
        
        # Setting default state parameters
        self.default_binning = None
        self.current_snr = None
        self.source_selection_active = False
        self.snr_selection_active = False
        self.background_selection_active = False
        self.binning_selection_active = False
        self.lookup = None

        # GUI settings
        self.bg_color = '#e1e1e1'
        self.fg_color = 'black'
        self.font_color = 'black'
        self.font = ('Helvetica Neue', 14)

        # Set the data
        self.grb = grb
        self.detector = detector
        self.data = grb.data.get_item(detector)
        self.data_type = self.data.headers['PRIMARY'].get('DATATYPE', '')

        self.canvas = None
        self.canvas_spec = None
        self.toolbar = None
        self.toolbar_spec = None
        self.lightcurve_displayed = True     
        self.selector = None  
        self._prev_xlim = None
        self._prev_ylim = None
        self.show_bblocks = False

        self.xlog_lightcurve = False
        self.ylog_lightcurve = False
        self.xlog_spectrum = True
        self.ylog_spectrum = True
        self.show_grid = True
        self.show_legend = True

        # Create the PyQT interface
        self._build_gui()

    def _build_gui(self):

        ########### Main Window ###########

        self.setWindowTitle(self.grb.name)
        self.setGeometry(100, 100, 1000, 600)

        # Central widget for the main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create the main layout
        main_layout = QHBoxLayout(central_widget)

        ########### Menu Bar ###########

        # Create menu bar
        self.menubar = self.menuBar()

        # File Menu
        file_menu = self.menubar.addMenu("File")

        # Add header submenu
        if self.data is not None:
            header_menu = QMenu("Header...", self)
            for header_name in self.data.headers.keys():
                action = QAction(header_name, self)
                action.triggered.connect(lambda checked, ext=header_name: self.display_header(ext))
                header_menu.addAction(action)
            file_menu.addMenu(header_menu)

        # Add lookup submenu
        lookup_menu = QMenu("Lookup...", self)
        lookup_items = [
            ("Save Lookup", self.command),
            ("Read Lookup", self.command),
            ("File Content", self.command),
            ("Erase Current", self.command),
        ]
        for name, method in lookup_items:
            action = QAction(name, self)
            action.triggered.connect(method)
            lookup_menu.addAction(action)
        file_menu.addMenu(lookup_menu)

        # Add response submenu
        response_menu = QMenu("Response...", self)
        response_items = [
            ("Load Response", self.command),
            ("Remove Response", self.command),
            ("Display Response", self.command),
        ]
        for name, method in response_items:
            action = QAction(name, self)
            action.triggered.connect(method)
            response_menu.addAction(action)
        file_menu.addMenu(response_menu)

        # Export Menu
        export_menu = self.menubar.addMenu("Export")

        # Add export data submenu
        export_data_menu = QMenu("Export Data...", self)
        export_data_items = [
            ("Selection to PHAII", self.command),
            ("Selection to PHA", self.command),
        ]
        for name, method in export_data_items:
            action = QAction(name, self)
            action.triggered.connect(method)
            export_data_menu.addAction(action)
        export_menu.addMenu(export_data_menu)

        # Add export background submenu
        export_bg_menu = QMenu("Export Background...", self)
        export_bg_items = [("Selection to BAK", self.command)]
        for name, method in export_bg_items:
            action = QAction(name, self)
            action.triggered.connect(method)
            export_bg_menu.addAction(action)
        export_menu.addMenu(export_bg_menu)

        ########### Left Side: Buttons & Toggles ###########

        # Left side panel and layout (buttons and controls) with fixed width
        left_panel = QWidget()
        left_panel.setFixedWidth(180)  # Set a fixed width for the left panel
        left_layout = QVBoxLayout(left_panel)

        # Toggle button for Spectra/Light Curve
        self.toggle_button = QPushButton("Show Spectra")
        self.toggle_button.clicked.connect(self.toggle)
        left_layout.addWidget(self.toggle_button)

        # Rebin light curve menu
        self.rebin_menu = QComboBox()
        self.rebin_menu.setPlaceholderText("Data Binning") # Add placeholder text as the first item
        self.rebin_menu.addItem("Data Binning:")  # Placeholder text at the top of the menu
        self.rebin_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.rebin_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder    
        self.rebin_menu.addItems([
            'Full Resolution', 'Temporal Resolution...', 'Combine by Factor...',
            'Combine Into Single Bin', 'Signal to Noise...'
        ])
        self.rebin_menu.currentTextChanged.connect(self.on_rebin_menu_selected)
        left_layout.addWidget(self.rebin_menu)

        # Source selection spectrum menu (hidden by default)
        self.source_selection_menu = QComboBox()
        self.source_selection_menu.setPlaceholderText("Source Selection") # Add placeholder text as the first item
        self.source_selection_menu.addItem("Source Selection:")  # Placeholder text at the top of the menu
        self.source_selection_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.source_selection_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder
        self.source_selection_menu.addItems([
            'Interactive Selection', 'Clear Selections'
        ])
        # Disable the clear selection option if no source range has been specified
        if self.grb.src_range is None:
            self.source_selection_menu.model().item(2).setEnabled(False)

        self.source_selection_menu.currentTextChanged.connect(self.on_src_menu_selected)
        left_layout.addWidget(self.source_selection_menu)

        # Adjust source selection menu
        self.adjust_source_selection_menu = QComboBox()
        self.adjust_source_selection_menu.setPlaceholderText("Source Adjustment") # Add placeholder text as the first item
        self.adjust_source_selection_menu.addItem("Source Adjustment:")  # Placeholder text at the top of the menu
        self.adjust_source_selection_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.adjust_source_selection_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder        
        self.adjust_source_selection_menu.addItems([
            '< Shift Selection', '> Shift Selection', '< Left Selection',
            '> Left Selection', '< Right Selection', '> Right Selection'
        ])
        self.adjust_source_selection_menu.currentTextChanged.connect(self.on_adjust_src_menu_selected)
        left_layout.addWidget(self.adjust_source_selection_menu)

        # Background menu
        self.bkgd_menu = QComboBox()
        self.bkgd_menu.setPlaceholderText("Background Fitting") # Add placeholder text as the first item
        self.bkgd_menu.addItem("Background Fitting:")  # Placeholder text at the top of the menu
        self.bkgd_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.bkgd_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder    
        self.bkgd_menu.addItems([
            'Fit Background', 'Refit Selections', 'Clear Model'
        ])

        # Disable 'Refit Selections' (index 2) and 'Clear Model' (index 3)
        self.bkgd_menu.model().item(2).setEnabled(False)
        self.bkgd_menu.model().item(3).setEnabled(False)

        self.bkgd_menu.currentIndexChanged.connect(self.on_bkgd_menu_selected)
        left_layout.addWidget(self.bkgd_menu)

        # Export Settings menu
        self.spatial_analysis_menu = QComboBox()
        self.spatial_analysis_menu.setPlaceholderText("Spatial Analysis") # Add placeholder text as the first item
        self.spatial_analysis_menu.addItem("Spatial Analysis:")  # Placeholder text at the top of the menu
        self.spatial_analysis_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.spatial_analysis_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder         
        self.spatial_analysis_menu.addItems([
            'Plot Localization', 'Plot Spacecraft Orbit', 'Calculate Detector Angles', 'Calculate Time to SAA'
        ])
        self.spatial_analysis_menu.currentTextChanged.connect(self.on_spatial_analysis_menu_selected)
        left_layout.addWidget(self.spatial_analysis_menu)

        # Export Settings menu
        self.temporal_analysis_menu = QComboBox()
        self.temporal_analysis_menu.setPlaceholderText("Temporal Analysis") # Add placeholder text as the first item
        self.temporal_analysis_menu.addItem("Temporal Analysis:")  # Placeholder text at the top of the menu
        self.temporal_analysis_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.temporal_analysis_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder         
        self.temporal_analysis_menu.addItems([
            'Calculate T90...', 'Bayesian Blocks...', 'Minimum Variability (SF)...', 'Lag Analysis...',
            'Leahy Periodogram...', 'Fast Fourier Transform...', 'Sum Lightcurves', 'Stack Lightcurves'
        ])
        self.temporal_analysis_menu.currentTextChanged.connect(self.on_temp_analysis_menu_selected)

        # if self.data_type == 'CSPEC' or self.data_type == 'CTIME':
        #     self.temporal_analysis_menu.model().item(2).setEnabled(False)  # Disable Unbinned QPO
        #     self.temporal_analysis_menu.model().item(3).setEnabled(False)  # Disable Binned QPO (Summed)

        left_layout.addWidget(self.temporal_analysis_menu)

        # Spectral fitting menu
        self.spectral_fitting_menu = QComboBox()
        self.spectral_fitting_menu.setPlaceholderText("Spectral Analysis") # Add placeholder text as the first item
        self.spectral_fitting_menu.addItem("Spectral Analysis:")  # Placeholder text at the top of the menu
        self.spectral_fitting_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.spectral_fitting_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder         
        self.spectral_fitting_menu.addItems([
            'Fit Selections...', 'Batch Fit Selections...', 'Line Search...', 'Plot Hardness Ratio...',
            'Calculate Liso', 'Calculate Eiso'
        ])
        self.spectral_fitting_menu.currentTextChanged.connect(self.on_spec_fitting_menu_selected)
        left_layout.addWidget(self.spectral_fitting_menu)

        ########### Check boxes ###########

        # label_plot_options = QLabel("Plot Options:")
        # left_layout.addWidget(label_plot_options)

        hbox_row1 = QHBoxLayout()
        self.xscale_button = QCheckBox("X Log")
        self.xscale_button.setDisabled(True)
        self.xscale_button.stateChanged.connect(self.toggle_x_log_scale)
        self.xscale_button.setChecked(self.xlog_lightcurve)
        hbox_row1.addWidget(self.xscale_button)

        self.yscale_button = QCheckBox("Y Log")
        self.yscale_button.stateChanged.connect(self.toggle_y_log_scale)
        self.yscale_button.setChecked(self.ylog_lightcurve)
        hbox_row1.addWidget(self.yscale_button)

        left_layout.addLayout(hbox_row1)

        hbox_row2 = QHBoxLayout()
        self.grid_button = QCheckBox("Grid")
        self.grid_button.setDisabled(False)
        self.grid_button.stateChanged.connect(self.toggle_grid)
        self.grid_button.setChecked(self.show_grid)
        hbox_row2.addWidget(self.grid_button)

        self.legend_button = QCheckBox("Legend")
        self.legend_button.stateChanged.connect(self.toggle_legend)
        self.legend_button.setChecked(self.show_legend)
        hbox_row2.addWidget(self.legend_button)

        left_layout.addLayout(hbox_row2)

        ########### Labels ###########

        # Add some padding
        left_layout.addSpacing(20)

        # Extract the header information
        header = self.data.headers['PRIMARY']
        try:
            self.eventLabel = QLabel(f"Event: {header['OBJECT']}")
        except:
            self.eventLabel = QLabel(f"Event: {self.grb.name}")

        self.telescopeLabel = QLabel(f"Telescope: {header['TELESCOP']}")
        self.instrumentLabel = QLabel(f"Instrument: {header['INSTRUME']}")
        self.detectorLabel = QLabel(f"Detector: {self.data.detector}")
        self.dataTypes = QLabel(f"Data type: {header.get('DATATYPE', '')}")

        self.selectionLabel = QLabel("Selections:")
        self.time_selection_text = " -- : -- s"
        self.time_selection = QLabel(self.time_selection_text)        
        self.energy_selection_text = " -- - -- keV"
        self.energy_selection = QLabel(self.energy_selection_text)        

        left_layout.addWidget(self.eventLabel)
        left_layout.addWidget(self.telescopeLabel)
        left_layout.addWidget(self.instrumentLabel)
        left_layout.addWidget(self.detectorLabel)
        left_layout.addWidget(self.dataTypes)

        left_layout.addSpacing(20)
        left_layout.addWidget(self.selectionLabel)
        left_layout.addWidget(self.time_selection)
        left_layout.addWidget(self.energy_selection)

        # Add stretch to push metadata to the top
        left_layout.addStretch()

        # Add the left panel to the main layout
        main_layout.addWidget(left_panel)

        self.left_panel = left_panel

        self.set_time_selection_label()
        self.set_energy_selection_label()

        ########### Plot Window ###########

        # Create the right widget
        right_widget = QWidget(self)

        # Create the right layout
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)   

        right_widget.setLayout(right_layout)
        right_widget.setMinimumSize(1000, 600)

        # Create a plotting canvas
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        self.canvas_spec = MplCanvas(self, width=10, height=6, dpi=100)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setStyleSheet("QToolBar { border: 0; background: none; }")

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar_spec = NavigationToolbar(self.canvas_spec, self)
        self.toolbar_spec.setMovable(False)
        self.toolbar_spec.setFloatable(False)
        self.toolbar_spec.setStyleSheet("QToolBar { border: 0; background: none; }")

        # Add the canvas and the toolbar to the widget
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.canvas_spec)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.toolbar_spec)

        # Add the plotting widget to the main layout
        main_layout.addWidget(right_widget, 3)

        # Create the light curve plot
        self.lightcurve = self.grb.plot_lightcurves(data=self.data, ax=self.canvas.axes)
        
        # Create the spectrum plot
        self.spectrum = self.grb.plot_spectra(data=self.data, detectors=[self.detector], ax=self.canvas_spec.axes)

        # Show grid lines by default
        if self.show_grid == True:
            self.canvas.axes.grid(True, alpha=0.25)
            self.canvas_spec.axes.grid(True, alpha=0.25)
        
        # Hide the spectrum plots
        self.canvas_spec.hide()
        self.toolbar_spec.hide()

        # Connect view range change callback and keep the connection id so we can disconnect later
        try:
            self._view_range_cid = self.canvas.fig.canvas.mpl_connect('draw_event', self._on_view_range_changed)
        except Exception:
            self._view_range_cid = None

        # Record the view range if not already set
        if self.grb.view_range is None:
            x0, x1 = self.canvas.axes.get_xlim()
            self.grb.view_range = (float(x0), float(x1))
 

        return

    def toggle(self):

        if self.lightcurve_displayed == True:            
            self.toggle_button.setText("Show Lightcurve")
            self.canvas.hide()
            self.toolbar.hide()
            self.canvas_spec.show()
            self.toolbar_spec.show()
            self.lightcurve_displayed = False

            self.bkgd_menu.setEnabled(False)
            self.rebin_menu.model().item(2).setEnabled(False)
            self.temporal_analysis_menu.setEnabled(False)
            self.source_selection_menu.model().item(2).setEnabled(False)

            self.xscale_button.setDisabled(False)
            self.xscale_button.setChecked(self.xlog_spectrum)
            self.yscale_button.setChecked(self.ylog_spectrum)

        else:
            self.toggle_button.setText("Show Spectra")
            self.canvas_spec.hide()
            self.toolbar_spec.hide()
            self.canvas.show()
            self.toolbar.show()
            self.lightcurve_displayed = True

            self.bkgd_menu.setEnabled(True)
            self.rebin_menu.model().item(2).setEnabled(True)
            self.temporal_analysis_menu.setEnabled(True)

            self.xscale_button.setDisabled(True)
            self.xscale_button.setChecked(self.xlog_lightcurve)
            self.yscale_button.setChecked(self.ylog_lightcurve)

            if self.grb.src_range is not None:
                self.source_selection_menu.model().item(2).setEnabled(True)

    def command(self):
        print("I'd buy that for a dollar")
        pass

    def toggle_x_log_scale(self, checked: bool):

        if self.lightcurve_displayed:
            self.xlog_lightcurve = bool(checked)
        else:
            self.xlog_spectrum = bool(checked)

        if self.canvas is not None and self.canvas_spec is not None:
            self.redraw_plots()

    def toggle_y_log_scale(self, checked: bool):

        if self.lightcurve_displayed:
            self.ylog_lightcurve = bool(checked)
        else:
            self.ylog_spectrum = bool(checked)

        if self.canvas is not None and self.canvas_spec is not None:
            self.redraw_plots()

    def toggle_grid(self, checked: bool):

        self.show_grid = bool(checked)

        if self.canvas is not None and self.canvas_spec is not None:
            self.redraw_plots()

    def toggle_legend(self, checked: bool):

        self.show_legend = bool(checked)

        if self.canvas is not None and self.canvas_spec is not None:
            self.redraw_plots()

    def on_bkgd_ranges_selected(self, selected_bkgd_ranges):
        print("Ranges selected:", selected_bkgd_ranges)
        self.selected_bkgd_ranges = selected_bkgd_ranges
        self.grb.bkgd_range = [selected_bkgd_ranges]

        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None

        self.lightcurve = self.grb.plot_lightcurves(data=self.data, ax=self.canvas.axes)
    
        # self.fit_selections()
        self.background_fit_options()
        
    def set_time_selection_label(self):
        """Set the time selection label"""
        try:
            bounds = (self.lookup.selections.source[0][0],
                      self.lookup.selections.source[-1][-1])
        except:
            bounds = self.grb.src_range

        if bounds is None:
            # self.time_selection_text = "            to               sec"
            self.time_selection_text = " -- to -- sec"
        else:
            self.time_selection_text = " {0:.2f} to {1:.2f} sec".format(*bounds)

        self.time_selection.setText(self.time_selection_text)

    def set_energy_selection_label(self):
        """Set the energy selection label"""
        try:
            bounds = (self.lookup.selections.energy[0][0],
                      self.lookup.selections.energy[-1][-1])
        except:

            if self.detector.startswith('n'):
                bounds = (self.grb.energy_range_nai[0], self.grb.energy_range_nai[1])
            else:
                bounds = (self.grb.energy_range_bgo[0], self.grb.energy_range_bgo[1])
                
        self.energy_selection_text = " {0:.2f} to {1:.2f} keV".format(*bounds)
        self.energy_selection.setText(self.energy_selection_text)

    def _on_view_range_changed(self, event):
        """Callback triggered only when axis limits (pan/zoom) actually change"""
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        
        if self._prev_xlim is None and self.grb.view_range is not None:
            self._prev_xlim = self.grb.view_range
            self._prev_ylim = None
        elif self._prev_xlim is None:
            self._prev_xlim = xlim
            self._prev_ylim = ylim
        elif (self._prev_xlim, self._prev_ylim) != (xlim, ylim):
            self._prev_xlim = xlim
            self._prev_ylim = ylim
            self.grb.view_range = xlim

    def _on_click(self, event):
        print("Mouse clicked!")

    def reset_menu(self, menu):

        # Block the menu from executing any commands
        menu.blockSignals(True)

        # Ensure no item is selected initially
        menu.setCurrentIndex(-1)         
        
        # Disable the placeholder 
        menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  

        # Re-enable the menu 
        menu.blockSignals(False)      

    def on_bkgd_menu_selected(self, index):
        if index <= 0:
            return  # Ignore placeholder

        text = self.bkgd_menu.itemText(index)

        # if text == "Fit Selections":
        if text == 'Fit Background':
            self.selector = RangeSelector(canvas=self.canvas, toolbar=self.toolbar, 
                                        finished_callback=self.on_bkgd_ranges_selected)

        if text == "Refit Selections":
            if self.selected_bkgd_ranges is not None:
                self.on_bkgd_ranges_selected(self.selected_bkgd_ranges)

        elif text == "Clear Model":
            self.grb.backgrounds = None
            self.bkgd_menu.model().item(3).setEnabled(False)  # Clear Model
            self.update_windows()

        self.reset_menu(self.bkgd_menu)

    def background_fit_options(self):
        
        print(self.grb.bkgd_range)

        message = "Select Order of Background Polynomial"
        options = ['Cancel', '0', '1', '2', '3', '4']

        commands = [self.fit_background_selections, 
                    lambda: self.fit_background_selections(0),
                    lambda: self.fit_background_selections(1),
                    lambda: self.fit_background_selections(2),
                    lambda: self.fit_background_selections(3),
                    lambda: self.fit_background_selections(4)]
        
        # Display the dialog
        self.background_options_dialog = OptionDialog(
            options=options,
            commands=commands,
            title="Background",
            message=message,
            parent=self,         
            width=180,
            height=300
        )

        self.background_options_dialog.exec()

    def fit_background_selections(self, order):
        print("fitting backgrounds...")

        # Fit the background
        self.grb.fit_backgrounds(order=order, plot_residuals=True, plot_fit=False)

        # Enable these options now that a fit exists
        self.bkgd_menu.model().item(2).setEnabled(True)  # Refit Selections
        self.bkgd_menu.model().item(3).setEnabled(True)  # Clear Model

        # Update all of the open windows
        self.update_windows()

    def on_src_menu_selected(self, selection):

        # If zoom or pan is active, turn if off. 
        self.reset_toolbars()

        if selection == "Interactive Selection":

            if self.lightcurve_displayed is True:
                self.selector = RangeSelector(canvas=self.canvas, toolbar=self.toolbar, 
                                            finished_callback=self.on_src_ranges_selected,
                                            src_selection=True)
            else:
                self.selector = RangeSelector(canvas=self.canvas_spec, toolbar=self.toolbar_spec, 
                                finished_callback=self.on_energy_ranges_selected,
                                src_selection=True)       

        elif selection == "Clear Selections":

            if self.lightcurve_displayed is True:
                self.grb.src_range = None
                self.source_selection_menu.model().item(2).setEnabled(False)
            else:
                if self.detector.startswith('n'):
                    self.grb.energy_range_nai = None
                else:
                    self.grb.energy_range_bgo = None

            self.grb.update_gui_windows()

        self.reset_menu(self.source_selection_menu)        

    def on_adjust_src_menu_selected(self, selection):


        # If zoom or pan is active, turn if off. 
        self.reset_toolbars()

        index_low = np.abs(self.data.data.tstart - self.grb.src_range[0]).argmin()
        index_high = np.abs(self.data.data.tstart - self.grb.src_range[1]).argmin()

        if selection == '< Shift Selection':
            index_low = index_low - 1
            index_high = index_high - 1
        elif selection == '> Shift Selection':
            index_low = index_low + 1
            index_high = index_high + 1
        elif selection == '< Left Selection':
            index_low = index_low - 1
        elif selection == '> Left Selection':
            index_low = index_low + 1
        elif selection == '< Right Selection':
            index_high = index_high - 1
        elif selection == '> Right Selection':
            index_high = index_high + 1
        else:
            pass

        # Set the new source range
        self.grb.src_range = (self.data.data.tstart[index_low], self.data.data.tstart[index_high])

        # Update the GUI
        self.reset_menu(self.adjust_source_selection_menu)        
        self.update_windows()

    def on_spatial_analysis_menu_selected(self, selection):

        # Reset the menu
        self.reset_menu(self.spatial_analysis_menu)    

        if selection == 'Plot Localization':
            # self.grb.localization_info()
            self.grb.plot_localization()
        if selection == 'Plot Spacecraft Orbit':
            self.grb.plot_orbit()
        if selection == 'Calculate Time to SAA':
            self.grb.calc_time_to_saa()
        if selection == 'Calculate Detector Angles':
            self.grb.calc_detector_angles()

    def on_temp_analysis_menu_selected(self, selection):

        # Reset the menu
        self.reset_menu(self.temporal_analysis_menu)

        # If zoom or pan is active, turn if off. 
        self.reset_toolbars()

        if selection == 'Calculate T90...':

            initial_params = {
                "analysis_range": self.grb.t90_range if self.grb.t90_range is not None else 
                (self.canvas.axes.get_xlim()[0], self.canvas.axes.get_xlim()[1])
            }

            title = 'T90'
            label = 'T90 Analysis Options'
            message = 'Select the temporal range over which to\nperform the counts based T90 calculation.'

            self.t90_options_dialog = AnalysisRangeDialog(title=title, label=label, message=message, parent=self, initial=initial_params)
            result = self.t90_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:

                params = self.t90_options_dialog.params
                if params['sum_detectors'] is True:
                    self.grb.calc_t90(analysis_range=params['analysis_range'])
                else:
                    self.grb.calc_t90(analysis_range=params['analysis_range'], detectors=self.detector)
                
                self.grb.t90_range = params['analysis_range']
            
            return
                        
        elif selection == 'Bayesian Blocks...':

            initial_params = {
                "p0": 0.05,
                "bg_method": "ends",
                "bg_percentage": 0.25,
                "ends_frac": 0.1,
                "sigma_thresh": 5,
                "factor_thresh": 1.5,
                "min_block_duration": 0.064,
                "merge_gap": 0.25,
            }

            self.bblocks_options_dialog = BayesianBlocksDialog(parent=self, initial=initial_params)        
            result = self.bblocks_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                params = self.bblocks_options_dialog.params

                p0 = params['p0']
                min_block_duration = params['min_block_duration']
                show_episodes = params['auto_detect_flares']
                sigma_thresh = params['sigma_thresh']
                merge_gap = params['merge_gap']

                self.grb.bayesian_blocks(detector=self.detector, p0=p0, sigma_thresh=sigma_thresh, 
                                        merge_gap=merge_gap, show_plot=True, show_episodes=show_episodes,
                                        min_block_duration=min_block_duration)
                self.show_bblocks = False
                self.redraw_plots()

                return 
                
            return None

        elif selection == 'Sum Lightcurves':
            self.grb.plot_summed_lightcurve()        
        
        elif selection == 'Stack Lightcurves':
            self.grb.plot_stacked_lightcurves()

        elif selection == 'Unbinned QPO':
            self.command()

        elif selection == 'Leahy Periodogram...':

            initial_params = {
                "analysis_range": self.grb.analysis_range if self.grb.analysis_range is not None else 
                (self.canvas.axes.get_xlim()[0], self.canvas.axes.get_xlim()[1])
            }

            title = 'Leahy Periodogram'
            label = 'Leahy Periodogram Options'
            message = 'Select the temporal range over which to\ngenerate the Leahy periodogram.'

            self.leahy_options_dialog = AnalysisRangeDialog(title=title, label=label, message=message, parent=self, initial=initial_params)
            result = self.leahy_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:

                params = self.leahy_options_dialog.params
                if params['sum_detectors'] is True:
                    self.grb.leahy_scan(analysis_range=params['analysis_range'])
                else:
                    print("Performing Leahy periodogram on detector:", self.detector)
                    self.grb.leahy_scan(analysis_range=params['analysis_range'], detectors=self.detector)

                self.grb.analysis_range = params['analysis_range']
            
            return


        elif selection == 'Fast Fourier Transform...':
            self.command()      

        elif selection == 'Lag Analysis...':

            initial_params = {
                "analysis_range": self.grb.analysis_range if self.grb.analysis_range is not None else 
                (self.canvas.axes.get_xlim()[0], self.canvas.axes.get_xlim()[1])
            }

            title = 'Lag Analysis'
            label = 'Lag Analysis Options'

            self.lag_options_dialog = LagAnalysisDialog(title=title, label=label, parent=self, initial=initial_params)
            result = self.lag_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:

                params = self.lag_options_dialog.params
                if params['sum_detectors'] is True:
                    self.grb.lag_analysis(subtract_bkgd=params['subtract_bkgd'], analysis_range=params['analysis_range'],
                    low_energy=params['low_energy'], high_energy=params['high_energy'])       
                else:
                    self.grb.lag_analysis(subtract_bkgd=params['subtract_bkgd'], analysis_range=params['analysis_range'],
                    low_energy=params['low_energy'], high_energy=params['high_energy'], detectors=self.detector)
                
                self.grb.analysis_range = params['analysis_range']
            
            return

            # analysis_range = (self.canvas.axes.get_xlim()[0], self.canvas.axes.get_xlim()[1])

        elif selection == 'Minimum Variability (SF)...':

            initial_params = {
                "max_lag": 25.0,
                "mc_trials": 500,
                "mc_quantiles": (0.05, 0.99),
                "mvt_rule": "mc",
                # "mvt_rule": "analytic",
                # "analytic_nsigma": 10.0,
                # "min_pairs_per_lag": 50,
            }

            self.bblocks_options_dialog = StructureFunctionDialog(parent=self, initial=initial_params)        
            result = self.bblocks_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                params = self.bblocks_options_dialog.params

                max_lag = params['max_lag']
                mc_trials = params['mc_trials']
                mc_quantiles = params['mc_quantiles']
                mvt_rule = params['mvt_rule']

                self.grb.calc_structure_function(detector=self.detector, max_lag=max_lag, mc_trials=mc_trials,
                                            mc_quantiles=mc_quantiles, mvt_rule=mvt_rule)

                return 
                
            return None
          
    def on_rebin_menu_selected(self, selection):

        self.reset_menu(self.rebin_menu)

        if selection == 'Full Resolution':
            self.grb.reset_binning()
            self.reset_toolbars()

        elif selection == 'Temporal Resolution...':
         
            initial_params = {
                "resolution": self.data.data.time_widths[0],
            }

            self.rebin_options_dialog = TemporalRebinDialog(parent=self, initial=initial_params)        
            result = self.rebin_options_dialog.exec()
            if result == QDialog.DialogCode.Accepted:
                params = self.rebin_options_dialog.params

                self.grb.rebin_lightcurve(params['resolution'], method='rebin_by_time')

        elif selection == 'Combine by Factor...':

            initial_params = {
                "factor": 1,
            }

            self.rebin_options_dialog = CombineByFactorDialog(parent=self, initial=initial_params)        
            
            result = self.rebin_options_dialog.exec()
            if result == QDialog.DialogCode.Accepted:
                params = self.rebin_options_dialog.params

                if self.lightcurve_displayed:
                    self.grb.rebin_lightcurve(params['factor'], method='combine_by_factor')
                else:
                    self.grb.rebin_spectra(params['factor'], method='combine_by_factor')

        elif selection == 'Signal to Noise...':

            initial_params = {
                "snr": 5.0,
            }

            self.rebin_options_dialog = RebinBySnrDialog(parent=self, initial=initial_params)        
            result = self.rebin_options_dialog.exec()
            if result == QDialog.DialogCode.Accepted:
                params = self.rebin_options_dialog.params

                if self.lightcurve_displayed:
                    self.grb.rebin_lightcurve(params['snr'], method='rebin_by_snr')
                else:
                    self.grb.rebin_spectra(params['snr'], method='rebin_by_snr')

        elif selection == 'Combine Into Single Bin':

            if self.lightcurve_displayed:
                self.grb.rebin_lightcurve(None, method='combine_into_one')
            else:
                self.grb.rebin_spectra(None, method='combine_into_one')

    def rebin(self, value):
        print("Rebinning data...")
        print(f"New binning value: {value} seconds")

        if self.lightcurve_displayed:
            self.grb.rebin_lightcurve(value)
        else:
            self.grb.rebin_spectra(value)

        # self.update_windows()
        print("Done.")

    def on_src_ranges_selected(self, selected_src_ranges):
        print("Ranges selected:", selected_src_ranges)
        self.selected_src_ranges = selected_src_ranges
        self.grb.src_range = selected_src_ranges[0]

        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None

        # Re-enable the clear button
        self.source_selection_menu.model().item(2).setEnabled(True)

        # Redraw the light curve
        self.update_windows()

    def on_energy_ranges_selected(self, selected_energy_ranges):
        print("Ranges selected:", selected_energy_ranges[0])
        if self.detector.startswith('n'):
            self.grb.energy_range_nai = selected_energy_ranges[0]
        else:
            self.grb.energy_range_bgo = selected_energy_ranges[0]

        # Redraw the light curve
        self.update_windows(reset_ylim=True)

    def redraw_plots(self):

        # Create the lightcure and spectrum canveses
        self.canvas.axes.clear()
        self.canvas_spec.axes.clear()

        # Lightcuve plot
        self.lightcurve = self.grb.plot_lightcurves(data=self.data, ax=self.canvas.axes, show_bblocks=self.show_bblocks)
        self.canvas.axes.set_xlim(self._prev_xlim)
        self.canvas.axes.set_ylim(self._prev_ylim)

        # Spectrum plot
        self.spectrum = self.grb.plot_spectra(data=self.data, detectors=[self.detector], ax=self.canvas_spec.axes)

        # Set the lightcurve xscale 
        if self.ylog_lightcurve == True:
            self.canvas.axes.set_yscale('log')
        else:
            self.canvas.axes.set_yscale('linear')        

        # Set the spectrum xscale 
        if self.xlog_spectrum == True:
            self.canvas_spec.axes.set_xscale('log')
        else:
            self.canvas_spec.axes.set_xscale('linear')    

        # Set the spectrum xscale 
        if self.ylog_spectrum == True:
            self.canvas_spec.axes.set_yscale('log')
        else:
            self.canvas_spec.axes.set_yscale('linear')    

        if self.show_grid == True:
            self.canvas.axes.grid(True, alpha=0.25)
            self.canvas_spec.axes.grid(True, alpha=0.25)

        if not self.show_legend:
            # Hide legends if present
            leg = self.canvas.axes.get_legend()
            if leg is not None:
                leg.set_visible(False)
                leg_spec = self.canvas_spec.axes.get_legend()
                if leg_spec is not None:
                    leg_spec.set_visible(False)

            # Hide text annotations (preserve axis labels/titles/ticks)
            for txt in list(self.canvas.axes.texts):
                txt.set_visible(False)
            for txt in list(self.canvas_spec.axes.texts):
                txt.set_visible(False)



        self.canvas.draw() 
        self.canvas_spec.draw() 

    def closeEvent(self, event):
        """Cleanup when the window is closed so the PhaViewer can be garbage-collected.

        - disconnects matplotlib callbacks
        - disconnects any active RangeSelector
        - closes and deletes canvases
        - removes self from the owning GRB's `_pha_viewers` list
        - clears references to large objects
        """
        # Disconnect the draw_event callback
        try:
            if getattr(self, "_view_range_cid", None) is not None and self.canvas is not None:
                try:
                    self.canvas.fig.canvas.mpl_disconnect(self._view_range_cid)
                except Exception:
                    pass
                self._view_range_cid = None
        except Exception:
            pass

        # Disconnect selector if present
        try:
            if getattr(self, "selector", None) is not None:
                try:
                    self.selector.disconnect()
                except Exception:
                    pass
                self.selector = None
        except Exception:
            pass

        # Close and delete canvases to break Qt/C++ refs
        try:
            if getattr(self, 'canvas', None) is not None:
                try:
                    self.canvas.close()
                except Exception:
                    pass
                try:
                    self.canvas.setParent(None)
                except Exception:
                    pass
                try:
                    self.canvas.deleteLater()
                except Exception:
                    pass
                self.canvas = None
        except Exception:
            pass

        try:
            if getattr(self, 'canvas_spec', None) is not None:
                try:
                    self.canvas_spec.close()
                except Exception:
                    pass
                try:
                    self.canvas_spec.setParent(None)
                except Exception:
                    pass
                try:
                    self.canvas_spec.deleteLater()
                except Exception:
                    pass
                self.canvas_spec = None
        except Exception:
            pass

        # Remove self from parent GRB viewer list
        try:
            if getattr(self, 'grb', None) is not None and getattr(self.grb, '_pha_viewers', None) is not None:
                try:
                    if self in self.grb._pha_viewers:
                        self.grb._pha_viewers.remove(self)
                except Exception:
                    pass
        except Exception:
            pass

        # Clear other large references
        try:
            self.lightcurve = None
            self.spectrum = None
            self.toolbar = None
            self.toolbar_spec = None
            self._prev_xlim = None
            self._prev_ylim = None
        except Exception:
            pass

        # Finally, call the base implementation to actually close the window
        try:
            super().closeEvent(event)
        except Exception:
            # If super fails (unlikely), accept the event anyway
            event.accept()

        return
    
    def update_windows(self, reset_ylim=False):
        self.grb.update_gui_windows(reset_ylim=reset_ylim)

    def update_window(self):
        self.redraw_plots()
        self.set_time_selection_label()
        self.set_energy_selection_label()

    def reset_toolbars(self):

        # If zoom is active, turn it off
        if getattr(self.toolbar, "mode", "") == "zoom rect":
            # zoom(False) explicitly deactivates zoom in NavigationToolbar2
            self.toolbar.zoom(False)

        # If pan is active, turn it off too 
        if getattr(self.toolbar, "mode", "") == "pan/zoom":
            self.toolbar.pan(False)

        # If zoom is active, turn it off
        if getattr(self.toolbar_spec, "mode", "") == "zoom rect":
            # zoom(False) explicitly deactivates zoom in NavigationToolbar2
            self.toolbar_spec.zoom(False)

        # If pan is active, turn it off too 
        if getattr(self.toolbar_spec, "mode", "") == "pan/zoom":
            self.toolbar_spec.pan(False)

        # Force the toolbar buttons to visually update
        self.toolbar._update_buttons_checked()
        self.toolbar_spec._update_buttons_checked()

    def display_header(self, ext):

        headers = self.data.headers
        header = headers[ext].tostring(sep='\n', padding=False)
        title = '{0} header for {1}'.format(ext, self.detector)
        font = QFont("Andale Mono", 11)
        height = len(headers[ext])
        if (height*16.0) > 608:
            height=38
        # self.text_display(header, title, font=font, 
        #                  height=height, bg="#e1e1e1")
        self.text_display(header, title, font=font, 
                         height=height)

    def text_display(self, thetext, title=None, font=None, height=None, bg=None):
        """
        Show a non-modal window with scrollable text.

        Args:
            thetext (str)
            title (str or None)
            font (QFont or None)
            height (int or None): number of text rows to display
            bg (str or None): background color (e.g., "#e1e1e1")
        """

        dialog = QDialog(self)
        dialog.setWindowTitle(title or "")

        if bg:
            dialog.setStyleSheet(f"background-color: {bg};")

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit(dialog)
        text_edit.setPlainText(thetext)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        text_edit.setMinimumSize(600, 800)

        # Apply background color and font to the text widget
        if bg:
            text_edit.setStyleSheet(f"background-color: {bg};")
        if font:
            text_edit.setFont(font)

        # If height in text rows is given, convert to pixels
        if height is not None:
            # Default to the text_edit's font if custom one not provided
            f = font if font else text_edit.font()
            metrics = text_edit.fontMetrics()
            row_height = metrics.lineSpacing()
            pixel_height = height * row_height + 20  # padding for scrollbars/margins
            text_edit.setMinimumHeight(pixel_height)

        layout.addWidget(text_edit)

        # Prevent garbage collection
        if not hasattr(self, "_text_windows"):
            self._text_windows = []
        self._text_windows.append(dialog)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def on_spec_fitting_menu_selected(self, selection):

        # If zoom or pan is active, turn if off. 
        self.reset_toolbars()

        # Reset the menu        
        self.reset_menu(self.spectral_fitting_menu)

        if selection == 'Fit Selections...' or selection == 'Batch Fit Selections...':

            if selection == 'Batch Fit Selections...':
                batch_fit = True
            else:
                batch_fit = False

            # Available spectral models
            models = functions.__all__

            self.fit_options_dialog = FitOptionsDialog(models, batch_fit=batch_fit, parent=self)

            self.fit_options_dialog.accepted.connect(lambda: self.on_dialog_finished( self.fit_options_dialog))
            self.fit_options_dialog.rejected.connect(lambda: self.on_dialog_cancelled( self.fit_options_dialog))

            # result = self.fit_options_dialog.exec()
            self.fit_options_dialog.show()
            self.fit_options_dialog.raise_()
            self.fit_options_dialog.activateWindow()

        elif selection == 'Line Search...':
            self.grb.line_search()

        elif selection == 'Plot Hardness Ratio...':
            self.hr_plot = self.grb.calc_hardness_ratio(detectors=self.detector, subtract_bkgd=False, sum_counts=False)  

        elif selection == 'Calculate Liso':

            initial_params = {
                "redshift": self.grb.redshift or 1.0,
                "energy_range": (1.0, 10000.0),
            }

            title = 'Liso Analysis'
            label = 'Liso Calculation Options'
            message = 'Provide the redshift and rest-frame energy\nbandpass for the Liso calculation.\n\nNote: Liso calculations require a spectral\nfit to have already been performed.'

            self.liso_options_dialog = EnergeticsDialog(title=title, label=label, message=message, parent=self, initial=initial_params)
            result = self.liso_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                params = self.liso_options_dialog.params
                self.grb.calc_liso(params['redshift'], emin_rest=params['energy_range'][0], emax_rest=params['energy_range'][1])
            return 

            self.grb.calc_liso(1)

        elif selection == 'Calculate Eiso':

            initial_params = {
                "redshift": self.grb.redshift or 1.0,
                "energy_range": (1.0, 10000.0),
            }

            title = 'Eiso Analysis'
            label = 'Eiso Calculation Options'
            message = 'Provide the redshift and rest-frame energy\nbandpass for the Eiso calculation.\n\nNote: Eiso calculations require a T90 and spectral\nfit to have already been performed.'

            self.eiso_options_dialog = EnergeticsDialog(title=title, label=label, message=message, parent=self, initial=initial_params)
            result = self.eiso_options_dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                params = self.eiso_options_dialog.params
                self.grb.calc_eiso(params['redshift'], emin_rest=params['energy_range'][0], emax_rest=params['energy_range'][1])

            return 

        
    def on_dialog_finished(self, dialog):

        # Get the user selected parameters
        models, statistic, weights, default_values, free = dialog.fit_options
        batch_fit = dialog.batch_fit

        if batch_fit is False:

            # Perform the spectral fit
            self.grb.fit_spectra(models=models, stat=statistic, default_values=default_values, free=free, 
                                create_gui=True)

        else:

            # Perform a batch fit
            self.grb.batch_fit_spectra(models=models, stat=statistic, default_values=default_values, free=free, use_previous_fit=True)

    def on_dialog_cancelled(self, dialog):
        print("Spectral fit cancelled.")
        return

