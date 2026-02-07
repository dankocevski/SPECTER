#! /usr/bin/env python3

import copy
import os
import random

import numpy
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QCheckBox, QRadioButton, QLabel,
    QFileDialog, QButtonGroup, QSizePolicy
)

from gdt.core.spectra import functions

from .dialogs import (FitOptionsDialog)
from .dialogs import PlotModelParametersDialog
from .qt_model_fit import QtModelFit  # your new subclass

# Use the classic matplotlib style sheet
# if float(matplotlib.__version__[0]) >= 2.0:
#     import matplotlib.style
#     matplotlib.style.use('classic')

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class FitPlotter(QMainWindow):
    """Class for displaying spectral fits."""

    name = 'Spectral Fit Display'

    def __init__(self, grb):
        super().__init__()

        self.setWindowTitle(self.name)

        # Set the data
        self.grb = grb

        # Default state variables (mirroring the tk version)
        self.show_count_display = True
        self.count_data_visible = True
        self.show_model_display = True
        self.show_spectral_model_components = True
        self.model_display_visible = True
        self.show_legend_display = True
        self.residual_selection = None
        self.spectrum_selection = 'Counts Spectrum'
        self.xlog = False
        self.ylog = False
        self.spectrum_view = 'counts'
        self.show_residuals = True
        self._plot_ready = False
        self.model_plot = None

        self.data_containers = []
        self.model_artists = []
        self.legend = None

        self.original_plot_ax_pos = None
        self.original_plot_resid_ax_pos = None

        self.show_grid = True
        self.show_legend = True

        self.build_gui()

    def build_gui(self):

        ########### Main Window ###########

        self.setWindowTitle(self.grb.name)
        self.setGeometry(100, 100, 1024, 768)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create the main layout
        main_layout = QHBoxLayout(central_widget)
        # main_layout.setContentsMargins(0, 0, 0, 0)
        # main_layout.setSpacing(0)

        ########### Menu Bar ###########

        # Create the menu bar
        self.menubar = self.menuBar()

        # ----- File menu example (if you need it) -----
        file_menu = self.menubar.addMenu("File")
        dismiss_action = file_menu.addAction("Dismiss")
        dismiss_action.triggered.connect(self.on_window_close)

        # ----- Options menu -----
        items = ['Colors', 'Plot Configuration', 'Set Fit Repeat Count',
                'Set Fluence Energies']
        commands = [self.command, self.command, self.command, self.command]

        # Create the Options menu
        self.options_menu = self.menubar.addMenu("Options")

        # Keep references to actions if you want to enable/disable them later
        self.options_actions = []

        for label, func in zip(items, commands):
            action = self.options_menu.addAction(label)
            action.triggered.connect(func)
            self.options_actions.append(action)

        # Disable all 4 items (equivalent to entryconfig(..., state='disable'))
        self.options_actions[0].setEnabled(False)
        self.options_actions[1].setEnabled(False)
        self.options_actions[2].setEnabled(False)
        self.options_actions[3].setEnabled(False)

        ########### Left Side: Buttons & Toggles ###########

       # ---- Top control frame (combos / buttons) ----
        left_panel = QWidget()
        left_panel.setFixedWidth(200)  # Set a fixed width for the left panel
        left_layout = QVBoxLayout(left_panel)

        # Fit display options combo
        self.fit_display_combo = QComboBox()
        self.fit_display_combo.addItems([
            'Cumulative', 'Raw Counts', 'Counts Spectrum',
            'Photon Spectrum', 'Energy Spectrum', 'Nu Fnu Spectrum'
        ])
        self.fit_display_combo.setCurrentText('Counts Spectrum')
        self.fit_display_combo.currentTextChanged.connect(self.on_fit_display_selection)
        left_layout.addWidget(self.fit_display_combo)

        # Batch fit options combo
        self.fit_diagnostics_options_menu = QComboBox()
        self.fit_diagnostics_options_menu.setPlaceholderText('Fit Diagnostics') 
        self.fit_diagnostics_options_menu.addItem("Fit Diagnostics:")  # Placeholder text at the top of the menu
        self.fit_diagnostics_options_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.fit_diagnostics_options_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder
        self.fit_diagnostics_options_menu.addItems(['ChiSqr 1D Plot', 'ChiSqr 2D Plot',
            'Residual Contours', 'NuFnu Contours', 'Stack Spectra'
        ])
        self.fit_diagnostics_options_menu.currentTextChanged.connect(
            self.on_fit_diag_options_selection
        )

        # Spectral fitting combo
        self.spectral_fitting_menu = QComboBox()
        self.spectral_fitting_menu.setPlaceholderText("Spectral Analysis") 
        self.spectral_fitting_menu.addItem("Spectral Analysis:")  # Placeholder text at the top of the menu
        self.spectral_fitting_menu.setCurrentIndex(-1)         # Ensure no item is selected initially
        self.spectral_fitting_menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable the placeholder
        self.spectral_fitting_menu.addItems([
            'Redo Last Fit...', 'Batch Fit Selections...',
            'Synthesize Burst...', 'Error Interval',
        ])
        self.spectral_fitting_menu.currentTextChanged.connect(
            self.on_spec_fitting_menu_selection
        )
        left_layout.addWidget(self.spectral_fitting_menu)
        left_layout.addWidget(self.fit_diagnostics_options_menu)

        # Fit results menu (export)
        self.fit_results_button = QPushButton('Write Results to File')
        self.fit_results_button.clicked.connect(self.command)
        left_layout.addWidget(self.fit_results_button)

        # left_controls_layout.addWidget(left_panel)
        # left_layout.addWidget(left_panel)

        ########### Check boxes ###########

        # Plot Options
        label_residual_display = QLabel("Plot Options:")
        left_layout.addWidget(label_residual_display)

        hbox_row1 = QHBoxLayout()
        self.cb_x_log = QCheckBox("X Log")
        self.cb_y_log = QCheckBox("Y Log")
        self.cb_x_log.toggled.connect(self.set_x_log_scale)
        self.cb_y_log.toggled.connect(self.set_y_log_scale)
        hbox_row1.addWidget(self.cb_x_log)

        hbox_row1.addWidget(self.cb_y_log)  
        left_layout.addLayout(hbox_row1)

        hbox_row2 = QHBoxLayout()
        self.cb_grid = QCheckBox("Grid")
        self.cb_legend = QCheckBox("Legend")
        self.cb_grid.toggled.connect(self.toggle_grid)
        self.cb_legend.toggled.connect(self.toggle_legend)
        hbox_row2.addWidget(self.cb_grid)
        hbox_row2.addWidget(self.cb_legend)
        left_layout.addLayout(hbox_row2)

        self.cb_x_log.setChecked(True)
        self.cb_y_log.setChecked(True)
        self.cb_grid.setChecked(True)
        self.cb_legend.setChecked(True)
        self.xlog = True
        self.ylog = True
        self.show_grid = True
        self.show_legend = True
        
        # Spectral Data Options
        label_count_display = QLabel("Spectral Data Options:")
        left_layout.addWidget(label_count_display)

        hbox_row1 = QHBoxLayout()
        self.rb_show_counts = QRadioButton("Show")
        self.rb_show_counts.setChecked(True)
        self.rb_show_counts.setEnabled(False)
        self.rb_show_counts.toggled.connect(self.on_count_display_selection)
        self.rb_hide_counts = QRadioButton("Hide")
        self.rb_hide_counts.setEnabled(False)

        hbox_row1.addWidget(self.rb_show_counts)
        hbox_row1.addWidget(self.rb_hide_counts)
        left_layout.addLayout(hbox_row1)

        # Spectral Model Options
        label_count_display = QLabel("Spectral Model Options:")
        left_layout.addWidget(label_count_display)

        hbox_row2 = QHBoxLayout()
        self.rb_show_model = QRadioButton("Show")
        self.rb_show_model.setChecked(True)
        self.rb_show_model.toggled.connect(self.on_model_display_selection)
        self.rb_hide_model = QRadioButton("Hide")

        hbox_row2.addWidget(self.rb_show_model)
        hbox_row2.addWidget(self.rb_hide_model)
        left_layout.addLayout(hbox_row2)

        # Residual Display Options
        label_residual_display = QLabel("Residual Display Options:")
        left_layout.addWidget(label_residual_display)

        self.residual_group = QButtonGroup(self)
        self.rb_sigma_residuals = QRadioButton("Sigma Residuals")
        self.rb_count_residuals = QRadioButton("Count Residuals")
        self.rb_no_residuals = QRadioButton("No Residuals")

        self.residual_group.addButton(self.rb_sigma_residuals)
        self.residual_group.addButton(self.rb_count_residuals)
        self.residual_group.addButton(self.rb_no_residuals)

        self.rb_sigma_residuals.setChecked(True)
        self.residual_group.buttonToggled.connect(self.on_residual_selection_group)

        left_layout.addWidget(self.rb_sigma_residuals)
        left_layout.addWidget(self.rb_count_residuals)
        left_layout.addWidget(self.rb_no_residuals)



        left_layout.addStretch()


        # # # ---- Toggle options frame ----
        # toggles_widget = QWidget()
        # toggles_layout = QGridLayout(toggles_widget)
        # toggles_layout.setContentsMargins(5, 5, 5, 5)
        # toggles_layout.setHorizontalSpacing(0)
        # toggles_layout.setVerticalSpacing(4)
        # toggles_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # row = 0

        # # Spectral Data Options
        # label_count_display = QLabel("Spectral Data Options:")
        # toggles_layout.addWidget(label_count_display, row, 0, 1, 2)
        # row += 1

        # self.count_display_group = QButtonGroup(self)
        # self.rb_show_counts = QRadioButton("Show")
        # self.rb_hide_counts = QRadioButton("Hide")
        # self.count_display_group.addButton(self.rb_show_counts)
        # self.count_display_group.addButton(self.rb_hide_counts)
        # self.rb_show_counts.setChecked(True)
        # self.rb_show_counts.toggled.connect(self.on_count_display_selection)
        # toggles_layout.addWidget(self.rb_show_counts, row, 0)
        # toggles_layout.addWidget(self.rb_hide_counts, row, 1)
        # row += 1
        
        # # Initially disabled
        # self.rb_show_counts.setEnabled(False)
        # self.rb_hide_counts.setEnabled(False)

        # # Spectral Model Options
        # label_model_display = QLabel("Spectral Model Options:")
        # label_model_display.setStyleSheet("margin-top: 10px;")
        # toggles_layout.addWidget(label_model_display, row, 0, 1, 2)
        # row += 1

        # self.rb_show_model = QRadioButton("Show")
        # self.rb_hide_model = QRadioButton("Hide")
        # self.rb_show_model.setChecked(True)
        # self.rb_show_model.toggled.connect(self.on_model_display_selection)
        # toggles_layout.addWidget(self.rb_show_model, row, 0)
        # toggles_layout.addWidget(self.rb_hide_model, row, 1)
        # row += 1

        # # Initially disabled
        # self.rb_show_model.setEnabled(False)
        # self.rb_hide_model.setEnabled(False)

        # # Residual Display Options
        # label_residual_display = QLabel("Residual Display Options:")
        # label_residual_display.setStyleSheet("margin-top: 10px;")
        # toggles_layout.addWidget(label_residual_display, row, 0, 1, 2)
        # row += 1

        # self.residual_group = QButtonGroup(self)
        # self.rb_sigma_residuals = QRadioButton("Sigma Residuals")
        # self.rb_count_residuals = QRadioButton("Count Residuals")
        # self.rb_no_residuals = QRadioButton("No Residuals")

        # self.residual_group.addButton(self.rb_sigma_residuals)
        # self.residual_group.addButton(self.rb_count_residuals)
        # self.residual_group.addButton(self.rb_no_residuals)

        # self.rb_sigma_residuals.setChecked(True)

        # # self.rb_sigma_residuals.toggled.connect(self.on_residual_selection)
        # # self.rb_count_residuals.toggled.connect(self.on_residual_selection)
        # # self.rb_no_residuals.toggled.connect(self.on_residual_selection)

        # self.residual_group.buttonToggled.connect(self.on_residual_selection_group)

        # # Sigma residuals initially disabled
        # # self.rb_sigma_residuals.setEnabled(False)

        # toggles_layout.addWidget(self.rb_sigma_residuals, row, 0, 1, 2)
        # row += 1
        # toggles_layout.addWidget(self.rb_count_residuals, row, 0, 1, 2)
        # row += 1
        # toggles_layout.addWidget(self.rb_no_residuals, row, 0, 1, 2)
        # row += 1

        # # Legend Options
        # label_legend_display = QLabel("Legend Options:")
        # label_legend_display.setStyleSheet("margin-top: 10px;")
        # toggles_layout.addWidget(label_legend_display, row, 0, 1, 2)
        # row += 1

        # self.legend_display_group = QButtonGroup(self)
        # self.rb_show_legend = QRadioButton("Show")
        # self.rb_hide_legend = QRadioButton("Hide")
        # self.legend_display_group.addButton(self.rb_show_legend)
        # self.legend_display_group.addButton(self.rb_hide_legend)
        # self.rb_show_legend.setChecked(True)
        # self.rb_show_legend.toggled.connect(self.on_legend_display_selection)
        # toggles_layout.addWidget(self.rb_show_legend, row, 0)
        # toggles_layout.addWidget(self.rb_hide_legend, row, 1)
        # row += 1

        # # Plot Scale Options
        # label_scale_display = QLabel("Plot Options:")
        # label_scale_display.setStyleSheet("margin-top: 10px;")
        # toggles_layout.addWidget(label_scale_display, row, 0, 1, 2)
        # row += 1

        # hbox_row1 = QHBoxLayout()
        # self.cb_x_log = QCheckBox("X Log")
        # self.cb_y_log = QCheckBox("Y Log")
        # self.cb_x_log.toggled.connect(self.set_x_log_scale)
        # self.cb_y_log.toggled.connect(self.set_y_log_scale)
        # # toggles_layout.addWidget(self.cb_x_log, row, 0)
        # # toggles_layout.addWidget(self.cb_y_log, row, 1)
        # hbox_row1.addWidget(self.cb_x_log)
        # hbox_row1.addWidget(self.cb_y_log)  
        # toggles_layout.addLayout(hbox_row1, row, 0)
        # row += 1

        # hbox_row2 = QHBoxLayout()
        # self.cb_grid = QCheckBox("Grid")
        # self.cb_legend = QCheckBox("Legend")
        # self.cb_grid.toggled.connect(self.toggle_grid)
        # self.cb_legend.toggled.connect(self.toggle_legend)
        # # toggles_layout.addWidget(self.cb_grid, row, 0)
        # # toggles_layout.addWidget(self.cb_legend, row, 1)
        # hbox_row2.addWidget(self.cb_grid)
        # hbox_row2.addWidget(self.cb_legend)
        # toggles_layout.addLayout(hbox_row2, row, 0)
        # row += 1

        # self.cb_x_log.setChecked(True)
        # self.cb_y_log.setChecked(True)
        # self.cb_grid.setChecked(True)
        # self.cb_legend.setChecked(True)
        # self.xlog = True
        # self.ylog = True
        # self.show_grid = True
        # self.show_legend = True

        # # left_controls.addWidget(toggles_widget)
        # left_layout.addWidget(toggles_widget)

        # Add the left panel to the main layout
        main_layout.addWidget(left_panel, 3)

        ########### Right Side: Plot Frame (Canvas + Toolbar) ###########

        # Create the right widget
        right_widget = QWidget(self)

        # Create the right layout
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_widget.setLayout(right_layout)
        right_widget.setMinimumSize(800, 600)

       # Create a plotting canvas
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setStyleSheet("QToolBar { border: 0; background: none; }")

        # Add the canvas and the toolbar to the widget
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.toolbar)

        # Add the plotting widget to the main layout
        main_layout.addWidget(right_widget, 3)

        ########### Initial Plot ###########

        if self.grb.specfitter is not None:
            self.create_plot()

        return

    # ---------------------------------------------------------------------
    # EVENT HANDLERS / LOGIC (renamed to snake_case)
    # ---------------------------------------------------------------------

    def create_plot(self):
        
        self.model_plot = self.grb.plot_spectral_fit(ax=self.canvas.axes, show_residuals=True, view='counts')

        # Save original axis positions for later restoration
        self.original_plot_ax_pos = self.model_plot._ax.get_position()
        self.original_plot_resid_ax_pos = self.model_plot._resid_ax.get_position()

        self._plot_ready = True

        if self.show_grid:
            self.canvas.axes.grid(True, alpha=0.25)

    def toggle_grid(self, checked: bool):
        """Show/Hide grid."""

        self.show_grid = bool(checked)

        if self._plot_ready:
            self.update_plot()

    def toggle_legend(self, checked: bool):
        """Show/Hide legend."""
    
        self.show_legend_display = self.cb_legend.isChecked()
        
        if self._plot_ready:
            legend = self.canvas.axes.get_legend()

            if legend is not None:
                legend.set_visible(self.show_legend_display)
                self.canvas.draw()

    def on_fit_display_selection(self, menu_selection: str):
        """Fit display combo changed."""

        self.spectrum_selection = menu_selection

        if self.spectrum_selection == 'Cumulative':
            self.spectrum_view = 'counts'
            self.command()

        if self.spectrum_selection == 'Raw Counts':
            self.spectrum_view = 'counts'
            self.command()

        if self.spectrum_selection == 'Counts Spectrum':
            self.spectrum_view = 'counts'

        if self.spectrum_selection == 'Photon Spectrum':
            self.spectrum_view = 'photon'

        if self.spectrum_selection == 'Energy Spectrum':
            self.spectrum_view = 'energy'

        if self.spectrum_selection == 'Nu Fnu Spectrum':
            self.spectrum_view = 'nufnu'

        self.update_plot()

    def on_spec_fitting_menu_selection(self, menu_selection):
        """Fit options combo changed."""

        if menu_selection == 'Redo Last Fit...' or menu_selection == 'Batch Fit Selections...':

            if menu_selection == 'Batch Fit Selections...':
                print('Batch fit selected.')
                batch_fit = True
            else:
                batch_fit = False

            # Available spectral models
            models = functions.__all__

            self.fit_options_dialog = FitOptionsDialog(models, batch_fit=batch_fit, parent=self)

            self.fit_options_dialog.accepted.connect(lambda: self.on_spec_fit_dialog_finished( self.fit_options_dialog))
            self.fit_options_dialog.rejected.connect(lambda: self.on_spec_fit_dialog_cancelled( self.fit_options_dialog))

            # result = self.fit_options_dialog.exec()
            self.fit_options_dialog.show()
            self.fit_options_dialog.raise_()
            self.fit_options_dialog.activateWindow()

        if menu_selection == 'Synthesize Burst':
            self.command()

        if menu_selection == 'Error Interval':
            self.command()

        if menu_selection == 'ChiSqr 1D Plot':
            self.command()

        if menu_selection == 'ChiSqr 2D Plot':
            self.command()

        if menu_selection == 'Batch Fit Selections':
            self.command()
            
        self.reset_menu(self.spectral_fitting_menu)

    def on_spec_fit_dialog_finished(self, dialog):

        # Get the user selected parameters
        models, statistic, weights, default_values, free = dialog.fit_options
        batch_fit = dialog.batch_fit

        print("Hello from on_spec_fit_dialog_finished")
        if batch_fit is False:

            # Perform the spectral fit
            self.grb.fit_spectra(models=models, stat=statistic, default_values=default_values, free=free, 
                                create_gui=False)
            
            self.update_plot()

        else:

            print("In batch fit mode")
            # Perform a batch fit
            self.grb.batch_fit_spectra(models=models, stat=statistic, default_values=default_values, free=free, use_previous_fit=True)

    def on_spec_fit_dialog_cancelled(self, dialog):
        print("Spectral fit cancelled.")
        return
    
    def on_residual_selection_group(self, button, checked):
        """Show/Hide residuals."""

        if not checked:
            return  # ignore the one being turned off

        if button is self.rb_sigma_residuals:
            self.residual_selection = 'Sigma Residuals'
            self.show_residuals = True
        elif button is self.rb_count_residuals:
            self.residual_selection = 'Count Residuals'
            self.show_residuals = True
        elif button is self.rb_no_residuals:
            self.residual_selection = 'No Residuals'
            self.show_residuals = False
            
        self.update_plot()

    def on_count_display_selection(self):
        """Show/Hide spectral data (points)."""
        self.show_count_display = self.rb_show_counts.isChecked()

        data_successfully_removed = False
        data_successfully_added = False

        if not hasattr(self, "data_containers"):
            self.data_containers = []

        if self.show_count_display is False:
            for data_container in self.data_containers:
                try:
                    if self.count_data_visible:
                        # cap
                        data_container[2][0].set_visible(False)
                        # bar
                        data_container[2][1].set_visible(False)
                        # arrow
                        if len(data_container[1]) != 0:
                            data_container[1][0].set_visible(False)
                        data_successfully_removed = True
                except Exception:
                    pass

            if data_successfully_removed and self.count_data_visible:
                self.count_data_visible = False
        else:
            for data_container in self.data_containers:
                try:
                    if not self.count_data_visible:
                        data_container[2][0].set_visible(True)
                        data_container[2][1].set_visible(True)
                        if len(data_container[1]) != 0:
                            data_container[1][0].set_visible(True)
                        data_successfully_added = True
                except Exception:
                    pass

            if data_successfully_added and not self.count_data_visible:
                self.count_data_visible = True

        self.canvas.draw()

    def on_model_display_selection(self):
        """Show/Hide model curves."""
        self.show_model_display = self.rb_show_model.isChecked()

        if not hasattr(self, "model_artists"):
            self.model_artists = []

        if not self.show_model_display:
            for model_artist in self.model_artists:
                if model_artist.get_visible():
                    model_artist.set_visible(False)
        else:
            for model_artist in self.model_artists:
                if not model_artist.get_visible():
                    model_artist.set_visible(True)

        self.canvas.draw()

    def on_model_components_display_selection(self):
        """Placeholder for model component toggles."""
        self.show_spectral_model_components = True  # or however you want
        print("Function not yet implemented")

    def on_mouse_click(self, event):
        # print("Mouse button clicked")
        pass

    def on_mouse_move(self, event):
        pass

    def command(self):
        print('function not yet implemented')

    def set_x_log_scale(self, checked: bool):
        self.xlog = checked

        if self._plot_ready:
            self.update_plot()

    def set_y_log_scale(self, checked: bool):
        self.ylog = checked

        if self._plot_ready:
            self.update_plot()

    # def plot_spectrum(self):

    #     # Enable fit display menu
    #     self.fit_display_combo.setEnabled(True)

    #     # Enable/disable toggles based on statistic (Sigma Residuals logic)
    #     if self.data is not None and 'stat_name' in self.data:
    #         if 'Chi-Squared' in self.data['stat_name']:
    #             self.rb_sigma_residuals.setEnabled(True)
    #         else:
    #             # Deselect sigma residuals if currently selected
    #             if self.rb_sigma_residuals.isChecked():
    #                 self.rb_no_residuals.setChecked(True)
    #                 self.residual_selection = 'No Residuals'
    #             self.rb_sigma_residuals.setEnabled(False)

    #     # (From here down: mostly direct translation of your existing math/plotting)

    #     numberOfDataGroups = len(self.data['x_ufspec'])
    #     data_labels = list(self.gspec_manager.data.keys()) if self.gspec_manager else [
    #         f"Dataset {i+1}" for i in range(numberOfDataGroups)
    #     ]

    #     colors = ['#394264', '#6E8846', '#7B3F5B', '#aa611d', '#463965',
    #               '#93894B', '#396F45', '#93524B', '#553462', '#8F924A',
    #               '#2F545B', '#946F4B']
    #     colors_model = ['#8b0000'] * 10

    #     self.ax.clear()
    #     if self.ax2 is not None:
    #         self.ax2.clear()

    #     self.ax.set_autoscale_on(True)

    #     if 'No Residuals' in self.residual_selection:
    #         if self.ax2 is not None:
    #             self.figure.delaxes(self.ax)
    #             self.figure.delaxes(self.ax2)
    #             self.ax = self.figure.add_subplot(111)
    #             self.ax2 = None

    #     if 'No Residuals' not in self.residual_selection:
    #         if self.ax2 is None:
    #             gs = gridspec.GridSpec(4, 1)
    #             self.ax.set_position(gs[0:3].get_position(self.figure))
    #             self.ax.set_subplotspec(gs[0:3])
    #             self.ax.xaxis.set_ticklabels([])

    #             self.ax2 = self.figure.add_subplot(gs[3])
    #             self.ax2.set_xlabel('Energy (keV)', fontsize=PLOTFONTSIZE)

    #             minorLocator = AutoMinorLocator()
    #             self.ax2.xaxis.set_minor_locator(minorLocator)
    #             minorLocator = AutoMinorLocator()
    #             self.ax2.yaxis.set_minor_locator(minorLocator)

    #             self.ax.get_shared_x_axes().join(self.ax, self.ax2)
    #             self.figure.subplots_adjust(hspace=0)
    #             self.ax.tick_params(labelbottom=False)

    #     # Extract data arrays (as in original)
    #     energy_data = self.data['x_ufspec']
    #     energy_error_data = self.data['xerr_ufspec']
    #     energy_summed_model = self.data['energies_model']

    #     counts = self.data['counts']
    #     counts_error = self.data['counts_error']
    #     counts_model = self.data['counts_model']

    #     counts_integrated = self.data['counts_integrated']
    #     counts_integrated_model = self.data['counts_integrated_model']

    #     flux_data = self.data['y_ufspec']
    #     flux_error_data = self.data['yerr_ufspec']
    #     flux_model = self.data['y_ufspec_model']
    #     flux_summed_model = self.data['ufspec_model']

    #     residuals_sigma = self.data['residuals_sigma']

    #     energy_data_flattened = numpy.hstack(energy_data)
    #     xmin = numpy.floor(numpy.min(energy_data_flattened))
    #     xmax = numpy.ceil(numpy.max(energy_data_flattened))

    #     alpha_data = 1 if self.show_count_display else 0
    #     alpha_model = 1 if self.show_model_display else 0
    #     alpha_legend = 1 if self.show_legend_display else 0  # not used directly but left for clarity

    #     # Spectrum type selection
    #     if self.spectrum_selection == 'Cumulative':
    #         flux_data = counts_integrated
    #         flux_error_data = []
    #         for counts_integrated_group in counts_integrated:
    #             flux_error_data.append(numpy.zeros(len(counts_integrated_group)))
    #         flux_error_data = numpy.array(flux_error_data)
    #         flux_model = counts_integrated_model
    #         residuals_data = flux_data - flux_model
    #         self.ax.set_ylabel(r'Normalized integrated counts', fontsize=PLOTFONTSIZE)

    #     elif self.spectrum_selection == 'Raw Counts':
    #         flux_data = counts
    #         flux_error_data = counts_error
    #         flux_model = counts_model
    #         residuals_data = flux_data - flux_model
    #         self.ax.set_ylabel(r'Counts bin$^{-1}$', fontsize=PLOTFONTSIZE)

    #     elif self.spectrum_selection == 'Counts Spectrum':
    #         flux_data = counts / energy_error_data
    #         flux_error_data = (counts_error / counts) * flux_data
    #         flux_model = counts_model / energy_error_data
    #         residuals_data = flux_data - flux_model
    #         self.ax.set_ylabel(r'Counts s$^{-1}$ keV$^{-1}$', fontsize=PLOTFONTSIZE)

    #     elif self.spectrum_selection == 'Photon Spectrum':
    #         residuals_data = flux_data - flux_model
    #         self.ax.set_ylabel(r'Flux (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)', fontsize=PLOTFONTSIZE)

    #     elif self.spectrum_selection == 'Energy Spectrum':
    #         flux_error_data = (flux_error_data / flux_data) * (flux_data * energy_data)
    #         flux_data = flux_data * energy_data
    #         flux_model = flux_model * energy_data
    #         flux_summed_model = flux_summed_model * energy_summed_model
    #         residuals_data = flux_data - flux_model
    #         self.ax.set_ylabel(r'Energy (Photons cm$^{-2}$ s$^{-1}$)', fontsize=PLOTFONTSIZE)

    #     elif self.spectrum_selection == 'Nu Fnu Spectrum':
    #         flux_error_data = (flux_error_data / flux_data) * (flux_data * (energy_data ** 2))
    #         flux_data = flux_data * (energy_data ** 2)
    #         flux_model = flux_model * (energy_data ** 2)
    #         flux_summed_model = flux_summed_model * (energy_summed_model ** 2)
    #         residuals_data = flux_data - flux_model
    #         self.ax.set_ylabel(r'$\nu$ F$_{\nu}$ (Photons keV cm$^{-2}$ s$^{-1}$)', fontsize=PLOTFONTSIZE)
    #     else:
    #         residuals_data = flux_data - flux_model  # default

    #     # Detection vs upper limits
    #     index_detections = []
    #     index_upperLimits = []
    #     for dataGroup in range(numberOfDataGroups):
    #         index_detections.append(numpy.where((flux_data[dataGroup] - flux_error_data[dataGroup]) > 0))
    #         index_upperLimits.append(numpy.where((flux_data[dataGroup] - flux_error_data[dataGroup]) <= 0))

    #     self.data_containers = []

    #     for dataGroup, detections, upperLimits in zip(
    #         range(numberOfDataGroups), index_detections, index_upperLimits
    #     ):
    #         data_detections_container = self.ax.errorbar(
    #             energy_data[dataGroup][detections],
    #             flux_data[dataGroup][detections],
    #             xerr=energy_error_data[dataGroup][detections],
    #             capsize=0,
    #             fmt='none',
    #             zorder=1,
    #             ecolor=colors[dataGroup % len(colors)],
    #             alpha=alpha_data,
    #             yerr=flux_error_data[dataGroup][detections],
    #             label=data_labels[dataGroup]
    #         )

    #         data_limits_container = self.ax.errorbar(
    #             energy_data[dataGroup][upperLimits],
    #             flux_data[dataGroup][upperLimits],
    #             xerr=energy_error_data[dataGroup][upperLimits],
    #             capsize=0,
    #             fmt='none',
    #             zorder=1,
    #             ecolor=colors[dataGroup % len(colors)],
    #             alpha=alpha_data,
    #             yerr=flux_data[dataGroup][upperLimits] / 2.0,
    #             uplims=True
    #         )

    #         self.data_containers.append(data_detections_container)
    #         self.data_containers.append(data_limits_container)

    #     # Residuals
    #     if self.ax2 is not None:
    #         for dataGroup in range(numberOfDataGroups):
    #             if 'Count Residuals' in self.residual_selection:
    #                 self.ax2.plot(
    #                     [xmin, xmax], [0, 0],
    #                     linestyle='-', linewidth=0.8,
    #                     c=colors_model[dataGroup % len(colors_model)],
    #                     zorder=2, alpha=alpha_model
    #                 )
    #                 self.ax2.errorbar(
    #                     energy_data[dataGroup],
    #                     residuals_data[dataGroup],
    #                     xerr=energy_error_data[dataGroup],
    #                     capsize=0,
    #                     fmt='none',
    #                     ecolor=colors[dataGroup % len(colors)],
    #                     zorder=1,
    #                     alpha=alpha_data,
    #                     yerr=flux_error_data[dataGroup]
    #                 )

    #             if 'Sigma Residuals' in self.residual_selection:
    #                 self.ax2.plot(
    #                     [xmin, xmax], [0, 0],
    #                     linestyle='-', linewidth=0.8,
    #                     c=colors_model[dataGroup % len(colors_model)],
    #                     zorder=2, alpha=alpha_model
    #                 )
    #                 self.ax2.errorbar(
    #                     energy_data[dataGroup],
    #                     residuals_sigma[dataGroup],
    #                     xerr=energy_error_data[dataGroup],
    #                     capsize=0,
    #                     fmt='none',
    #                     ecolor=colors[dataGroup % len(colors)],
    #                     zorder=1,
    #                     alpha=alpha_data,
    #                     yerr=numpy.ones(len(residuals_sigma[dataGroup]))
    #                 )
    #                 self.ax2.set_ylabel('Sigma', fontsize=PLOTFONTSIZE)

    #     self.ax.set_xlabel('Energy (keV)', fontsize=PLOTFONTSIZE)

    #     minorLocator = AutoMinorLocator()
    #     self.ax.xaxis.set_minor_locator(minorLocator)
    #     minorLocator = AutoMinorLocator()
    #     self.ax.yaxis.set_minor_locator(minorLocator)

    #     self.ax.xaxis.set_tick_params(labelsize=PLOTFONTSIZE)
    #     self.ax.yaxis.set_tick_params(labelsize=PLOTFONTSIZE)

    #     self.ax.set_xlim(xmin / 2, xmax * 2)

    #     if self.xlog:
    #         self.ax.set_xscale('log')
    #         if self.ax2 is not None:
    #             self.ax2.set_xscale('log')

    #     if self.ylog:
    #         self.ax.set_yscale('log')

    #     flux_data_model_flattened = numpy.hstack(
    #         [numpy.hstack(flux_data), numpy.hstack(flux_model)]
    #     )
    #     ymin = numpy.floor(numpy.min(flux_data_model_flattened))
    #     ymax = numpy.ceil(numpy.max(flux_data_model_flattened))

    #     if ymin == 0:
    #         ymin = numpy.min(flux_data_model_flattened)
    #     try:
    #         self.ax.set_ylim(ymin / 2, ymax * 2)
    #     except Exception:
    #         pass

    #     self.ax.set_autoscale_on(False)

    #     # Model curves
    #     self.model_artists = []

    #     if self.spectrum_selection in ['Raw Counts', 'Counts Spectrum', 'Cumulative']:
    #         for dataGroup in range(numberOfDataGroups):
    #             for i in range(len(energy_data[dataGroup])):
    #                 energy_low = energy_data[dataGroup][i] - energy_error_data[dataGroup][i]
    #                 energy_high = energy_data[dataGroup][i] + energy_error_data[dataGroup][i]

    #                 artist, = self.ax.plot(
    #                     [energy_low, energy_high],
    #                     [flux_model[dataGroup][i], flux_model[dataGroup][i]],
    #                     c=colors_model[dataGroup % len(colors_model)],
    #                     zorder=2, linewidth=0.8, alpha=alpha_model
    #                 )
    #                 self.model_artists.append(artist)
    #                 try:
    #                     artist, = self.ax.plot(
    #                         [energy_high, energy_high],
    #                         [flux_model[dataGroup][i], flux_model[dataGroup][i + 1]],
    #                         c=colors_model[dataGroup % len(colors_model)],
    #                         zorder=2, linewidth=0.8, alpha=alpha_model
    #                     )
    #                     self.model_artists.append(artist)
    #                 except Exception:
    #                     pass
    #     else:
    #         artist, = self.ax.plot(
    #             energy_summed_model,
    #             flux_summed_model,
    #             c=colors_model[0],
    #             zorder=2,
    #             alpha=alpha_model
    #         )
    #         self.model_artists.append(artist)

    #     # Adjust residual axes ticks
    #     if self.ax2 is not None:
    #         yticks = self.ax2.get_yticks()
    #         if len(yticks) > 2:
    #             yticks = yticks[1:-1]
    #             self.ax2.set_yticks(yticks)
    #         self.ax2.xaxis.set_tick_params(labelsize=PLOTFONTSIZE)
    #         self.ax2.yaxis.set_tick_params(labelsize=PLOTFONTSIZE)

    #     # Legend
    #     if self.show_legend_display:
    #         if self.spectrum_selection == 'Cumulative':
    #             legend_location = 'lower right'
    #         elif self.spectrum_selection in ['Raw Counts', 'Counts Spectrum', 'Energy Spectrum']:
    #             legend_location = 'lower left'
    #         elif self.spectrum_selection == 'Photon Spectrum':
    #             legend_location = 'upper right'
    #         elif self.spectrum_selection == 'Nu Fnu Spectrum':
    #             legend_location = 'upper left'
    #         else:
    #             legend_location = 'best'

    #         self.legend = self.ax.legend(
    #             numpoints=1,
    #             scatterpoints=1,
    #             fontsize='x-small',
    #             frameon=False,
    #             loc=legend_location
    #         )

    #     self.canvas.draw()

    # def plot_batch_fit_results(self, parameter_index=(0,)):
    #     self.fit_diagnostics_options_menu.setEnabled(True)

    #     if not self.xlog:
    #         self.cb_x_log.setChecked(False)
    #     if not self.ylog:
    #         self.cb_y_log.setChecked(False)

    #     self.ax.clear()

    #     if self.ax2 is not None:
    #         self.figure.delaxes(self.ax)
    #         self.figure.delaxes(self.ax2)
    #         self.ax = self.figure.add_subplot(111)
    #         self.ax2 = None

    #     # disable all toggles except log ones
    #     for rb in [
    #         self.rb_show_counts, self.rb_hide_counts,
    #         self.rb_show_model, self.rb_hide_model,
    #         self.rb_sigma_residuals, self.rb_count_residuals,
    #         self.rb_no_residuals, self.rb_show_legend,
    #         self.rb_hide_legend
    #     ]:
    #         rb.setEnabled(False)

    #     model = self.data['model'][0]
    #     parameter_names = self.data['parameter_names']
    #     parameter_values = self.data['parameter_values']
    #     parameter_sigmas = self.data['parameter_sigmas']

    #     if len(parameter_index) == 1:
    #         time_start = self.data['time_start']
    #         time_end = self.data['time_end']
    #         time_exposure = self.data['exposure']

    #         time = time_start + (time_end - time_start) / 2.0
    #         dtime = (time_end - time_start) / 2.0

    #         x_name = 'Time (sec)'
    #         x_value = time
    #         x_sigma = dtime

    #         y_name = parameter_names[parameter_index[0]]
    #         y_value = parameter_values[:, parameter_index[0]]
    #         y_sigma = parameter_sigmas[:, parameter_index[0]]
    #     else:
    #         x_name = "%s - %s" % (model, parameter_names[parameter_index[0]])
    #         x_value = parameter_values[:, parameter_index[0]]
    #         x_sigma = parameter_sigmas[:, parameter_index[0]]

    #         y_name = parameter_names[parameter_index[1]]
    #         y_value = parameter_values[:, parameter_index[1]]
    #         y_sigma = parameter_sigmas[:, parameter_index[1]]

    #     self.ax.errorbar(
    #         x_value, y_value,
    #         xerr=x_sigma, yerr=y_sigma,
    #         capsize=0, fmt='none',
    #         zorder=1, ecolor='#394264', alpha=1
    #     )

    #     self.ax.set_xlabel(x_name, fontsize=PLOTFONTSIZE)
    #     self.ax.set_ylabel("%s - %s" % (model, y_name), fontsize=PLOTFONTSIZE)

    #     minorLocator = AutoMinorLocator()
    #     self.ax.xaxis.set_minor_locator(minorLocator)
    #     minorLocator = AutoMinorLocator()
    #     self.ax.yaxis.set_minor_locator(minorLocator)

    #     if self.xlog:
    #         self.ax.set_xscale('log')
    #         if self.ax2 is not None:
    #             self.ax2.set_xscale('log')

    #     if self.ylog:
    #         self.ax.set_yscale('log')

    #     self.ax.xaxis.set_tick_params(labelsize=PLOTFONTSIZE)
    #     self.ax.yaxis.set_tick_params(labelsize=PLOTFONTSIZE)

    #     self.canvas.draw()

    def on_fit_diag_options_selection(self, menu_selection: str):
        self.command()
        # if 'Plot Model Parameters...' in menu_selection:
        #     selected_model = self.data['model']
        #     # reset combo text
        #     self.fit_diagnostics_options_menu.blockSignals(True)
        #     self.fit_diagnostics_options_menu.setCurrentText('Plot Model Parameters...')
        #     self.fit_diagnostics_options_menu.blockSignals(False)

        #     dialog = PlotModelParametersDialog(
        #         self, selected_model, self.plot_batch_fit_results
        #     )
        #     dialog.exec()
        # else:
        #     print("function not yet implemented")
        #     self.fit_diagnostics_options_menu.blockSignals(True)
        #     self.fit_diagnostics_options_menu.setCurrentText('Plot Model Parameters...')
        #     self.fit_diagnostics_options_menu.blockSignals(False)

    def quit_confirmation(self):
        # you could show a QMessageBox if you want
        self.close()

    def on_window_close(self):
        self.close()

    def update_plot(self):

        if self.model_plot is None:
            print('Creating plot...')
            self.create_plot()
    
        self.model_plot._view = self.spectrum_view
        self.model_plot.set_fit(fitter=self.grb.specfitter)

        # Set the xscale 
        if self.xlog == True:
            self.canvas.axes.set_xscale('log')
        else:
            self.canvas.axes.set_xscale('linear')

        # Set the yscale
        if self.ylog == True:
            self.canvas.axes.set_yscale('log')
        else:
            self.canvas.axes.set_yscale('linear')

        # Set the residuals display
        if self.show_residuals is True:
            self.model_plot.show_residuals()
        else:
            self.model_plot.hide_residuals()
        
        # Grid lines
        if self.show_grid:
            self.canvas.axes.grid(True, alpha=0.25)

        # Legend
        legend = self.canvas.axes.get_legend()
        if legend is not None:
            legend.set_visible(self.show_legend)

        self.canvas.draw() 

        # # Set the plotting limits
        # self.canvas.axes.set_xlim(self.xlim)
        # self.canvas.axes.set_ylim(self.ylim)

    def reset_menu(self, menu):

        # Block the menu from executing any commands
        menu.blockSignals(True)

        # Ensure no item is selected initially
        menu.setCurrentIndex(-1)         
        
        # Disable the placeholder 
        menu.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  

        # Re-enable the menu 
        menu.blockSignals(False)            

    def closeEvent(self, event):
        """Cleanup when the window is closed so the FitPlotter can be garbage-collected.

        - closes and deletes canvases
        - clears references to large objects
        """

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


        # # Remove self from parent GRB viewer list
        # try:
        #     if getattr(self, 'grb', None) is not None and getattr(self.grb, '_pha_viewers', None) is not None:
        #         try:
        #             if self in self.grb._pha_viewers:
        #                 self.grb._pha_viewers.remove(self)
        #         except Exception:
        #             pass
        # except Exception:
        #     pass

        # Clear other large references
        try:
            self.model_plot = None
            self.toolbar = None
            self.grb._fit_plotter = None

        except Exception:
            pass

        # Finally, call the base implementation to actually close the window
        try:
            super().closeEvent(event)
        except Exception:
            # If super fails (unlikely), accept the event anyway
            event.accept()

        return
    
    def update_window(self):
        self.update_plot()