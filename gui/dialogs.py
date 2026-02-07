import os
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QDialog, QVBoxLayout, QTextEdit,
    QScrollBar, QLabel, QLineEdit, QGridLayout,
    QListWidget, QPushButton, QRadioButton, QButtonGroup, QFrame,
    QHBoxLayout, QMessageBox, QCheckBox, QSpacerItem, QSizePolicy,
    QDoubleSpinBox, QComboBox, QDialogButtonBox, QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtGui import QCloseEvent

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class Dialog(QDialog):
    # @gspec_debug
    def __init__(self, title=None, parent=None, width=250, height=250):
        super().__init__(parent)

        # GUI settings
        # self.bg_color = '#e1e1e1'
        # self.fg_color = 'black'
        # self.font_color = 'black'
        # self.font = QFont('Helvetica Neue', 14)

        self.setWindowTitle(title or "")

        # Size
        self.resize(width, height)

        # Style
        # self.setAutoFillBackground(True)
        # self.setFont(self.font)

        # Center on parent or screen
        self._center_on_parent_or_screen()

    def _center_on_parent_or_screen(self):
        """Center dialog on parent if available, otherwise on primary screen."""
        parent_widget = self.parentWidget()   # <-- use Qt API

        if parent_widget is not None:
            parent_geom = parent_widget.frameGeometry()
            center_point = parent_geom.center()
        else:
            screen = QApplication.primaryScreen()
            if screen is None:
                return
            center_point = screen.availableGeometry().center()

        geom = self.frameGeometry()
        geom.moveCenter(center_point)
        self.move(geom.topLeft())

class Splash(Dialog):
    dir = os.path.dirname(os.path.realpath(__file__))
    _IMAGE_PATH = os.path.join(dir, "gdt_logo.png")

    def __init__(self, parent=None):
        super().__init__(title='', parent=parent, width=300, height=435)

        # Frameless, translucent, on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image
        pixmap = QPixmap(self._IMAGE_PATH)
        image_label = QLabel(self)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("background: transparent;")
        layout.addWidget(image_label)

        # Text lines
        lines = [
            '\nGSpec',
            'GBM Spectral Analysis Package',
            f'Version {0.1}\n',
            'Adam Goldstein',
            'Daniel Kocevski',
            'Rob Preece',
            'William Cleveland\n',
            'Universities Space Research Association',
            'NASA Marshall Space Flight Center',
            'University of Alabama in Huntsville',
            ''
        ]

        for line in lines:
            label = QLabel(line, self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                f"color: {self.font_color}; background-color: {self.bg_color};"
            )
            label.setFont(self.font)
            layout.addWidget(label)

        # Any key / mouse dismisses
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.grabKeyboard()
        self.grabMouse()

    def keyPressEvent(self, event):
        self.dismiss()

    def mousePressEvent(self, event):
        self.dismiss()

    def dismiss(self):
        self.done(QDialog.DialogCode.Accepted)

class TextDisplayDialog(Dialog):
    """A dialog that displays text with self-adjusting scrollbars.

    Args:
        thetext (str): The text to display
        title (str, optional): The title of the dialog box
        parent (QWidget, optional): The parent PyQt6 window
    """
    def __init__(self, thetext, title=None, parent=None,
                 window_width=600, window_height=800, **kwargs):
        super().__init__(title=title, parent=parent,
                         width=window_width, height=window_height)

        # Layout
        layout = QVBoxLayout(self)

        # Text widget
        text_edit = QTextEdit(self)
        text_edit.setPlainText(thetext)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        text_edit.setFont(self.font)
        text_edit.setStyleSheet(
            f"background-color: {self.bg_color}; color: {self.font_color};"
        )
        layout.addWidget(text_edit)

        # Scrollbars
        y_scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        text_edit.setVerticalScrollBar(y_scrollbar)

        x_scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        text_edit.setHorizontalScrollBar(x_scrollbar)

        # Bring to front
        self.raise_()

class OptionDialog(Dialog):
    """
    A dialog that presents a list of options on buttons.

    Args:
        options (list): A list of option labels.
        commands (list): A list of callables corresponding to each option.
        title (str, optional): The title of the dialog box.
        message (str, optional): Informational message.
        parent (QWidget, optional): The parent QWidget.
        width (int, optional): Width in pixels.
        height (int, optional): Height in pixels.
    """

    def __init__(self, options, commands, title=None, message=None,
                 parent=None, width=160, height=220):
        super().__init__(title=title or "Options",
                         parent=parent, width=width, height=height)

        self.commands = commands

        layout = QVBoxLayout(self)

        # Optional message
        if message:
            message_label = QLabel(message, self)
            message_label.setWordWrap(True)
            message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(message_label)

        # Buttons
        for i, option in enumerate(options):
            button = QPushButton(option, self)
            button.clicked.connect(self._create_button_handler(commands[i]))
            layout.addWidget(button)

    def _create_button_handler(self, command):
        """Wrap command to execute and close dialog."""
        def handler():
            command()
            self.accept()
        return handler

class PlotDialog(Dialog):
    """A simple dialog intended to host a plotting widget."""
    def __init__(self, title=None, parent=None,
                 width=800, height=600, plot_widget: QWidget | None = None):
        super().__init__(title=title, parent=parent, width=width, height=height)

        layout = QVBoxLayout(self)

        if plot_widget is not None:
            layout.addWidget(plot_widget)

        self.setLayout(layout)

class TextOptionDialog(Dialog):
    """A dialog that presents a parameter input and radio toggle.

    Args:
        title (str, optional): Dialog title.
        parent (QWidget, optional): Parent window.
        width (int, optional): Width in pixels.
        height (int, optional): Height in pixels.
        message (str, optional): Informational message.
        default_value (int or float, optional): Default numeric value.
        button_label (str, optional): Label for the main button.
        button_options (tuple, optional): 2-tuple of radio options.
        button_command (function): Callback called on accept/cancel.
    """
    def __init__(
        self, title=None, parent=None, width=160, height=220, message=None,
        default_value=0, button_label="Accept", button_options=None, button_command=None
    ):
        super().__init__(title=title, parent=parent, width=width, height=height)

        self.button_command = button_command

        layout = QVBoxLayout(self)

        # Message
        if message:
            label_message = QLabel(message, self)
            label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label_message)

        # Radio buttons
        self.option_group = QButtonGroup(self)
        option1_text = button_options[0] if button_options else "Full Range"
        option2_text = button_options[2] if button_options else "View Range"
        option3_text = button_options[1] if button_options else "Source Range"

        radio1 = QRadioButton(option1_text, self)
        radio2 = QRadioButton(option2_text, self)
        radio3 = QRadioButton(option3_text, self)
        radio2.setChecked(True)
        self.option_group.addButton(radio1)
        self.option_group.addButton(radio2)
        self.option_group.addButton(radio3)

        radio_layout = QVBoxLayout()
        radio_layout.addWidget(radio1)
        radio_layout.addWidget(radio2)
        radio_layout.addWidget(radio3)

        # Input field
        self.default_input = QLineEdit(self)
        self.default_input.setText(str(default_value))
        self.default_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.default_input.setMaximumWidth(self.width() // 3)

        radio_input_layout = QHBoxLayout()
        radio_input_layout.addLayout(radio_layout)
        radio_input_layout.addWidget(self.default_input)
        layout.addLayout(radio_input_layout)    

        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout.addItem(spacer)

        # Buttons
        button_layout = QHBoxLayout()
        accept_button = QPushButton(button_label, self)
        accept_button.clicked.connect(self._evaluate_button_command)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.on_window_close)
        button_layout.addWidget(accept_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.finished.connect(self.on_window_close)

    def _evaluate_button_command(self):
        try:
            value = float(self.default_input.text())
        except ValueError:
            value = None

        # selected_option = self.option_group.checkedButton().text()
        # self.button_command(value, selected_option)
        self.button_command(value)

        self.accept()

    def on_window_close(self):
        """Handle cancel/close:"""
        if getattr(self, "_closing", False):
            return
        self._closing = True

        self.reject()

class ManualInputDialog(Dialog):
    """Manual entry for X and/or Y range selection."""

    def __init__(
        self,
        command,
        title=None,
        parent=None,
        message=None,
        xinput=True,
        yinput=True,
        xdefaults=None,
        ydefaults=None,
        width=165,
        height=120,
    ):
        if xdefaults is None:
            xdefaults = (0.0, 0.0)
        if ydefaults is None:
            ydefaults = (0.0, 0.0)

        # Adjust height based on input fields
        if xinput:
            height += 50
        if yinput:
            height += 50

        super().__init__(title=title, parent=parent, width=width, height=height)
        self.button_command = command
        self._xinput = xinput
        self._yinput = yinput

        layout = QVBoxLayout(self)

        # Message
        if message:
            message_label = QLabel(message, self)
            message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(message_label)

        # Inputs
        self._x0default = QLineEdit(self)
        self._x0default.setText(str(xdefaults[0]))

        self._x1default = QLineEdit(self)
        self._x1default.setText(str(xdefaults[1]))

        self._y0default = QLineEdit(self)
        self._y0default.setText(str(ydefaults[0]))

        self._y1default = QLineEdit(self)
        self._y1default.setText(str(ydefaults[1]))

        grid_layout = QGridLayout()
        row = 0
        if xinput:
            grid_layout.addWidget(QLabel("X Lo:"), row, 0)
            grid_layout.addWidget(self._x0default, row, 1)
            row += 1
            grid_layout.addWidget(QLabel("X Hi:"), row, 0)
            grid_layout.addWidget(self._x1default, row, 1)
            row += 1

        if yinput:
            grid_layout.addWidget(QLabel("Y Lo:"), row, 0)
            grid_layout.addWidget(self._y0default, row, 1)
            row += 1
            grid_layout.addWidget(QLabel("Y Hi:"), row, 0)
            grid_layout.addWidget(self._y1default, row, 1)
            row += 1

        layout.addLayout(grid_layout)

        # Buttons
        button_layout = QHBoxLayout()
        accept_button = QPushButton("Accept", self)
        accept_button.clicked.connect(self._evaluate_button_command)
        button_layout.addWidget(accept_button)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.on_window_close)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.raise_()

    def _evaluate_button_command(self):
        try:
            x0 = float(self._x0default.text())
            x1 = float(self._x1default.text())
            y0 = float(self._y0default.text())
            y1 = float(self._y1default.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Inputs must be numbers.")
            return

        if not self._xinput:
            x0, x1 = None, None
        if not self._yinput:
            y0, y1 = None, None

        self.button_command(x0, x1, y0, y1)
        self.accept()

    def on_window_close(self):
        self.button_command(None, None, None, None)
        self.reject()

class FitOptionsDialog(Dialog):
    """A dialog that presents a list of models to fit and fitting options."""

    def __init__(self, models, parent=None, batch_fit=False,
                 width=400, height=700):
        title = "Photon Models"
        super().__init__(title=title, parent=parent, width=width, height=height)

        self.models = models
        self.batch_fit = batch_fit
        self.fit_options = None
        self.default_values = None
        self.free = None
        self._construct()

    def cancel_fit(self):
        self.fit_options = None
        self.reject()

    def _construct(self):
        layout = QVBoxLayout(self)

        # Title
        label_title = QLabel("Select one or more photon models", self)
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_title)

        # Model list
        self.model_listbox = QListWidget(self)
        self.model_listbox.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        # self.model_listbox.addItems(self.models.model_names)
        self.model_listbox.addItems(self.models)
        self.model_listbox.itemSelectionChanged.connect(self.on_select_model)
        layout.addWidget(self.model_listbox)

        # Model parameters
        frame_model_parameters = QFrame(self)
        frame_model_parameters_layout = QVBoxLayout(frame_model_parameters)
        label_model_parameters = QLabel("Photon Model Parameters:", self)
        label_model_parameters.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_model_parameters_layout.addWidget(label_model_parameters)

        self.set_parameters_button = QPushButton("Set Parameters", self)
        self.set_parameters_button.clicked.connect(self.on_select_new_parameters)
        self.set_parameters_button.setEnabled(False)
        self.set_parameters_button.setFixedWidth(180)
        frame_model_parameters_layout.addWidget(
            self.set_parameters_button,
            alignment=Qt.AlignmentFlag.AlignHCenter
        )
        layout.addWidget(frame_model_parameters)

        # Statistic
        label_fit_statistic = QLabel("Fitting Statistic:", self)
        label_fit_statistic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_fit_statistic)

        self.statistic_group = QButtonGroup(self)
        fit_stat_container = QWidget(self)
        fit_stat_container.setFixedWidth(350)
        frame_fit_statistic = QHBoxLayout(fit_stat_container)
        for text, value in zip(["Chi2", "C-Stat", "P-Stat", "PG-Stat"],
                               ["chisq", "cstat", "pstat", "pgstat"]):
            rb = QRadioButton(text, self)
            rb.setChecked(text == "PG-Stat")
            self.statistic_group.addButton(rb)
            frame_fit_statistic.addWidget(rb)
        layout.addWidget(fit_stat_container)
        layout.setAlignment(fit_stat_container, Qt.AlignmentFlag.AlignHCenter)

        # Weights
        label_fit_weights = QLabel("Fit Weighting:", self)
        label_fit_weights.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_fit_weights)

        self.weight_group = QButtonGroup(self)
        fit_weight_container = QWidget(self)
        fit_weight_container.setFixedWidth(200)
        frame_fit_weights = QHBoxLayout(fit_weight_container)
        for text in ["Standard", "Model"]:
            rb = QRadioButton(text, self)
            rb.setChecked(text == "Standard")
            self.weight_group.addButton(rb)
            frame_fit_weights.addWidget(rb)
        layout.addWidget(fit_weight_container)
        layout.setAlignment(fit_weight_container, Qt.AlignmentFlag.AlignHCenter)

        # Batch options
        if self.batch_fit:
            label_undetermined_values = QLabel(
                "Undetermined Values in Batch Fit:", self
            )
            label_undetermined_values.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label_undetermined_values)

            self.batch_group = QButtonGroup(self)
            batch_container = QWidget(self)
            batch_container.setFixedWidth(250)
            frame_undetermined_values = QHBoxLayout(batch_container)
            for text in ["Leave free", "Automatically fix"]:
                rb = QRadioButton(text, self)
                rb.setChecked(text == "Leave free")
                self.batch_group.addButton(rb)
                frame_undetermined_values.addWidget(rb)
            # layout.addLayout(frame_undetermined_values)
            layout.addWidget(batch_container)
            layout.setAlignment(batch_container, Qt.AlignmentFlag.AlignHCenter)

        # Buttons
        button_layout = QHBoxLayout()
        self.accept_button = QPushButton("Accept", self)
        self.accept_button.clicked.connect(self.perform_spectral_fit)
        self.accept_button.setEnabled(False)
        button_layout.addWidget(self.accept_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_fit)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def on_select_model(self):
        self.accept_button.setEnabled(True)
        self.set_parameters_button.setEnabled(True)

    def on_select_new_parameters(self):
        selected_models = [item.text() for item in self.model_listbox.selectedItems()]
        print(selected_models)
        for model in selected_models:
            print(model)

    def on_select_new_parameters(self):
        selected_models = [item.text() for item in self.model_listbox.selectedItems()]
        print(selected_models)

        if not selected_models:
            return  # or show a warning message

        self.parameter_options_dialog = ParameterOptionsDialog(
            grb=self.parent().grb,
            model_names=selected_models,
            parent=self,            # <-- parent is this FitOptionsDialog
        )

        self.parameter_options_dialog.accepted.connect(lambda: self.on_param_dialog_finished( self.parameter_options_dialog))
        self.parameter_options_dialog.rejected.connect(lambda: self.on_param_dialog_cancelled( self.parameter_options_dialog))

        self.parameter_options_dialog.show()
        self.parameter_options_dialog.raise_()
        self.parameter_options_dialog.activateWindow()

    def on_param_dialog_finished(self, dialog):
    
        self.fit_model = dialog.fit_model
        self.default_values = dialog.default_values
        self.free = dialog.free

    def on_param_dialog_cancelled(self, dialog):
        print("Parameter dialog cancelled.")
        return
    
    def perform_spectral_fit(self):
        print("perform_spectral_fit called")
        statistic = self.statistic_group.checkedButton().text()
        weights = self.weight_group.checkedButton().text()
        selected_models = [item.text() for item in self.model_listbox.selectedItems()]

        self.fit_options = (selected_models, statistic, weights, self.default_values, self.free)

        self.accept()

class ParameterOptionsDialog(Dialog):
    """Dialog to modify parameter defaults and fixed/free state."""

    def __init__(self, grb, model_names, parent=None):
        title = "Model Parameters"

        fit_model = grb.resolve_spectral_model(model_names[0])
        for model_name in model_names[1:]:
            print(model_name)
            model = grb.resolve_spectral_model(model_name)
            fit_model += model

        nparams = fit_model.nparams
        width = 400
        height = 110 + 25 * nparams

        super().__init__(title=title, parent=parent, width=width, height=height)
    
        # Values and states  
        self.grb = grb
        self.fit_model = fit_model
        self.model_name = fit_model.name    
        self.params = fit_model.param_list
        self.values = fit_model.default_values
        self.states = fit_model.free
        self.param_names = []
        self.units = []
        self.description = []
        self.disable_reset_button = True

        for param in self.params:
            self.param_names.append(param[0])
            self.units.append(param[1])
            self.description.append(param[2])

        # Use the parameters from the last fit if available
        if self.grb.specfitter is not None and self.grb.specfitter.success and self.grb.fit_model.name == self.model_name:
            self.values = np.array(self.values, dtype=float)
            self.values[np.array(self.states, dtype=bool)] = self.grb.specfitter.parameters
            self.disable_reset_button = False       

        self._construct()

    def _construct(self):
        layout = QVBoxLayout(self)

        # Header
        header_layout = QGridLayout()
        layout.addLayout(header_layout)

        label_title = QLabel(self.model_name, self)
        label_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(label_title, 0, 0, 1, 4)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        header_layout.addWidget(separator, 1, 0, 1, 4)

        # Parameters grid
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(20)
        layout.addLayout(grid_layout)

        grid_layout.addWidget(QLabel("Parameter", self), 0, 0)
        grid_layout.addWidget(QLabel("Value", self), 0, 1)
        grid_layout.addWidget(QLabel("Units", self), 0, 2)
        grid_layout.addWidget(QLabel("Fixed", self), 0, 3)

        self.entries = []
        self.checkboxes = []

        for i, param in enumerate(self.params):
            param_name = QLabel(self.param_names[i], self)
            param_name.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            grid_layout.addWidget(param_name, i + 1, 0)

            entry = QLineEdit(self)
            entry.setFixedHeight(20)
            try:
                entry.setText(f"{float(self.values[i]):.2f}")
            except (ValueError, TypeError):
                entry.setText(str(self.values[i]))
            entry.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.entries.append(entry)
            grid_layout.addWidget(entry, i + 1, 1)

            label_units = QLabel(self.units[i], self)
            grid_layout.addWidget(label_units, i + 1, 2)

            checkbox = QCheckBox(self)
            checkbox.setChecked(not self.states[i])
            self.checkboxes.append(checkbox)
            grid_layout.addWidget(checkbox, i + 1, 3)

        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        accept_button = QPushButton("Accept", self)
        accept_button.clicked.connect(self.set_params)
        button_layout.addWidget(accept_button)


        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self.reset_params)
        button_layout.addWidget(reset_button)

        if self.disable_reset_button:
            reset_button.setEnabled(False)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

    def set_params(self):
        nparams = len(self.params)
        self.default_values = []
        self.free = []
        for i in range(nparams):
            param_name = self.param_names[i]
            self.default_values.append(float(self.entries[i].text()))
            self.free.append(not self.checkboxes[i].isChecked()) 

        self.accept()

    def reset_params(self):
        nparams = len(self.params)
        self.values = self.fit_model.default_values
        self.states = self.fit_model.free

        for i in range(nparams):
            try:
                self.entries[i].setText(f"{float(self.values[i]):.2f}")
            except (ValueError, TypeError):
                self.entries[i].setText(str(self.values[i]))
            self.checkboxes[i].setChecked(not self.states[i])

class PlotModelParametersDialog(Dialog):
    """A dialog to select model parameters for plotting."""

    def __init__(self, master, selected_model, callback):
        title = "Batch Fit Plotter"
        width = 325
        height = 300

        super().__init__(title=title, parent=master, width=width, height=height)

        self.callback = callback
        self.selected_model = selected_model
        self.selection_history = []
        self.parameter_names = []

        self._construct()

    def _construct(self):
        layout = QVBoxLayout(self)

        label_title = QLabel(
            "Select one parameter to plot against time\n"
            "or two parameters to plot against each other",
            self,
        )
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_title)

        self.parameter_listbox = QListWidget(self)
        self.parameter_listbox.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.parameter_listbox.itemSelectionChanged.connect(self.on_select_parameter)
        layout.addWidget(self.parameter_listbox)

        self.populate_parameter_listbox()

        button_layout = QHBoxLayout()
        self.plot_fit_parameter = QPushButton("Plot", self)
        self.plot_fit_parameter.setEnabled(False)
        self.plot_fit_parameter.clicked.connect(self.plot_parameter)
        button_layout.addWidget(self.plot_fit_parameter)

        self.cancel_fit_parameters = QPushButton("Cancel", self)
        self.cancel_fit_parameters.clicked.connect(self.cancel)
        button_layout.addWidget(self.cancel_fit_parameters)

        layout.addLayout(button_layout)

    def populate_parameter_listbox(self):
        # TODO: replace placeholder with real parameter names from model manager
        self.parameter_names = ['TBD']
        for parameter_name in self.parameter_names:
            self.parameter_listbox.addItem(parameter_name)

    def on_select_parameter(self):
        self.plot_fit_parameter.setEnabled(bool(self.parameter_listbox.selectedItems()))

    def plot_parameter(self):
        selected_parameters = [
            item.text() for item in self.parameter_listbox.selectedItems()
        ]
        if selected_parameters:
            self.callback(selected_parameters)
        self.accept()

    def cancel(self):
        self.reject()

class MatplotlibDialog(QDialog):
    """
    Generic dialog that hosts a Matplotlib Figure and Canvas.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        The Matplotlib figure associated with this dialog.
    canvas : FigureCanvasQTAgg
        The canvas used to draw the figure.
    axes : matplotlib.axes.Axes or np.ndarray of Axes
        Axes created by `figure.subplots(...)`. Shape depends on nrows/ncols.
    """
    def __init__(
        self,
        parent=None,
        title: str | None = "Plot",
        nrows: int = 1,
        ncols: int = 1,
        sharex=False,
        sharey=False,
        with_toolbar: bool = True,
        figsize=(6, 4),
        dpi: int = 100,
        gridspec_kw: dict | None = None,
    ):
        super().__init__(parent)

        if title is not None:
            self.setWindowTitle(title)

        # --- Figure & Canvas ---
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.setMinimumSize(900, 700)

        # Create default axes grid (you can always clear & recreate later)
        self.axes = self.figure.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=gridspec_kw
        )

        # --- Layout ---
        layout = QVBoxLayout(self)

        if with_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

    # Convenience methods
    def draw(self):
        """Redraw the canvas."""
        self.canvas.draw_idle()

    def clear(self):
        """Clear the figure and remove existing axes."""
        self.figure.clear()
        self.axes = None
        self.draw()

    def create_subplots(self, nrows=1, ncols=1, sharex=False, sharey=False):
        """
        Clear the figure and create a new set of subplots.

        Returns
        -------
        axes : matplotlib.axes.Axes or np.ndarray of Axes
        """
        self.figure.clear()
        self.axes = self.figure.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
        )
        self.draw()
        return self.axes

    def closeEvent(self, event: QCloseEvent) -> None:
        pass
    
        # # Optional: call a callback supplied by the creator
        # if self.on_close_callback is not None:
        #     self.on_close_callback()

        # # Always call the base-class implementation at the end
        # super().closeEvent(event)    

class BayesianBlocksDialog(Dialog):
    """
    Dialog to configure Bayesian Blocks parameters for get_bayesian_blocks().
    Returns a dict of parameters via .params or run_dialog().
    """

    def __init__(self, parent=None, initial=None, width=400, height=0):
        title = "Bayesian Blocks"
        super().__init__(title=title, parent=parent, width=width, height=height or 0)

        defaults = {
            "auto_detect_flares": True,
            "p0": 0.05,
            "bg_method": "ends",        # "ends" or "low_quartile"
            "bg_percentage": 0.25,
            "ends_frac": 0.1,
            "sigma_thresh": 5,
            "factor_thresh": 1.5,
            "min_block_duration": 0.064,
            "merge_gap": 0.25,
        }
        if initial:
            defaults.update(initial)

        self._params = defaults.copy()

        self._build_ui(defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        # form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel('Bayesian Block Analysis Options', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # p0
        spin_width = 120

        self.p0_spin = QDoubleSpinBox()
        self.p0_spin.setDecimals(2)
        self.p0_spin.setRange(0.0001, 1.0)
        self.p0_spin.setSingleStep(0.01)
        self.p0_spin.setValue(float(defaults["p0"]))
        self.p0_spin.setFixedWidth(spin_width)
        form.addRow("p₀ (False Positive Rate):", self.p0_spin)

        # min_block_duration
        self.min_block_duration_spin = QDoubleSpinBox()
        self.min_block_duration_spin.setDecimals(3)
        self.min_block_duration_spin.setRange(0.0, 1e6)
        self.min_block_duration_spin.setSingleStep(0.001)
        self.min_block_duration_spin.setValue(float(defaults["min_block_duration"]))
        self.min_block_duration_spin.setFixedWidth(spin_width)
        form.addRow("Minimum Block Duration (s):", self.min_block_duration_spin)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        self.auto_detect_checkbox = QCheckBox(
            "Automatically Determine Flaring Episodes:", self
        )
        # Center the checkbox using layout alignment instead of setAlignment
        self.auto_detect_checkbox.setChecked(bool(defaults["auto_detect_flares"]))
        self.auto_detect_checkbox.toggled.connect(self._toggle_flaring_fields)
        form.addRow(self.auto_detect_checkbox)
        form.setAlignment(self.auto_detect_checkbox, Qt.AlignmentFlag.AlignCenter)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(3)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        self.sigma_thresh_spin = QDoubleSpinBox()
        self.sigma_thresh_spin.setDecimals(4)
        self.sigma_thresh_spin.setRange(1, 25)
        self.sigma_thresh_spin.setSingleStep(0.1)
        self.sigma_thresh_spin.setValue(float(defaults["sigma_thresh"]))
        self.sigma_thresh_spin.setFixedWidth(spin_width)
        form.addRow("Flare Threshold (sigma):", self.sigma_thresh_spin)

        # merge_gap
        self.merge_gap_spin = QDoubleSpinBox()
        self.merge_gap_spin.setDecimals(4)
        self.merge_gap_spin.setRange(0.0, 1e6)
        self.merge_gap_spin.setSingleStep(0.001)
        self.merge_gap_spin.setValue(float(defaults["merge_gap"]))
        self.merge_gap_spin.setFixedWidth(spin_width)
        form.addRow("Merge Episodes with Gaps <= (s):", self.merge_gap_spin)

        # Initialize enabled/disabled state for flaring-related inputs
        self._toggle_flaring_fields(self.auto_detect_checkbox.isChecked())

        main_layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents

    # ----- Helpers ---------------------------------------------------------
    def _toggle_flaring_fields(self, enabled: bool):
        """Enable/disable fields related to automatic flare detection."""
        for widget in (
            self.sigma_thresh_spin,
            # self.factor_thresh_spin,
            # self.min_block_duration_spin,
            self.merge_gap_spin,
        ):
            widget.setEnabled(enabled)

    @staticmethod
    def _parse_optional_float(line_edit: QLineEdit, field_name: str):
        text = line_edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{field_name} must be a number or left blank.")

    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""
        try:
            params = {
                "auto_detect_flares": bool(self.auto_detect_checkbox.isChecked()),
                "p0": float(self.p0_spin.value()),
                # "bg_method": self.bg_method_combo.currentText(),
                # "bg_percentage": float(self.bg_pct_spin.value()),
                # "ends_frac": float(self.ends_frac_spin.value()),
                "sigma_thresh": self._parse_optional_float(
                    self.sigma_thresh_spin, "sigma_thresh"
                ),
                # "factor_thresh": float(self.factor_thresh_spin.value()),
                "min_block_duration": float(self.min_block_duration_spin.value()),
                "merge_gap": float(self.merge_gap_spin.value()),
            }
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = BayesianBlocksDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None

class StructureFunctionDialog(Dialog):
    """
    Dialog to configure Structure Function parameters for calc_structure_function().
    Returns a dict of parameters via .parameters or run_dialog().
    """

    def __init__(self, parent=None, initial=None, width=400, height=0):
        title = "Structure Function"
        super().__init__(title=title, parent=parent, width=width, height=height or 0)

        defaults = {
            "max_lag": 25.0,
            "mc_trials": 500,
            "mc_quantiles": (0.05, 0.99),
            "mvt_rule": "mc",
            # "mvt_rule": "analytic",
            # "analytic_nsigma": 10.0,
            # "min_pairs_per_lag": 50,
        }
        if initial:
            defaults.update(initial)

        self._params = defaults.copy()

        self._build_ui(defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        # form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel('Structure Function Analysis Options', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # max_lag
        spin_width = 120

        self.max_lag_spin = QDoubleSpinBox()
        self.max_lag_spin.setDecimals(2)
        self.max_lag_spin.setRange(0.1, 100)
        self.max_lag_spin.setSingleStep(0.1)
        self.max_lag_spin.setValue(float(defaults["max_lag"]))
        self.max_lag_spin.setFixedWidth(spin_width)
        form.addRow("Maximum Lag in Seconds:", self.max_lag_spin)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        self.monte_carlo_checkbox = QCheckBox(
            "Use Monte Carlo to Estimate Noise-Only SF Band:", self
        )
        # Center the checkbox using layout alignment instead of setAlignment
        self.monte_carlo_checkbox.setChecked(True)
        self.monte_carlo_checkbox.toggled.connect(self._toggle_monte_carlo_fields)
        form.addRow(self.monte_carlo_checkbox)
        form.setAlignment(self.monte_carlo_checkbox, Qt.AlignmentFlag.AlignCenter)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(3)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # min_block_duration
        self.mc_trials_spin = QDoubleSpinBox()
        self.mc_trials_spin.setDecimals(0)
        self.mc_trials_spin.setRange(1, 1e6)
        self.mc_trials_spin.setSingleStep(1)
        self.mc_trials_spin.setValue(int(defaults["mc_trials"]))
        self.mc_trials_spin.setFixedWidth(spin_width)
        form.addRow("Monte Carlo Trials:", self.mc_trials_spin)

        self.mc_quantiles_lo_spin = QDoubleSpinBox()
        self.mc_quantiles_lo_spin.setDecimals(4)
        self.mc_quantiles_lo_spin.setRange(0.0, 1.0)
        self.mc_quantiles_lo_spin.setSingleStep(0.01)
        self.mc_quantiles_lo_spin.setValue(float(defaults["mc_quantiles"][0]))
        self.mc_quantiles_lo_spin.setFixedWidth(spin_width)
        form.addRow("Quantiles for MC bands (low):", self.mc_quantiles_lo_spin)

        self.mc_quantiles_hi_spin = QDoubleSpinBox()
        self.mc_quantiles_hi_spin.setDecimals(4)
        self.mc_quantiles_hi_spin.setRange(0.0, 1.0)
        self.mc_quantiles_hi_spin.setSingleStep(0.01)
        self.mc_quantiles_hi_spin.setValue(float(defaults["mc_quantiles"][1]))
        self.mc_quantiles_hi_spin.setFixedWidth(spin_width)
        form.addRow("Quantiles for MC bands (high):", self.mc_quantiles_hi_spin)

        # Initialize enabled/disabled state for flaring-related inputs
        self._toggle_monte_carlo_fields(self.monte_carlo_checkbox.isChecked())

        main_layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- Helpers ---------------------------------------------------------
    def _toggle_monte_carlo_fields(self, enabled: bool):
        """Enable/disable fields related to automatic flare detection."""
        for widget in (
            self.mc_trials_spin,
            self.mc_quantiles_lo_spin,
            self.mc_quantiles_hi_spin,
        ):
            widget.setEnabled(enabled)

    @staticmethod
    def _parse_optional_float(line_edit: QLineEdit, field_name: str):
        text = line_edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{field_name} must be a number or left blank.")

    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""
        try:
            params = {
                "max_lag": float(self.max_lag_spin.value()),
                "mc_trials": int(self.mc_trials_spin.value()),
                "mc_quantiles": (float(self.mc_quantiles_lo_spin.value()), float(self.mc_quantiles_hi_spin.value())),
                "mvt_rule": "mc" if self.monte_carlo_checkbox.isChecked() else "analytic",
                # "analytic_nsigma": float(self.analytic_nsigma_spin.value()) if not self}
            }
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = StructureFunctionDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None

class AnalysisRangeDialog(Dialog):
    """
    Dialog to configure temporal analysis parameters.
    """

    def __init__(self, title, label=None, message=None, parent=None, initial=None, width=400, height=0):
        super().__init__(parent=parent, width=width, height=height or 0)

        self.title = title
        self.label = label
        self.message = message

        self.defaults = {
            "analysis_range": (-10,10),
            "lag_method": "Cross-correlation",
        }
        if initial:
            self.defaults.update(initial)

        self._params = self.defaults.copy()

        self._build_ui(self.defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        # form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel(self.label, self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(2)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Message
        label_message = QLabel(self.message, self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(False)
        label_message.setFont(font)
        form.addRow(label_message)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Analysis range inputs (tmin "to" tmax)
        range_container = QWidget(self)
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Analysis Range (s):", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ar = defaults.get("analysis_range")
        self.tmin_input = QLineEdit(self)
        self.tmin_input.setText("" if ar is None else "%.2f" % ar[0])
        self.tmin_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tmin_input.setFixedWidth(65)

        label_to = QLabel("to", self)
        label_to.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.tmax_input = QLineEdit(self)
        self.tmax_input.setText("" if ar is None else "%.2f" % ar[1])
        self.tmax_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tmax_input.setFixedWidth(65)

        range_layout.addWidget(self.tmin_input)
        range_layout.addWidget(label_to)
        range_layout.addWidget(self.tmax_input)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(range_container)
        form.addRow(row_widget)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Sum detectors option
        self.sum_detectors_checkbox = QCheckBox(" Sum counts from all open detectors", self)
        self.sum_detectors_checkbox.setChecked(True)
        form.addRow(self.sum_detectors_checkbox)
        form.setAlignment(self.sum_detectors_checkbox, Qt.AlignmentFlag.AlignLeft)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(3)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        main_layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""

        try:
            params = {
                "sum_detectors": bool(self.sum_detectors_checkbox.isChecked()),
                "analysis_range": (float(self.tmin_input.text()), float(self.tmax_input.text()))
            }
                
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = T90OptionsDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None   

def test_FitOptionsDialog():
    from model_manager import ModelManager  # Replace with actual import path

    def callback(model, stat, weights, undet, batch=None):
        modelnames = [m.name for m in model]
        print(f"Models: {', '.join(modelnames)}")
        print(f"Statistic: {stat}")
        print(f"Weights: {weights}")
        print(f"Undetermined: {undet}")
        print(f"Batch: {batch}")

    app = QApplication([])
    dialog = FitOptionsDialog(ModelManager(), callback)
    dialog.exec()

def test_ParameterOptionsDialog():
    from model_manager import ModelManager  # Replace with actual import path

    manager = ModelManager()
    app = QApplication([])
    dialog = ParameterOptionsDialog("Band", manager)
    dialog.exec()

def test_OptionDialog():
    def cancel():
        print("Canceled")

    def fit(order):
        print(f"Fit order {order}")

    message = "Select Order of Background Polynomial"
    options = ["Cancel", "0", "1", "2", "3", "4"]
    commands = [
        cancel,
        lambda: fit(0),
        lambda: fit(1),
        lambda: fit(2),
        lambda: fit(3),
        lambda: fit(4),
    ]

    app = QApplication([])
    dialog = OptionDialog(
        options=options,
        commands=commands,
        title="Background",
        message=message,
    )
    dialog.exec()

def test_TextOptionDialog():
    def rebin(meth, val, opt):
        print(meth, val, opt)

    command = lambda method, value, option: rebin(method, value, option)

    title = "Combine by Factor"
    message = "Enter the desired rebinning factor"
    app = QApplication([])
    dialog = TextOptionDialog(
        title=title,
        width=275,
        height=150,
        message=message,
        default_value=2,
        button_label="Rebin",
        button_command=command,
    )
    # dialog.exec()

def test_ManualInputDialog():
    def man_select(x0, x1, y0, y1):
        print(x0, x1, y0, y1)

    title = "Manual Selection"
    message = "Manual Selection Input"
    command = lambda x0, x1, y0, y1: man_select(x0, x1, y0, y1)

    app = QApplication([])
    dialog = ManualInputDialog(
        command=command,
        title=title,
        message=message,
        xinput=True,
        yinput=True,
    )
    dialog.exec()

class TemporalRebinDialog(Dialog):
    """
    Dialog to configure the temporal resolution parameters for rebin_lightcurve().
    Returns a dict of parameters via .parameters or run_dialog().
    """

    def __init__(self, parent=None, initial=None, width=400, height=0):
        title = "Temporal Resolution"
        super().__init__(title=title, parent=parent, width=width, height=height or 0)

        defaults = {
            "resolution": 0.064,
        }

        if initial:
            defaults.update(initial)

        self._params = defaults.copy()

        self._build_ui(defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        # form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel('Temporal Binning Options', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # Message
        label_message = QLabel('Bin size must be an integer\nmultiple of the independent axis data', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        label_message.setFont(font)
        form.addRow(label_message)

        # max_lag
        spin_width = 75

        self.temporal_resolution_spin = QDoubleSpinBox()
        self.temporal_resolution_spin.setDecimals(3)
        self.temporal_resolution_spin.setRange(defaults['resolution'], 8.192)
        self.temporal_resolution_spin.setSingleStep(0.064)
        self.temporal_resolution_spin.setValue(float(defaults["resolution"]))
        self.temporal_resolution_spin.setFixedWidth(spin_width)
        form.addRow("Temporal Resolution (sec):", self.temporal_resolution_spin)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        main_layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""
        try:
            params = {
                "resolution": float(self.temporal_resolution_spin.value()),
            }
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = TemporalRebinDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None
    
class CombineByFactorDialog(Dialog):
    """
    Dialog to set the combine by factor parameter.
    Returns a dict of parameters via .params or run_dialog().
    """

    def __init__(self, parent=None, initial=None, width=400, height=0):
        title = "Rebinning Factor"
        super().__init__(title=title, parent=parent, width=width, height=height or 0)

        defaults = {
            "factor": 1,
        }

        if initial:
            defaults.update(initial)

        self._params = defaults.copy()

        self._build_ui(defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        # form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel('Combine by Factor', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # Message
        label_message = QLabel('Bin factor must be an integer value', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        label_message.setFont(font)
        form.addRow(label_message)

        # max_lag
        spin_width = 75

        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setDecimals(0)
        self.factor_spin.setRange(1, 100)
        self.factor_spin.setSingleStep(1)
        self.factor_spin.setValue(int(defaults["factor"]))
        self.factor_spin.setFixedWidth(spin_width)
        form.addRow("Combine by Factor:", self.factor_spin)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        main_layout.addLayout(form)

        # Buttons
        button_box_widget = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box_widget.accepted.connect(self.accept)
        button_box_widget.rejected.connect(self.reject)

        # Wrap the button box in a horizontal container and center it.
        container = QWidget(self)
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(button_box_widget)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rebind name so the later `main_layout.addWidget(button_box)` adds the centered container
        button_box = container

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""
        try:
            params = {
                "factor": int(self.factor_spin.value()),
            }
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = CombineByFactorDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None    
    
class RebinBySnrDialog(Dialog):
    """
    Dialog to set the rebin by SNR parameter.
    Returns a dict of parameters via .params or run_dialog().
    """

    def __init__(self, parent=None, initial=None, width=400, height=0):
        title = "Rebin by SNR"
        super().__init__(title=title, parent=parent, width=width, height=height or 0)

        defaults = {
            "snr": 1,
        }

        if initial:
            defaults.update(initial)

        self._params = defaults.copy()

        self._build_ui(defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        # form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel('Rebin by SNR', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # Message
        label_message = QLabel('Signal To Noise Ratio (SNR) in Sigma', self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        label_message.setFont(font)
        form.addRow(label_message)

        # max_lag
        spin_width = 75

        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setDecimals(0)
        self.snr_spin.setRange(1, 100)
        self.snr_spin.setSingleStep(1)
        self.snr_spin.setValue(int(defaults["snr"]))
        self.snr_spin.setFixedWidth(spin_width)
        form.addRow("Signal to Noise Ratio (sigma):", self.snr_spin)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        main_layout.addLayout(form)

        # Buttons
        button_box_widget = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box_widget.accepted.connect(self.accept)
        button_box_widget.rejected.connect(self.reject)

        # Wrap the button box in a horizontal container and center it.
        container = QWidget(self)
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(button_box_widget)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rebind name so the later `main_layout.addWidget(button_box)` adds the centered container
        button_box = container

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""
        try:
            params = {
                "snr": int(self.snr_spin.value()),
            }
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = RebinBySNRDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None        

# class LagAnalysisDialog(AnalysisRangeDialog):
#     """
#     Dialog to configure lag analysis parameters.

#     Inherits the AnalysisRangeDialog UI and adds:
#       - a max_lag numeric input
#       - two radio buttons to choose the lag method
#     """

#     def __init__(self, title=None, label=None, message=None, parent=None, initial=None, width=400, height=0):
#         title = "Lag Analysis Options"

#         # Defaults for this subclass (merged into AnalysisRangeDialog.defaults)
#         defaults = {
#             "max_lag": 10.0,
#             "lag_method": "Cross-correlation",
#         }
#         if initial:
#             defaults.update(initial)

#         # Pass the merged defaults as `initial` so AnalysisRangeDialog will include them
#         super().__init__(title=title, parent=parent, initial=defaults,
#                          width=width, height=height or 0)

#     def _build_ui(self, defaults):
#         # Let the base AnalysisRangeDialog build its UI first
#         super()._build_ui(defaults)

#         # Retrieve the main layout created by the parent build method
#         main_layout = self.layout()

#         # Add a small separator/title for lag-specific options
#         label = QLabel("Lag Analysis Options:", self)
#         label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         main_layout.addWidget(label)

#         # Max lag input
#         form_layout = QFormLayout()
#         spin_width = 120
#         self.max_lag_spin = QDoubleSpinBox()
#         self.max_lag_spin.setDecimals(2)
#         self.max_lag_spin.setRange(0.0, 1e6)
#         self.max_lag_spin.setSingleStep(0.1)
#         self.max_lag_spin.setValue(float(defaults.get("max_lag", 10.0)))
#         self.max_lag_spin.setFixedWidth(spin_width)
#         form_layout.addRow("Max Lag (s):", self.max_lag_spin)
#         main_layout.addLayout(form_layout)

#         # Radio buttons for lag method
#         self.lag_method_group = QButtonGroup(self)
#         method_container = QWidget(self)
#         method_layout = QHBoxLayout(method_container)
#         method_layout.setContentsMargins(0, 0, 0, 0)

#         methods = ["Cross-correlation", "Fourier"]
#         default_method = defaults.get("lag_method", methods[0])

#         for m in methods:
#             rb = QRadioButton(m, self)
#             if m == default_method:
#                 rb.setChecked(True)
#             self.lag_method_group.addButton(rb)
#             method_layout.addWidget(rb)

#         main_layout.addWidget(method_container)
#         main_layout.setAlignment(method_container, Qt.AlignmentFlag.AlignHCenter)

#     def accept(self):
#         """Collect and validate parameters (includes parent's fields) before closing."""
#         try:
#             params = {
#                 "sum_detectors": bool(self.sum_detectors_checkbox.isChecked()),
#                 "analysis_range": (
#                     float(self.tmin_input.text()),
#                     float(self.tmax_input.text()),
#                 ),
#                 "max_lag": float(self.max_lag_spin.value()),
#                 "lag_method": (
#                     self.lag_method_group.checkedButton().text()
#                     if self.lag_method_group.checkedButton() is not None
#                     else None
#                 ),
#             }
#         except ValueError as e:
#             msg = QMessageBox(self)
#             msg.setIcon(QMessageBox.Icon.Critical)
#             msg.setWindowTitle("Invalid input")
#             text = str(e)
#             msg.setText(text[:1].upper() + text[1:])
#             msg.setStandardButtons(QMessageBox.StandardButton.Ok)
#             msg.exec()
#             return

#         self._params = params
#         # Directly accept the QDialog to avoid calling the parent's accept which would
#         # re-parse/overwrite _params.
#         QDialog.accept(self)

class LagAnalysisDialog(Dialog):
    """
    Dialog to configure lag analysis parameters.
    Returns a dict of parameters via .params or run_dialog().
    """

    def __init__(self, title, label=None, message=None, parent=None, initial=None, width=400, height=0):
        super().__init__(parent=parent, width=width, height=height or 0)

        self.title = title
        self.label = label
        self.message = message or 'Select the temporal range and energy bands over which to\nperform the lag analysis.'

        self.defaults = {
            "analysis_range": (-10,10),
            "low_energy": (25,50),
            "high_energy": (100,300),
        }
        if initial:
            self.defaults.update(initial)

        self._params = self.defaults.copy()

        self._build_ui(self.defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Message
        label_message = QLabel(self.label, self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(2)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Message
        label_message = QLabel(self.message, self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(False)
        label_message.setFont(font)
        form.addRow(label_message)

        # small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Analysis range inputs (tmin "to" tmax)
        range_container = QWidget(self)
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Analysis Range (sec):", self)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ar = defaults.get("analysis_range")
        self.tmin_input = QLineEdit(self)
        self.tmin_input.setText("" if ar is None else "%.2f" % ar[0])
        self.tmin_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tmin_input.setFixedWidth(65)

        label_to = QLabel("to", self)
        label_to.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.tmax_input = QLineEdit(self)
        self.tmax_input.setText("" if ar is None else "%.2f" % ar[1])
        self.tmax_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tmax_input.setFixedWidth(65)

        range_layout.addWidget(self.tmin_input)
        range_layout.addWidget(label_to)
        range_layout.addWidget(self.tmax_input)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(range_container)
        form.addRow(row_widget)

        # Low energy range inputs 
        range_container = QWidget(self)
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Low Energy Range (keV):", self)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ar = defaults.get("low_energy")
        self.emin1_input = QLineEdit(self)
        self.emin1_input.setText("" if ar is None else "%.2f" % ar[0])
        self.emin1_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.emin1_input.setFixedWidth(65)

        label_to = QLabel("to", self)
        label_to.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.emax1_input = QLineEdit(self)
        self.emax1_input.setText("" if ar is None else "%.2f" % ar[1])
        self.emax1_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.emax1_input.setFixedWidth(65)

        range_layout.addWidget(self.emin1_input)
        range_layout.addWidget(label_to)
        range_layout.addWidget(self.emax1_input)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(range_container)
        form.addRow(row_widget)

        # High energy range inputs 
        range_container = QWidget(self)
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("High Energy Range (keV):", self)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ar = defaults.get("high_energy")
        self.emin2_input = QLineEdit(self)
        self.emin2_input.setText("" if ar is None else "%.2f" % ar[0])
        self.emin2_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.emin2_input.setFixedWidth(65)

        label_to = QLabel("to", self)
        label_to.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.emax2_input = QLineEdit(self)
        self.emax2_input.setText("" if ar is None else "%.2f" % ar[1])
        self.emax2_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.emax2_input.setFixedWidth(65)

        range_layout.addWidget(self.emin2_input)
        range_layout.addWidget(label_to)
        range_layout.addWidget(self.emax2_input)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(range_container)
        form.addRow(row_widget)


        # Max Lag inputs
        range_container = QWidget(self)
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Maximum Lag Value (sec):          ", self)
        # label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ar = defaults.get("low_energy")
        self.maxlag_input = QLineEdit(self)
        self.maxlag_input.setText("" if ar is None else "%.2f" % ar[0])
        # self.maxlag_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.maxlag_input.setFixedWidth(65)

        range_layout.addWidget(self.maxlag_input)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(range_container)
        form.addRow(row_widget)

        # # small vertical spacer
        # spacer = QWidget()
        # spacer.setFixedHeight(3)
        # spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # form.addRow(spacer)

        # # max_lag
        # spin_width = 75

        # self.temporal_resolution_spin = QDoubleSpinBox()
        # self.temporal_resolution_spin.setDecimals(3)
        # self.temporal_resolution_spin.setRange(0.0, 10.0)
        # self.temporal_resolution_spin.setSingleStep(0.064)
        # self.temporal_resolution_spin.setValue(float(defaults.get("max_lag", 1.0)))
        # self.temporal_resolution_spin.setFixedWidth(spin_width)

        # # Put label and spin into a horizontal container and right-align the row
        # row_widget = QWidget(self)
        # row_layout = QHBoxLayout(row_widget)
        # row_layout.setContentsMargins(0, 0, 0, 0)
        # row_layout.addStretch()  # push contents to the right
        # label = QLabel("Maximum Lag Value (sec):", self)
        # label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        # row_layout.addWidget(label)
        # row_layout.addWidget(self.temporal_resolution_spin)

        # form.addRow(row_widget)

        # # small vertical spacer
        # spacer = QWidget()
        # spacer.setFixedHeight(2)
        # spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # form.addRow(spacer)

        # # Message
        # label_message = QLabel('Lag Method:', self)
        # label_message.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # label_message.setSizePolicy(
        #     QSizePolicy.Policy.Expanding,
        #     QSizePolicy.Policy.Preferred,
        # )
        # form.addRow(label_message)

        # # Lag method radio buttons
        # self.lag_method_group = QButtonGroup(self)
        # method_container = QWidget(self)
        # method_layout = QHBoxLayout(method_container)
        # method_layout.setContentsMargins(0, 0, 0, 0)

        # methods = ["Cross-correlation", "Fourier"]
        # default_method = defaults.get("lag_method", methods[0])

        # for method in methods:
        #     rb = QRadioButton(method, self)
        #     if method == default_method:
        #         rb.setChecked(True)
        #     self.lag_method_group.addButton(rb)
        #     method_layout.addWidget(rb)

        # form.addRow(method_container)
        # form.setAlignment(method_container, Qt.AlignmentFlag.AlignCenter)

        # # small vertical spacer
        # spacer = QWidget()
        # spacer.setFixedHeight(5)
        # spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # form.addRow(spacer)

        # Sum detectors option
        self.subtract_bkgd_checkbox = QCheckBox(" Subtract background counts", self)
        self.subtract_bkgd_checkbox.setChecked(True)
        form.addRow(self.subtract_bkgd_checkbox)
        form.setAlignment(self.subtract_bkgd_checkbox, Qt.AlignmentFlag.AlignLeft)

        # Sum detectors option
        self.sum_detectors_checkbox = QCheckBox(" Sum counts from all open detectors", self)
        self.sum_detectors_checkbox.setChecked(True)
        form.addRow(self.sum_detectors_checkbox)
        form.setAlignment(self.sum_detectors_checkbox, Qt.AlignmentFlag.AlignLeft)

        # Add form to main layout
        main_layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""

        try:
            params = {
                "analysis_range": (float(self.tmin_input.text()), float(self.tmax_input.text())),
                "low_energy": (float(self.emin1_input.text()), float(self.emax1_input.text())),
                "high_energy": (float(self.emin2_input.text()), float(self.emax2_input.text())),
                "max_lag": float(self.maxlag_input.text()),
                "subtract_bkgd": bool(self.subtract_bkgd_checkbox.isChecked()),
                "sum_detectors": bool(self.sum_detectors_checkbox.isChecked()),

            }   
                
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = T90OptionsDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None   

class EnergeticsDialog(Dialog):
    """
    Dialog to configure energetics analysis parameters.
    Returns a dict of parameters via .params or run_dialog().
    """

    def __init__(self, title, label=None, message=None, parent=None, initial=None, width=400, height=0):
        super().__init__(parent=parent, width=width, height=height or 0)

        self.title = title
        self.label = label
        self.message = message 
        
        self.defaults = {
            "redshift": 0.0,
            "energy_range": (0,10000),
        }
        if initial:
            self.defaults.update(initial)

        self._params = self.defaults.copy()

        self._build_ui(self.defaults)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self, defaults):
        main_layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # Title
        label_message = QLabel(self.label, self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(True)
        label_message.setFont(font)
        form.addRow(label_message)

        # Small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(2)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Message
        label_message = QLabel(self.message, self)
        label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = label_message.font()
        font.setBold(False)
        label_message.setFont(font)
        form.addRow(label_message)

        # Small vertical spacer
        spacer = QWidget()
        spacer.setFixedHeight(5)
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        form.addRow(spacer)

        # Redshift input (label left, input right)
        label = QLabel("Redshift:", self)
        # label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        ar = defaults.get("redshift")
        self.redshift_input = QLineEdit(self)
        self.redshift_input.setText("" if ar is None else "%.2f" % ar)
        # self.redshift_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.redshift_input.setFixedWidth(65)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()                      # push the input to the far right
        row_layout.addWidget(self.redshift_input, 0) # add input at right edge
        form.addRow(row_widget)

        # Analysis range inputs (tmin "to" tmax)
        range_container = QWidget(self)
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Energy Range (keV):", self)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ar = defaults.get("energy_range")
        self.emin_input = QLineEdit(self)
        self.emin_input.setText("" if ar is None else "%.2f" % ar[0])
        self.emin_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.emin_input.setFixedWidth(65)

        label_to = QLabel("to", self)
        label_to.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.emax_input = QLineEdit(self)
        self.emax_input.setText("" if ar is None else "%.2f" % ar[1])
        self.emax_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.emax_input.setFixedWidth(65)

        range_layout.addWidget(self.emin_input)
        range_layout.addWidget(label_to)
        range_layout.addWidget(self.emax_input)

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(range_container)
        form.addRow(row_widget)

        # # Sum detectors option
        # self.subtract_bkgd_checkbox = QCheckBox(" Subtract background counts", self)
        # self.subtract_bkgd_checkbox.setChecked(True)
        # form.addRow(self.subtract_bkgd_checkbox)
        # form.setAlignment(self.subtract_bkgd_checkbox, Qt.AlignmentFlag.AlignLeft)

        # # Sum detectors option
        # self.sum_detectors_checkbox = QCheckBox(" Sum counts from all open detectors", self)
        # self.sum_detectors_checkbox.setChecked(True)
        # form.addRow(self.sum_detectors_checkbox)
        # form.setAlignment(self.sum_detectors_checkbox, Qt.AlignmentFlag.AlignLeft)

        # Add form to main layout
        main_layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.adjustSize()  # shrink-wrap to contents


    # ----- QDialog accept override -----------------------------------------
    def accept(self):
        """Collect and validate parameters before closing."""

        try:
            params = {
                "energy_range": (float(self.emin_input.text()), float(self.emax_input.text())),
                "redshift": float(self.redshift_input.text()),
            }   
                
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Invalid input")
            text = str(e)
            msg.setText(text[:1].upper() + text[1:])
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self._params = params
        super().accept()

    # ----- Public API ------------------------------------------------------
    @property
    def params(self):
        """Return the last accepted parameter set as a dict."""
        return self._params

    @classmethod
    def run_dialog(cls, parent=None, initial=None):
        """
        Convenience helper:
            params = T90OptionsDialog.run_dialog(parent, initial)
        Returns dict if accepted, or None if cancelled.
        """
        dlg = cls(parent=parent, initial=initial)
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            return dlg.params
        return None   
