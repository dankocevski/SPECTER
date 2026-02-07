import os

from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
    QMenuBar, QMenu, QListWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt


class FileManager(QMainWindow):
    """Primary GSPEC application class."""

    def __init__(self, grb):
        super().__init__()

        # self.name = f"GSpec v{gspec.__version__}"
        self.grb = grb
        self.name = grb.name if hasattr(grb, 'name') else "GSpec"
        self.files = self.grb.data_filenames
        self._open_windows: dict[str, QWidget] = {}
        self.filenames: list[str] = []

        self._export_files = None
        self._last_fit_batch = None
        self._current_fit_batch = None
        self.logger = None
        self.scat = None
        self.xspec_version: str | None = None
        self.temp_dirs: list[str] = []

        # valid files for GSpec
        self.datafiletypes = "PHA Files (*.pha);;FITS Files (*.fit *.fits)"
        self.lookupfiletypes = "GSpec Lookup (*.json)"

        self.build_gui()

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------
    def build_gui(self) -> None:
        # Main window
        self.setWindowTitle(self.name)
        self.setMinimumSize(450, 236)
        self.resize(600, 300)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # List widget
        self.list_widget = QListWidget(self)

        # Populate with initial files
        self.list_widget.addItems(self.files)

        # Buttons
        load_button = QPushButton("Load", self)
        display_button = QPushButton("Display", self)
        hide_button = QPushButton("Hide", self)
        delete_button = QPushButton("Delete", self)

        load_button.clicked.connect(self.load_file)
        display_button.clicked.connect(self.display)
        hide_button.clicked.connect(self.hide)
        delete_button.clicked.connect(self.delete_file)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(load_button)
        button_layout.addWidget(display_button)
        button_layout.addWidget(hide_button)
        button_layout.addWidget(delete_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.list_widget)
        main_layout.addLayout(button_layout)

        central_widget.setLayout(main_layout)

        # Menu bar
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # File menu
        file_menu = QMenu("File", self)
        file_menu.addAction("Load Data...", self.load_file)
        file_menu.addAction("Load From Lookup...", self.load_lookup)
        menubar.addMenu(file_menu)

        # Options menu
        options_menu = QMenu("Options", self)
        options_menu.addAction("Display", self.display)
        options_menu.addAction("Hide", self.hide)
        options_menu.addAction("Delete", self.delete_file)
        menubar.addMenu(options_menu)

        # Export menu (currently stubs)
        export_menu = QMenu("Export", self)
        export_menu.addAction(
            "Selected Data to XSPEC PHAI Format...", self.prepare_export_to_xspec_pha1
        )
        export_menu.addAction(
            "Selected Data to XSPEC PHAII Format...", self.prepare_export_to_xspec_pha2
        )
        export_menu.addAction("Fit Results to FITS File...", self.export_fit_results)
        export_menu.addAction("Fit Log to Text File...", self.export_fit_log)
        menubar.addMenu(export_menu)

        # Windows menu
        windows_menu = QMenu("Windows", self)
        windows_menu.addAction("Cascade Windows", self.cascade_windows)
        windows_menu.addAction("Tile Windows", self.tile_windows)
        menubar.addMenu(windows_menu)

    # ------------------------------------------------------------------
    # XSPEC / window registration helpers
    # ------------------------------------------------------------------
    def test_xspec(self) -> None:
        """Test that XSPEC is available and meets the minimum version."""
        try:
            test_xspec = Xspec()
            output_lines = test_xspec.do_command("version")
            if not output_lines:
                raise RuntimeError("Could not get XSPEC version output.")

            v = output_lines[-1].split(":")[-1].strip()
            major, minor, build = v.split(".")
            self.xspec_version = v

            if (int(major) < 12) or (int(major) == 12 and int(minor) < 10):
                msg = (
                    f"You are using XSPEC version {v}.\n"
                    "Some functionality will not be available for "
                    "versions < 12.10. Please consult the GSPEC "
                    "documentation for more information."
                )
                QMessageBox.warning(self, "Old XSPEC Version", msg)

                # remove models that depend on v12.10
                self.models.remove_model("Band (Epeak)")
                self.models.remove_model("Comptonized")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "XSPEC not found",
                "GSpec cannot find XSPEC.\nMake sure it is in your PATH.\n\n"
                f"Details: {exc}",
            )

    def register_window(self, name: str, window: QWidget) -> None:
        """Track open child windows by name."""
        self._open_windows[name] = window

    def unregister_window(self, name: str) -> None:
        """Remove a window from the registry."""
        if name in self._open_windows:
            del self._open_windows[name]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_file(self) -> None:
        selected_files, _ = QFileDialog.getOpenFileNames(
            self, "Load Data File", "", self.datafiletypes
        )
        if not selected_files:
            return

        for filename in selected_files:
            if filename in self.filenames:
                QMessageBox.information(
                    self,
                    "File Already Loaded",
                    f"{os.path.basename(filename)} is already loaded.",
                )
                continue

            print(f"Loading {filename}")
            self.filenames.append(filename)
            basename = os.path.basename(filename)
            self.list_widget.addItem(basename)
            self.grb.add_data_file(filename)

            # Select and display the newly added file
            row = self.list_widget.count() - 1
            self.list_widget.setCurrentRow(row)
            self.display(index=row)

    def load_lookup(self) -> None:
        pass
    
    # def load_lookup(self) -> None:
    #     selected_filename, _ = QFileDialog.getOpenFileName(
    #         self, "Load Lookup File", "", self.lookupfiletypes
    #     )
    #     if not selected_filename:
    #         return

    #     # Load lookup, register with Gspec, and launch window for each detector
    #     directory = os.path.dirname(selected_filename)
    #     lu = LookupFile.read_from(selected_filename)

    #     for dataname in lu.files():
    #         datafilename = os.path.join(directory, dataname)
    #         try:
    #             self.gspec.add_data_file(datafilename)
    #             self.gspec.load_gspec_lookup(dataname, lu)
    #         except Exception as exc:
    #             print(f"Failed to load {datafilename} from lookup: {exc}")
    #             continue

    #         if datafilename not in self.filenames:
    #             self.filenames.append(datafilename)
    #             self.list_widget.addItem(dataname)

    #     # Optionally display the first item from the lookup
    #     if self.list_widget.count() > 0:
    #         self.list_widget.setCurrentRow(self.list_widget.count() - 1)
    #         self.display()

    #     QMessageBox.information(
    #         self,
    #         "Lookup Loaded",
    #         f"Loaded lookup file:\n{selected_filename}",
    #     )

    # ------------------------------------------------------------------
    # Display / hide / delete
    # ------------------------------------------------------------------
    def display(self, index: int | None = None) -> None:
        if index is None:
            index = self.list_widget.currentRow()

        if index is None or index < 0 or index >= len(self.filenames):
            QMessageBox.warning(
                self, "No Selection", "Please select an item to display."
            )
            return

        selected_filename = self.filenames[index]
        datename = os.path.basename(selected_filename)

        print(f"Displaying {selected_filename}")

        # If we someday want to re-use windows:
        # if datename in self._open_windows:
        #     self._open_windows[datename].raise_()
        #     self._open_windows[datename].activateWindow()
        #     return

        viewer = DataViewer(selected_filename, gspec_root=self)
        viewer.show()

        self.register_window(datename, viewer)

    def hide(self, index: int | None = None) -> None:
        if index is None:
            index = self.list_widget.currentRow()

        if index is None or index < 0 or index >= len(self.filenames):
            QMessageBox.warning(
                self, "No Selection", "Please select an item to hide."
            )
            return

        selected_filename = self.filenames[index]
        datename = os.path.basename(selected_filename)

        print(f"Hiding {selected_filename}")

        try:
            window = self._open_windows[datename]
        except KeyError:
            print("Window is already hidden")
            return

        # Unregister by the key we used to store it
        self.unregister_window(datename)

        # Expect DataViewer to have an on_window_close() method; fall back to close()
        if hasattr(window, "on_window_close"):
            window.on_window_close()
        else:
            window.close()

    def delete_file(self, index: int | None = None) -> None:
        if index is None:
            index = self.list_widget.currentRow()

        if index is None or index < 0 or index >= len(self.filenames):
            QMessageBox.warning(
                self, "No Selection", "Please select an item to delete."
            )
            return

        # Hide (if visible) before deleting
        self.hide(index)

        item = self.list_widget.takeItem(index)
        del item  # let Qt clean it up

        selected_filename = self.filenames.pop(index)
        datename = os.path.basename(selected_filename)

        # Remove from gspec if it behaves like a dict keyed by datename
        try:
            del self.gspec[datename]
        except Exception:
            # If GspecManager doesn't support this, ignore
            pass

    # ------------------------------------------------------------------
    # Stubs / placeholders for future functionality
    # ------------------------------------------------------------------
    def cascade_windows(self) -> None:
        QMessageBox.information(
            self, "Cascade Windows", "Cascade Windows feature is not yet implemented."
        )

    def tile_windows(self) -> None:
        QMessageBox.information(
            self, "Tile Windows", "Tile Windows feature is not yet implemented."
        )

    def prepare_export_to_xspec_pha1(self) -> None:
        QMessageBox.information(
            self,
            "Export",
            "Export to XSPEC PHAI Format feature is not yet implemented.",
        )

    def prepare_export_to_xspec_pha2(self) -> None:
        QMessageBox.information(
            self,
            "Export",
            "Export to XSPEC PHAII Format feature is not yet implemented.",
        )

    def export_fit_results(self) -> None:
        QMessageBox.information(
            self, "Export", "Export Fit Results feature is not yet implemented."
        )

    def export_fit_log(self) -> None:
        QMessageBox.information(
            self, "Export", "Export Fit Log feature is not yet implemented."
        )


def run_app() -> None:
    """Convenience function to launch the application."""
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    main_window = Application()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    run_app()
