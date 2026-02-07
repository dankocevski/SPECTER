from matplotlib.widgets import RectangleSelector
import matplotlib.pylab as plt
import numpy as np

class RangeSelector:
    """Supports zoom via toolbar or 'z', and multi-interval selection: (min,max) pairs.
    Click outside the axes to finish. Press 'u' to undo last interval, 'h' for home."""
    def __init__(self, fig=None, ax=None, canvas=None, toolbar=None, src_selection=False,
                 finished_callback=None):

        if canvas is not None:
            self.canvas = canvas
            self.fig = canvas.fig
            self.ax = canvas.axes
            self.toolbar = canvas.toolbar
            self.toolmanager=None

        else:
            self.fig = fig
            self.ax = ax
            self.canvas = self.fig.canvas
            # toolbar (classic) and toolmanager (newer)
            self.toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)
            self.toolmanager = getattr(getattr(fig.canvas, "manager", None), "toolmanager", None)

        self.finished_callback = finished_callback
        self.src_selection = src_selection

        # zoom state & fallback rectangle zoom
        self.zoom_active = False
        self.rect_zoom = None

        # state for intervals
        self.pending = None              # first click in a pair
        self.ranges = []                 # [(tmin, tmax), ...]
        self.spans = []                  # axvspan artists
        self.vlines = []                 # vertical guide lines (all)
        self._last_pair_vlines = []      # vlines for current pair (to undo cleanly)

        # original limits for manual home
        self._orig_xlim = self.ax.get_xlim()
        self._orig_ylim = self.ax.get_ylim()

        # # events
        # self.cid_click   = self.fig.canvas.mpl_connect('button_press_event',   self._on_click)
        # self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        # self.cid_key     = self.fig.canvas.mpl_connect('key_press_event',      self._on_key)
        # self.cid_draw    = self.fig.canvas.mpl_connect('draw_event',           self._on_draw)

        # events
        self.cid_click   = self.canvas.mpl_connect('button_press_event',   self._on_click)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_key     = self.canvas.mpl_connect('key_press_event',      self._on_key)
        self.cid_draw    = self.canvas.mpl_connect('draw_event',           self._on_draw)

        # ToolManager signal (if available)
        self._tm_cid = None
        if self.toolmanager is not None:
            try:
                self._tm_cid = self.toolmanager.messengers.connect('tool_trigger_zoom', self._on_tool_trigger_zoom)
            except Exception:
                self._tm_cid = None

        self._set_title_instructions()
        self._sync_zoom_state(update_title=False)
        self.fig.canvas.draw_idle()

    # ---------- Zoom helpers ----------
    def _start_rect_zoom(self):
        if self.rect_zoom is not None:
            return
        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return
            self.ax.set_xlim(min(x0, x1), max(x0, x1))
            self.ax.set_ylim(min(y0, y1), max(y0, y1))
            self.fig.canvas.draw_idle()
        self.rect_zoom = RectangleSelector(
            self.ax, onselect,
            useblit=True, button=[1],
            minspanx=0, minspany=0,
            interactive=False, drag_from_anywhere=False
        )

    def _stop_rect_zoom(self):
        if self.rect_zoom is not None:
            self.rect_zoom.set_active(False)
            self.rect_zoom.disconnect_events()
            self.rect_zoom = None

    def _toggle_zoom_key(self):
        if self.toolbar is not None and hasattr(self.toolbar, "zoom"):
            self.toolbar.zoom()
        else:
            if not self.zoom_active:
                self._start_rect_zoom()
            else:
                self._stop_rect_zoom()
        self._sync_zoom_state()

    def _home_view(self):
        if self.toolbar is not None and hasattr(self.toolbar, "home"):
            self.toolbar.home()
        else:
            self.ax.set_xlim(*self._orig_xlim)
            self.ax.set_ylim(*self._orig_ylim)
        self.fig.canvas.draw_idle()

    def _toolbar_zoom_mode(self):
        return getattr(self.toolbar, "mode", "") if self.toolbar is not None else ""

    def _sync_zoom_state(self, update_title=True):
        mode = self._toolbar_zoom_mode()
        active = (mode == "zoom rect")
        if self.toolbar is None and self.rect_zoom is not None:
            active = True
        changed = (active != self.zoom_active)
        self.zoom_active = active
        if update_title and changed:
            self._set_title_instructions()
            self.fig.canvas.draw_idle()

    # ---------- Titles / instructions ----------
    def _set_title_instructions(self):
        if self.zoom_active:
            text = "Zoom ON. Use toolbar zoom or press 'z' (drag to zoom). Press 'z' again to exit zoom."
        else:
            if self.src_selection == True:
                text = ("Select source range: click min, then max to add an interval. "
                        "Press 'u' to undo last interval. "
                        "Click outside axes to finish. ")
            else:
                text = ("Select background ranges: click min, then max to add an interval. "
                        "Press 'u' to undo last interval. "
                        "Click outside axes to finish. ")
        self.ax.set_title(text, fontsize=10)

    # ---------- Event handlers ----------
    def _on_tool_trigger_zoom(self, sender, tool, data):
        self._sync_zoom_state()

    def _on_draw(self, event):
        self._sync_zoom_state(update_title=False)

    def _on_key(self, event):
        if event.key == 'z':
            self._toggle_zoom_key()
            self._set_title_instructions()
            self.fig.canvas.draw_idle()
        elif event.key == 'h':
            self._home_view()
        elif event.key in ('u', 'backspace'):
            self._undo_last_interval()

    def _on_release(self, event):
        self._sync_zoom_state(update_title=False)

    def _on_click(self, event):
        # finish this detector if clicked outside axes
        if event.inaxes is None:
            
            if len(self.ranges[-1]) == 1:
                print("Undoing last interval! 1")
                self._undo_last_interval()

            # mark done / notify parent instead of just closing the fig
            if self.finished_callback is not None:

                # remove the guide lines and shaded spans
                self._clear_artists()

                # Reset the titel
                self.canvas.axes.set_title("")
                self.canvas.draw_idle()

                self.finished_callback(self.get_ranges())

                return  
            
            else:

                plt.close(self.fig)
  

        if self.src_selection == True and len(self.spans) > 1:
            self._undo_last_interval()
        
        self._sync_zoom_state(update_title=False)

        # ignore selection while zoom is active, wrong button, or no xdata
        if self.zoom_active or event.button  != 1 or event.xdata is None:
            return

        # map the click to the nearest plotted bin edge (if available)
        x_click = float(event.xdata)
        try:
            # collect x-data from all Line2D artists on the axis
            x_arrays = [np.asarray(line.get_xdata()) for line in self.ax.get_lines() if len(line.get_xdata()) > 1]
            if x_arrays:
                # concatenate and get unique sorted edges (step plots often repeat edges)
                edges = np.unique(np.concatenate(x_arrays))
                # find nearest edge to the clicked x
                idx = np.argmin(np.abs(edges - x_click))
                x = float(edges[idx])
            else:
                x = x_click
        except Exception:
            # fallback to raw click location if anything goes wrong
            x = x_click

        # first click in a pair
        if self.pending is None:
            self.pending = x
            v = self.ax.axvline(x, linestyle='--', linewidth=1)
            self.vlines.append(v)
            self._last_pair_vlines = [v]
            self.fig.canvas.draw_idle()
            return

        # second click completes the pair
        x0, x1 = sorted((self.pending, x))
        v = self.ax.axvline(x, linestyle='--', linewidth=1)
        self.vlines.append(v)
        self._last_pair_vlines.append(v)

        span = self.ax.axvspan(x0, x1, alpha=0.15)
        self.spans.append(span)
        self.ranges.append((x0, x1))

        # reset pending for next pair
        self.pending = None

        # update title
        if self.src_selection == True:
            self.ax.set_title(
                "Click outside axes to finish. Press 'u' to undo last. ",
                fontsize=10
            )
        else:
            self.ax.set_title(
                "Add more intervals or click outside axes to finish. Press 'u' to undo last. ",
                fontsize=10
            )
        
        self.fig.canvas.draw_idle()


    def _undo_last_interval(self):
        # only if at least one interval exists and we're not mid-pair
        if not self.ranges:
            return
        # remove last span
        span = self.spans.pop()
        try:
            span.remove()
        except Exception:
            pass
        # remove the two vlines from the last pair
        for _ in range(2):
            if self.vlines:
                ln = self.vlines.pop()
                try:
                    ln.remove()
                except Exception:
                    pass
        # remove from ranges
        last = self.ranges.pop()
        # clear last-pair vlines
        self._last_pair_vlines = []
        # update title

        if self.src_selection == True:
            self.ax.set_title(
                "Click outside axes to finish. Press 'u' to undo last. ",
                fontsize=10
            )
        else:
            self.ax.set_title(
                "Add more intervals or click outside axes to finish. Press 'u' to undo last. ",
                fontsize=10
            )

        self.fig.canvas.draw_idle()

    def get_ranges(self):
        # If user clicked outside with a dangling single click, ignore it
        if self.pending is not None:
            self.pending = None
        return list(self.ranges)

    def disconnect(self):
        for cid in (self.cid_click, self.cid_release, self.cid_key, self.cid_draw):
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        if self.toolmanager is not None and self._tm_cid is not None:
            try:
                self.toolmanager.messengers.disconnect(self._tm_cid)
            except Exception:
                pass
        self._stop_rect_zoom()

    def _clear_artists(self):
        """Remove all span and vline artists from the axes."""
        for span in self.spans:
            try:
                span.remove()
            except Exception:
                pass

        for ln in self.vlines:
            try:
                ln.remove()
            except Exception:
                pass

        self.spans.clear()
        self.vlines.clear()
        self._last_pair_vlines = []

        # clear any pending click
        self.pending = None

        # redraw without overlays
        self.fig.canvas.draw_idle()        