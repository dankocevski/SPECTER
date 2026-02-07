import numpy as np
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
import warnings
import sys, time

def _infer_edges_from_centers(tcenters, dt=None):
    """Infer bin edges from centers. If dt is None, assume (approximately) uniform bins."""
    tcenters = np.asarray(tcenters, float)
    if tcenters.size < 2 and dt is None:
        raise ValueError("Need at least 2 centers or specify dt to infer edges.")
    if dt is None:
        # use neighbor spacings; extrapolate at ends
        dt_left  = np.diff(tcenters, prepend=tcenters[0] - (tcenters[1]-tcenters[0]))
        dt_right = np.diff(tcenters, append = tcenters[-1] + (tcenters[-1]-tcenters[-2]))
        # average of left/right spacings
        widths = 0.5*(dt_left + dt_right)
    else:
        widths = np.full_like(tcenters, float(dt))
    edges = np.empty(tcenters.size + 1, float)
    edges[:-1] = tcenters - 0.5*widths
    edges[-1]  = tcenters[-1] + 0.5*widths[-1]
    return edges

def _blocks_from_events(event_times, p0=0.05):
    """Bayesian blocks for event data; returns edges and per-block rates."""
    event_times = np.asarray(event_times, float)
    # edges = bayesian_blocks(event_times, fitness='events', p0=p0)
    edges = bayesian_blocks(event_times, p0=p0)

    # rate per block = counts / duration, using event data
    counts = np.histogram(event_times, bins=edges)[0]
    durations = np.diff(edges)
    rates = counts / durations
    return edges, rates, counts

def _blocks_from_binned(edges, counts, p0=0.05, max_expand=900_000):
    """
    Approximate Poisson-binned BB by expanding each bin's counts into events at the bin center.
    This is exact for uniform-in-bin arrival times and is often good enough for GRB light curves.
    If total counts are huge, warn and thin deterministically.
    """
    edges  = np.asarray(edges, float)
    counts = np.asarray(counts, float)
    if edges.size != counts.size + 1:
        raise ValueError("For binned input, len(edges) must be len(counts)+1.")

    centers = 0.5*(edges[:-1] + edges[1:])
    # Expand counts -> event times at bin centers
    total = int(np.round(counts.sum()))
    if total <= 0:
        return np.array([edges[0], edges[-1]]), np.array([0.0]), np.array([0])

    if total > max_expand:
        warnings.warn(
            f"Total counts ({total}) exceed max_expand={max_expand}. "
            "Thinning events for speed; results should still be close."
        )
        # proportional thinning to cap at max_expand
        factor = max_expand / total
        expanded = []
        for c, tc in zip(counts, centers):
            n = int(np.round(c))
            if n > 0:
                expanded.extend([tc]*n)
        event_times = np.array(expanded, float)
    else:
        # exact expansion
        expanded = []
        for c, tc in zip(counts, centers):
            n = int(np.round(c))
            if n > 0:
                expanded.extend([tc]*n)
        event_times = np.array(expanded, float)

    edges_bb = bayesian_blocks(event_times, fitness='events', p0=p0)
    # Compute rates & counts on the original data for accurate normalization
    counts_bb = np.histogram(centers, bins=edges_bb, weights=counts)[0]
    durations = np.diff(edges_bb)
    rates_bb = counts_bb / durations
    return edges_bb, rates_bb, counts_bb

def get_bayesian_blocks(
    *,
    event_times=None,
    t=None,
    counts=None,
    dt=None,
    edges=None,
    p0=0.05,
    background=None,
    background_rate=None,
    emin=None,
    emax=None,
    bg_method="ends",           # "low_quartile" | "ends"
    bg_percentage=0.25,         # if bg_method="low_quartile", use this percentile
    ends_frac=0.1,              # fraction of time at each end for bg if bg_method="ends"
    sigma_thresh=None,          # if set, use (rate - bg) > n_sigma*sqrt(bg/dt) per block
    factor_thresh=1.5,          # fallback threshold: rate > factor_thresh * bg
    min_block_duration=0.064,   # discard blocks shorter than this (sec)
    merge_gap=0.25,             # merge above-threshold blocks separated by <= merge_gap (sec)
):
    """
    Detect emission episodes in a GRB light curve using Bayesian Blocks.

    Provide EITHER:
      - event_times (unbinned photon times, seconds), OR
      - binned counts via (t & counts) or (edges & counts):
         * If you pass 't' with same length as 'counts', it's treated as bin centers (dt optional).
         * If you pass 'edges', it must be len(counts)+1.

    Returns
    -------
    result : dict with keys
        'block_edges'      : np.ndarray, block edge times
        'block_rates'      : np.ndarray, rate [counts/sec] per block
        'block_counts'     : np.ndarray, counts per block
        'background_rate'  : float
        'episodes'         : list of dicts: [{'t_start','t_stop','peak_rate','counts'}]
    """
    # 1) Build blocks
    if event_times is not None:
        edges_bb, rates_bb, counts_bb = _blocks_from_events(event_times, p0=p0)
        tmin, tmax = event_times.min(), event_times.max()
    else:
        if edges is None:
            if t is None or counts is None:
                raise ValueError("Provide event_times OR binned data via (t & counts) or (edges & counts).")
            edges = _infer_edges_from_centers(t, dt=dt)
        edges_bb, rates_bb, counts_bb = _blocks_from_binned(edges, counts, p0=p0)
        tmin, tmax = edges[0], edges[-1]

        # Print Bayesian block edges as two columns: lower_edge | upper_edge
        lower_edges = edges_bb[:-1]
        upper_edges = edges_bb[1:]
        print("\nBayesian Blocks Boundries:")
        print("+----------------------+----------------------+")
        print("| lower edge (s)       | upper edge (s)       |")
        print("+----------------------+----------------------+")
        for lo, hi in zip(lower_edges, upper_edges):
            print(f"| {float(lo):20.6f} | {float(hi):20.6f} |")
        print("+----------------------+----------------------+")

    # Optional: prune ultra-short blocks (noise protection)
    if min_block_duration > 0:
        keep = np.diff(edges_bb) >= min_block_duration
        # ensure at least edges remain consistent (keep both edges around kept blocks)
        if not np.all(keep):
            new_edges = [edges_bb[0]]
            new_rates = []
            new_counts = []
            for i, k in enumerate(keep):
                if k:
                    new_edges.append(edges_bb[i+1])
                    new_rates.append(rates_bb[i])
                    new_counts.append(counts_bb[i])
            edges_bb = np.array(new_edges, float)
            rates_bb = np.array(new_rates, float)
            counts_bb = np.array(new_counts, float)

    # 2a) Estimate the background 
    if background is None:

        if bg_method == "ends":
            print("Using 'ends' background estimation")
            # estimate from ends_frac at each end of (tmin, tmax)
            span = tmax - tmin
            left, right = tmin + ends_frac*span, tmax - ends_frac*span
            # weight blocks by overlap with end regions
            def overlap(a0, a1, b0, b1):
                return max(0.0, min(a1, b1) - max(a0, b0))
            bg_counts = 0.0
            bg_time = 0.0
            for e0, e1, c in zip(edges_bb[:-1], edges_bb[1:], counts_bb):
                dt_block = e1 - e0
                # left segment
                ol = overlap(e0, e1, tmin, left)
                # right segment
                orr = overlap(e0, e1, right, tmax)
                olr = ol + orr
                if olr > 0:
                    frac = olr / dt_block
                    bg_counts += c * frac
                    bg_time   += olr
            background_rate = (bg_counts / bg_time) if bg_time > 0 else np.percentile(rates_bb, 25.0)
        elif bg_method == "median":
            print
            background_rate = np.median(rates_bb)
        else:
            print("Using 'low_quartile' background estimation")
            # default: use lower quartile of block rates
            background_rate = np.percentile(rates_bb, 0.25)

    # 2b) Use the supplied background 
    if background is not None:

        # background_rates_per_bin = background.integrate_energy()[0].rates
        # background_counts_per_bin = background.integrate_energy()[0].counts.squeeze()
        # tcenter = background.time_centroids()[0]
        # tstart = background.tstart()[0]
        # tstop = background.tstop()[0]

        background_rates_per_bin = background.integrate_energy(emin=emin, emax=emax).rates
        background_counts_per_bin =background.integrate_energy(emin=emin, emax=emax).counts.squeeze()
        
        tcenter = background.time_centroids
        tstart = background.tstart
        tstop = background.tstop

        # For each Bayesian block, find which bins fall within its edges
        avg_bg_rate_per_block = []
        for i in range(len(edges_bb) - 1):
            t0, t1 = edges_bb[i], edges_bb[i+1]
            in_block = (tcenter >= t0) & (tcenter < t1)
            
            if np.any(in_block):
                # Duration-weighted average background rate for the block
                durations = tstop[in_block] - tstart[in_block]
                weights = durations / durations.sum()
                avg_rate = np.sum(background_rates_per_bin[in_block] * weights)
            else:
                avg_rate = np.nan  # No bins overlap this block
            avg_bg_rate_per_block.append(avg_rate)

        avg_bg_rate_per_block = np.array(avg_bg_rate_per_block)
        background_rate = avg_bg_rate_per_block

        # For each Bayesian block, find which bins fall within its edges
        bg_counts_per_block = []
        for i in range(len(edges_bb) - 1):
            t0, t1 = edges_bb[i], edges_bb[i+1]
            in_block = (tcenter >= t0) & (tcenter < t1)
            
            if np.any(in_block):
                # Total background counts in this Bayesian block
                total_bg_counts = np.sum(background_counts_per_bin[in_block])
            else:
                total_bg_counts = np.nan
            bg_counts_per_block.append(total_bg_counts)

        bg_counts_per_block = np.array(bg_counts_per_block)
        background_counts = bg_counts_per_block        

    # 3) Threshold blocks, merge into episodes
    episodes = []
    above = np.zeros_like(rates_bb, dtype=bool)
    if sigma_thresh is not None:

        # For each block, require excess above bg by N sigma.
        # print(f"Using sigma_thresh={sigma_thresh} for episode detection")
        
        # sigma on rate ~ sqrt(bg * duration)/duration = sqrt(bg/duration)
        durations = np.diff(edges_bb)
        sigma_rate = np.sqrt(np.maximum(background_rate, 1e-12) / np.maximum(durations, 1e-12))
        above = (rates_bb - background_rate) > (sigma_thresh * sigma_rate)

        # # Poisson uncertainty on expected background counts
        # sigma_counts = np.sqrt(np.maximum(counts_bb, 1e-12))
        # for i in range(len(counts_bb)):
        #     print(f"Block {i}: background_counts={background_counts[i]}, sigma_counts={sigma_counts[i]}, counts_bb={counts_bb[i]}, difference={counts_bb[i] - background_counts[i]}")

        # # Identify blocks that are N-sigma above background
        # above = (counts_bb - background_counts) > (sigma_thresh * sigma_counts)

    else:
        above = rates_bb > (factor_thresh * background_rate)

    # Merge adjacent "above" blocks; allow small gaps <= merge_gap
    i = 0
    while i < above.size:
        if not above[i]:
            i += 1
            continue

        # start a new episode
        t_start = edges_bb[i]
        peak_rate = rates_bb[i]
        total_counts = counts_bb[i]
        t_stop = edges_bb[i+1]
        j = i + 1

        turning_point = False
        while j < above.size:
            gap = edges_bb[j] - t_stop
            # if above[j] and gap < merge_gap:
            if above[j]:
                # Check bounds before accessing rates_bb[j+1]
                if j > 0 and j+1 < rates_bb.size:
                    if rates_bb[j] < rates_bb[j-1] and rates_bb[j] < rates_bb[j+1]:
                        turning_point = True

                # extend episode (include gap if tiny)
                t_stop = edges_bb[j+1]
                peak_rate = max(peak_rate, rates_bb[j])
                total_counts += counts_bb[j]
                j += 1

                if turning_point:
                    break
            else:
                break
            
        episodes.append(dict(t_start=float(t_start), t_stop=float(t_stop),
                             peak_rate=float(peak_rate), counts=float(total_counts)))
        i = j   

    if background is not None:
        background_rate = np.median(avg_bg_rate_per_block[~np.isnan(avg_bg_rate_per_block)])

    return dict(
        block_edges=edges_bb,
        block_rates=rates_bb,
        block_counts=counts_bb,
        background_rate=float(background_rate),
        episodes=episodes
    )

# -----------------------
# Example usage patterns:
# 1) Unbinned photon times (seconds):
# result = bayesian_blocks(event_times=arrivals, p0=0.05, sigma_thresh=3, merge_gap=0.2)
# plot_grb_bb(result, event_times=arrivals, title="Events → Blocks")
#
# 2) Binned counts with centers + counts:
# result = bayesian_blocks_grb(t=bin_centers, counts=bin_counts, dt=bin_width,
#                              p0=0.05, factor_thresh=2.0, merge_gap=0.2)
# plot_grb_bb(result, t=bin_centers, counts=bin_counts, title="Binned → Blocks")
#
# 3) Binned counts with edges + counts:
# result = bayesian_blocks_grb(edges=bin_edges, counts=bin_counts, p0=0.05,
#                              sigma_thresh=3.0, merge_gap=0.3)
# plot_grb_bb(result, edges=bin_edges, counts=bin_counts, title="Binned → Blocks")
#
# The output 'episodes' is a list of {t_start, t_stop, peak_rate, counts}.
# -----------------------


def plot_bayesian_blocks(
    result,
    *,
    event_times=None,
    t=None,
    counts=None,
    edges=None,
    title="GRB Light Curve with Bayesian Blocks",
    background=None,
    show_background=True,
    show_blocks=True,
    show_episodes=True,
    color_episodes=True,
    bar_alpha=0.35,
    episode_alpha=0.2,
    block_linewidth=2.0,
    figsize=(10, 4),
    show_episode_dividers=True,             
    episode_divider_mode="boundaries",      # "boundaries" | "midgap" 
    episode_divider_kwargs=None,             # dict of line kwargs
    xmin=None,
    xmax=None,
    detector_label=None,
    energy_label=None,
):
    """
    Plot the light curve (events or binned counts), Bayesian-Blocks step model,
    and shaded emission episodes.

    Parameters
    ----------
    result : dict
        Output from get_bayesian_blocks(...). Must contain:
        'block_edges', 'block_rates', 'block_counts', 'background_rate', 'episodes'
    event_times : array-like, optional
        Unbinned photon arrival times (sec). If provided, a histogram is drawn
        using the block edges for binning so model/data align perfectly.
    t, counts : array-like, optional
        Binned light curve using bin centers + counts. (Ignored if edges+counts given.)
    edges, counts : array-like, optional
        Binned light curve using bin edges + counts.
    """

    block_edges  = np.asarray(result["block_edges"], float)
    block_rates  = np.asarray(result["block_rates"], float)
    background_rate   = float(result["background_rate"])
    episodes     = result.get("episodes", [])

    # Build a consistent view of the raw data to draw under the model
    have_events = event_times is not None
    have_edges  = (edges is not None) and (counts is not None)
    have_centers= (t is not None) and (counts is not None) and (not have_edges)

    if not (have_events or have_edges or have_centers):
        raise ValueError("Provide event_times OR (t & counts) OR (edges & counts) to plot the data.")

    if have_centers:
        # infer edges from centers by using midpoints + extrapolated endcaps
        tc = np.asarray(t, float)
        if tc.size < 2:
            raise ValueError("Need at least 2 bin centers to infer edges.")
        dt_left  = tc[1] - tc[0]
        dt_right = tc[-1] - tc[-2]
        mids = 0.5*(tc[1:] + tc[:-1])
        edges = np.concatenate(([tc[0] - 0.5*dt_left], mids, [tc[-1] + 0.5*dt_right]))
        counts = np.asarray(counts, float)
    elif have_edges:
        edges  = np.asarray(edges, float)
        counts = np.asarray(counts, float)

    fig, ax = plt.subplots(figsize=figsize)

    # --- 1) Raw data
    if have_events:
        # Histogram events using the BB edges so the bars align with the model
        ax.hist(event_times, bins=block_edges, alpha=bar_alpha, label="Counts (events)", histtype="stepfilled")
        ylabel = "Counts per block"
        pass
    else:
        # Draw binned counts (step or bars). Use edges to align with blocks.
        # A step plot of count rate helps compare with block rates:
        widths = np.diff(edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = counts / widths
        # bars for counts; step for rate to compare with model
        # ax.bar(edges[:-1], counts, width=widths, align="edge", alpha=bar_alpha, edgecolor="none", label="Counts")
        ax.step(edges[:-1], rate, where="post", linewidth=1.0, label="Binned rate")
        ylabel = "Counts Rate (counts/s)"

    # --- 2) Bayesian-Blocks step model (rate)
    if show_blocks:
        ax.step(block_edges[:-1], block_rates, where="post", linewidth=block_linewidth, label="Bayesian Blocks")

    # --- 3) Background line
    if show_background:

        if background is not None:
            background_rates = background.integrate_energy()[0].rates
            tcenter = background.time_centroids()[0]
            tstart = background.tstart()[0]
            tstop = background.tstop()[0]

            # Plot the background rate as a step function
            ax.step(tcenter, background_rates, where="mid", linestyle="--", color='red', alpha=0.75, linewidth=1.0, label="Background")  

        elif background_rate is not None:
            ax.axhline(background_rate, linestyle="--", color='red', alpha=0.75, linewidth=1.0, label=f"Background")

        else:
            print("No background rate provided or estimated; skipping background line.")

    # --- 4) Episodes shading
    if show_episodes and episodes:

        print("\nFlaring Episode Boundries:")
        print("+----------------------+----------------------+")
        print("| lower edge (s)       | upper edge (s)       |")
        print("+----------------------+----------------------+")

        for ep in episodes:
            # ax.axvspan(ep["t_start"], ep["t_stop"], alpha=episode_alpha, label="Episode", ymin=0, ymax=1)

            print(f"| {float(ep["t_start"]):20.6f} | {float(ep["t_stop"]):20.6f} |")

            if color_episodes is False:
                episode_alpha = 0

            colors = ["red", "orange", "yellow", "green", "steelblue", "purple"]
            color = colors[episodes.index(ep) % len(colors)]
            ax.axvspan(
                ep["t_start"], ep["t_stop"],
                alpha=episode_alpha,
                label="Emission Episode" if episodes.index(ep) == 0 else None,
                ymin=0, ymax=1,
                color=color
            )

        print("+----------------------+----------------------+")

        # --- 4b) Optional dotted dividers between episodes
        if show_episodes and episodes and show_episode_dividers:

            # Default style: dotted, slightly transparent
            if episode_divider_kwargs is None:
                episode_divider_kwargs = dict(linestyle=":", linewidth=1.2, color='gray', alpha=0.9)

            # Sort episodes by start time
            eps = sorted(episodes, key=lambda e: e["t_start"])
            divider_positions = []

            if episode_divider_mode == "midgap":
                # One divider at the midpoint of each gap between episodes
                for a, b in zip(eps[:-1], eps[1:]):
                    mid = 0.5 * (a["t_stop"] + b["t_start"])
                    divider_positions.append(mid)
            else:
                # "boundaries": draw at each start/stop (dedup so we don't double-draw)
                for ep in eps:
                    divider_positions.append(float(ep["t_start"]))
                    divider_positions.append(float(ep["t_stop"]))

            # Deduplicate (tolerate tiny FP differences)
            uniq = []
            seen = set()
            for x in sorted(divider_positions):
                key = round(x, 9)
                if key not in seen:
                    uniq.append(x)
                    seen.add(key)

            # Draw episode dividers without adding them to the legend
            for x in uniq:
                kw = dict(episode_divider_kwargs)
                ax.axvline(x, **kw)

            
        # avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend(*zip(*unique), loc="best")
    else:
        ax.legend(loc="best")

    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)

    # Add a detector label and energy range label
    if detector_label is not None:
        ax.text(0.0175, 0.95, detector_label, transform=ax.transAxes,
                    ha='left', va='top', fontsize=10)
    if energy_label is not None:
        ax.text(max(0.01, 1.0 - 0.00925 * len(energy_label)), 0.95, energy_label, 
        transform=ax.transAxes, ha='left', va='top', fontsize=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.25)



    fig.tight_layout()
    fig.show()

    # return fig, ax
    return

