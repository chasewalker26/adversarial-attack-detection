import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def _normalize_scale(attr, scale_factor):
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def _cumulative_sum_threshold(values, percentile):
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_attr(attr, norm, outlier_perc = 2, reduction_axis = 2):
    attr_combined = attr
    attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if norm == "absolute":
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif norm == "positive":
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif norm == "negative":
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif norm == "all":
        norm = norm
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)

    return _normalize_scale(attr_combined, threshold)

def attr_to_subplot(input, title, axs, norm = "absolute", cmap = None, original_image = False, blended_image = None):
    axs.set_title(title)
    if original_image == False:
        if norm == "absolute":
            cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, '#ffffff'), (0.25, '#0000ff'), (1, '#0000ff')], N = 256)   
            vmin, vmax = 0, 1
        elif norm == "positive":
            cmap = "Greens"
            vmin, vmax = 0, 1
        elif norm == "negative":
            cmap = "Reds"
            vmin, vmax = 0, 1
        elif norm == "all":
            cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
            vmin, vmax = -1, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")

        axs.imshow(_normalize_attr(input, norm), cmap = cmap, vmin = vmin, vmax = vmax)

        if blended_image is not None:
            axs.imshow(np.mean(np.transpose(blended_image.squeeze().detach().cpu().numpy(), (1, 2, 0)), axis = 2), cmap = "gray")
            axs.imshow(_normalize_attr(input, norm), cmap = cmap, vmin = vmin, vmax = vmax, alpha = 0.7)
    else:
        # normalize input [0, 1] and show
        axs.imshow(np.transpose(input.squeeze().detach().cpu().numpy(), (1, 2, 0)))

    axs.set_xticks([])
    axs.set_yticks([])

    return