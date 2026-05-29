import numpy as np

def gaussian_kernel(size, sigma=1.0):
    """
    Create a 1D Gaussian kernel.
    
    Parameters:
    -----------
    size : int
        Size of the kernel (should be odd for symmetric kernel)
    sigma : float
        Standard deviation of the Gaussian
        
    Returns:
    --------
    kernel : ndarray
        1D Gaussian kernel normalized to sum to 1
    """
    if size % 2 == 0:
        size += 1  # Make it odd for symmetry
    
    x = np.arange(size) - size // 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    return kernel

def gaussian_smooth_1d(data, sigma=1.0, kernel_size=None):
    """
    Smooth 1D data using Gaussian kernel convolution.
    
    Parameters:
    -----------
    data : array-like
        Input 1D data to smooth
    sigma : float
        Standard deviation of the Gaussian kernel
    kernel_size : int, optional
        Size of the kernel. If None, calculated as 6*sigma + 1
        
    Returns:
    --------
    smoothed : ndarray
        Smoothed data
    """
    data = np.asarray(data)
    
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Use 'same' mode to keep output same length as input
    smoothed = np.convolve(data, kernel, mode='same')
    
    return smoothed



def segmented_gaussian_smooth(x_values, y_values, smooth_dict):
    """
    Apply Gaussian smoothing with varying sigma across different segments.
    Trims edge points from each segment to remove convolution artifacts,
    then interpolates back to the original x grid.

    Parameters:
    -----------
    x_values : array-like
        X coordinates of data points (must be sorted)
    y_values : array-like
        Y values to smooth
    smooth_dict : dict, optional
        Keys:
            segments : list/array
                X-values defining interior segment boundaries (e.g., [5, 10])
            sigma : list of float
                Sigma per segment. Length must be len(segments) + 1.
            trim : int, number of sigmas to use to trim


    Returns:
    --------
    x_out : ndarray
        Original x_values (unchanged)
    y_out : ndarray
        Smoothed y values on the original x grid
    segment_map : list of tuples
        (start_idx, end_idx) for each segment in the original x_values array
    """
    if smooth_dict is None:
        smooth_dict = {}

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    segment_points = np.asarray(sorted(smooth_dict["segments"]))
    sigma_list = smooth_dict["sigma"]
    edge_trim_factor = smooth_dict["trim"]


    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    segment_points = np.asarray(sorted(segment_points))
    
    if len(sigma_list) != (len(segment_points)+1) :
        raise ValueError("For each segment, you need to privede two sigma values for left and right segments." \
        " Length of sigma_list should be one more than number of segment_points.")

    # Ensure boundaries cover data range
    if segment_points[0] > x_values.min():
        segment_points = np.concatenate([[x_values.min()], segment_points])
    if segment_points[-1] < x_values.max():
        segment_points = np.concatenate([segment_points, [x_values.max()]])
    
    n_segments = len(segment_points) - 1
    
    x_smooth_parts = []
    y_smooth_parts = []
    segment_map = []
    current_idx = 0
    print(n_segments, "segments defined by points:", segment_points)
    for seg_idx in range(n_segments):
        x_start = segment_points[seg_idx]
        x_end = segment_points[seg_idx + 1]
        sigma = sigma_list[seg_idx]
        
        # Find data in this segment
        mask = (x_values >= x_start) & (x_values <= x_end)
        seg_x = x_values[mask]
        seg_y = y_values[mask]
        
        if len(seg_y) < 3:
            print("Segmentation problem!")
            continue
        
        # Apply smoothing
        seg_y_smooth = gaussian_smooth_1d(seg_y, sigma=sigma)
        
        # Calculate trim amount (3σ covers 99.7% of Gaussian mass)
        trim = int(edge_trim_factor * sigma)
        trim = min(trim, len(seg_y) // 3)  # Safety: don't over-trim
        
        # Trim edges: first segment only trims right, last only trims left,
        # middle segments trim both
        if seg_idx == 0:
            x_trimmed = seg_x[:-trim] if trim > 0 else seg_x
            y_trimmed = seg_y_smooth[:-trim] if trim > 0 else seg_y_smooth
        elif seg_idx == n_segments - 1:
            x_trimmed = seg_x[trim:] if trim > 0 else seg_x
            y_trimmed = seg_y_smooth[trim:] if trim > 0 else seg_y_smooth
        else:
            x_trimmed = seg_x[trim:-trim] if trim > 0 else seg_x
            y_trimmed = seg_y_smooth[trim:-trim] if trim > 0 else seg_y_smooth
        
        x_smooth_parts.append(x_trimmed)
        y_smooth_parts.append(y_trimmed)
        
        seg_length = len(x_trimmed)
        segment_map.append((current_idx, current_idx + seg_length))
        current_idx += seg_length
    
    x_smooth = np.concatenate(x_smooth_parts)
    y_smooth = np.concatenate(y_smooth_parts)
    
    return x_smooth, y_smooth, segment_map


def gaussian_smooth_1d_reflected(data, sigma):
    """Smooth with reflected padding to eliminate boundary artifacts."""
    data = np.asarray(data)
    pad = int(3 * sigma)
    
    # Reflect data at both ends
    padded = np.concatenate([data[pad:0:-1], data, data[-2:-pad-2:-1]])
    smoothed = gaussian_smooth_1d(padded, sigma=sigma)
    
    # Remove padding
    tmp = smoothed[pad:-pad] if pad > 0 else smoothed

    print(len(data))
    print(len(tmp))
    return tmp


def segmented_gaussian_smooth_v2(x_values, y_values, smooth_dict=None):
    """
    Apply Gaussian smoothing with varying sigma across different segments.
    Trims edge points from each segment to remove convolution artifacts,
    then interpolates back to the original x grid.

    Parameters:
    -----------
    x_values : array-like
        X coordinates of data points (must be sorted)
    y_values : array-like
        Y values to smooth
    smooth_dict : dict, optional
        Keys:
            segments : list/array
                X-values defining interior segment boundaries (e.g., [5, 10])
            sigma : list of float
                Sigma per segment. Length must be len(segments) + 1.


    Returns:
    --------
    x_out : ndarray
        Original x_values (unchanged)
    y_out : ndarray
        Smoothed y values on the original x grid
    segment_map : list of tuples
        (start_idx, end_idx) for each segment in the original x_values array
    """
    if smooth_dict is None:
        smooth_dict = {}

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    segment_points = np.asarray(sorted(smooth_dict["segments"]))
    sigma_list = smooth_dict["sigma"]
    

    if len(sigma_list) != len(segment_points) + 1:
        raise ValueError(
            f"Expected {len(segment_points) + 1} sigma values for "
            f"{len(segment_points)} boundary point(s), got {len(sigma_list)}."
        )

    # Ensure boundaries cover full data range
    if segment_points[0] > x_values.min():
        segment_points = np.concatenate([[x_values.min()], segment_points])
    if segment_points[-1] < x_values.max():
        segment_points = np.concatenate([segment_points, [x_values.max()]])

    n_segments = len(segment_points) - 1
    print(f"{n_segments} segment(s) defined by points: {segment_points}")

    y_smooth_parts = []
    segment_map = []

    for seg_idx in range(n_segments):
        x_start = segment_points[seg_idx]
        x_end = segment_points[seg_idx + 1]
        sigma = sigma_list[seg_idx]

        mask = (x_values >= x_start) & (x_values <= x_end)
        seg_x = x_values[mask]
        seg_y = y_values[mask]

        if len(seg_y) < 3:
            print(f"Segment {seg_idx} too short, skipping.")
            continue

        seg_y_smooth = gaussian_smooth_1d_reflected(seg_y, sigma=sigma)
        y_smooth_parts.extend(seg_y_smooth)
        
        pass

    y_smooth_parts = np.array(y_smooth_parts)
    # print(y_smooth_parts.shape)
    return y_smooth_parts, segment_map

