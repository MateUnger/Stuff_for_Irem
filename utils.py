import copy
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths
from scipy.interpolate import CubicSpline


def walk_direction_peaks(data: np.ndarray, kpt_labels: list, fps: float) -> np.ndarray:
    """
    Determines walking state (straight/turning) based on shoulder coordinates using peak detection method.

    Args:
        data:           Input data with shape (n_keypoints, n_dims, n_frames).
        kpt_labels:     List of keypoint labels corresponding to data (left_ankle, nose, etc).
        fps:            sampling frequency

    Returns:
        valid_segments: boolean array indicating straight walking segments (True) and turning segments (False).

    """

    # get the y coord of shoulders (for qualisys x-y plane is horizontal, y coord was forwards/backwards movement)
    shoulder_R = data[kpt_labels.index("right_shoulder"), 1, :]
    shoulder_L = data[kpt_labels.index("left_shoulder"), 1, :]

    shoulder_diff = np.abs(np.diff(shoulder_R - shoulder_L))
    shoulder_diff = shoulder_diff / np.nanmax(shoulder_diff)

    # Interpolate NaNs in shoulder_diff
    if np.any(np.isnan(shoulder_diff)):
        nans = np.isnan(shoulder_diff)
        not_nans = ~nans
        shoulder_diff[nans] = np.interp(
            np.flatnonzero(nans), np.flatnonzero(not_nans), shoulder_diff[not_nans]
        )

    # min distance from peak to peak (2 seconds of straight movement) in samples
    min_peak_distance = 2 * fps

    peaks, peak_properties = find_peaks(shoulder_diff, height=0.3, distance=min_peak_distance)
    widths, width_heights, left_ips, right_ips = peak_widths(
        x=shoulder_diff, peaks=peaks, rel_height=0.95
    )

    # mark turning segments as invalid for gait analysis
    # valid_segments = np.ones_like(data[0, 0, :], dtype=bool)
    # for left_base, right_base in zip(left_ips, right_ips):
    #     valid_segments[int(np.floor(left_base)) : int(np.ceil(right_base))] = False

    # valid_segments = np.full(shape=data[0, 0, :].shape, fill_value="straight")
    # for left_base, right_base in zip(left_ips, right_ips):
    #     valid_segments[int(np.floor(left_base)) : int(np.ceil(right_base))] = str("turn")

    valid_segments = ["straight"] * data.shape[2]
    for left_base, right_base in zip(left_ips, right_ips):
        turn = ["turn"] * (int(np.floor(right_base)) - int(np.ceil(left_base)) + 2)
        valid_segments[int(np.floor(left_base)) : int(np.ceil(right_base))] = turn

    return valid_segments


def interpolate_gaps(data: np.ndarray, fps: int, max_gap: float) -> np.ndarray:
    """
    Fills gaps below max_gap size along each dimension (1D) of the input array.
    Args:
        data (np.ndarray):  Input data with shape (n_keypoints, n_dims, n_frames).
        fps (int):          Frames per second of the data.
        max_gap (float):    Maximum gap size in seconds


    """
    n_keypoints, n_dims, n_frames = data.shape
    max_gap_frames = int(max_gap * fps)

    filled_data = np.copy(data)  # Preserve original data

    for kpt in range(n_keypoints):
        for dim in range(n_dims):
            signal = data[kpt, dim, :]
            nan_indices = np.where(np.isnan(signal))[0]

            if len(nan_indices) == 0:
                continue

            # Identify NaN segments
            diff = np.diff(nan_indices)
            segment_starts = np.insert(nan_indices[np.where(diff > 1)[0] + 1], 0, nan_indices[0])
            segment_ends = np.append(nan_indices[np.where(diff > 1)[0]], nan_indices[-1])

            # Interpolate gaps within the allowed size
            for start, end in zip(segment_starts, segment_ends):
                gap_size = end - start + 1
                if gap_size <= max_gap_frames:

                    valid_indices = np.where(~np.isnan(signal))[0]
                    # fit cubic spline if there are enough points
                    if len(valid_indices) >= 2:
                        cs = CubicSpline(valid_indices, signal[valid_indices])
                        filled_data[kpt, dim, start : end + 1] = cs(np.arange(start, end + 1))

                    # use linear interpolation if not enough valid points
                    else:
                        non_nan_idx = np.where(~np.isnan(signal))[0]
                        filled_data[kpt, dim, :] = np.interp(
                            np.arange(n_frames), non_nan_idx, signal[non_nan_idx]
                        )

    return filled_data


def filter_data(
    data: np.ndarray, sampling_rate: float, cutoff: float, order: int, gap_size: int
) -> np.ndarray:
    """
    Interpolates and applies a low-pass filter to data (with NaNs).

    Args:
        data (np.ndarray):      Input data with shape (n_kpt, n_dims, n_frames).
        sampling_rate (float):  Sampling rate of the data.
        cutoff (float):         Cutoff frequency for the low-pass filter.
        order (int):            Order of the Butterworth filter.
        gap_size (int):         Maximum size (in seconds) of gaps to fill

    Returns:
        filtered_data (np.ndarray): Interpolated and filtered data with the origianl Nan values in place.

    """
    n_kpt, n_dims, n_frames = data.shape
    data = interpolate_gaps(data, sampling_rate, gap_size)  # Fill short gaps
    filtered_data = data.copy()

    # Design Butterworth low-pass filter
    b, a = butter(N=order, Wn=cutoff / (0.5 * sampling_rate), btype="low", analog=False)

    for kpt in range(n_kpt):
        for dim in range(n_dims):
            trajectory = data[kpt, dim, :]

            # search for NaNs
            nans = np.isnan(trajectory)

            # if there are nans, interpolate the missing values for subsequent filtering
            if np.any(nans):
                valid_indices = ~nans
                trajectory[nans] = np.interp(
                    np.flatnonzero(nans), np.flatnonzero(valid_indices), trajectory[valid_indices]
                )

            trajectory = filtfilt(b, a, trajectory)  # Apply filter
            trajectory[nans] = np.nan  # Restore NaNs

            filtered_data[kpt, dim, :] = trajectory

    return filtered_data


def step_detection(data: np.ndarray, kpt_labels: list, properties: dict) -> dict:
    """
    
    Args:
        data: filtered pose data (keypoints x dims x frames)
        kpt_labels: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        properties: dictionary of static parameters from properties.json

    Returns:
        events: dictionary matching frame indices to gait events (IC, FC)...
                for each side (left, right) along with perspective (frontal, saggital)
    """

    # perspective = walk_direction(data, kpt_labels, properties["stride_min"], properties["fps"])
    perspective = walk_direction_peaks(data, kpt_labels, properties["fps"])

    events = {"left": {}, "right": {}}

    for side in ["left", "right"]:
        _, _, velocity = bruening_ridge_detection(data, 1000, side, kpt_labels, properties)
        ICs, FCs, _ = bruening_ridge_detection(data, velocity, side, kpt_labels, properties)
        events[side]["ICs"] = ICs
        events[side]["FCs"] = FCs

    IC_events = [
        {"frame": frame, "side": side, "perspective": perspective[frame]}
        for side in ["left", "right"]
        for frame in events[side]["ICs"]
    ]
    FC_events = [
        {"frame": frame, "side": side, "perspective": perspective[frame]}
        for side in ["left", "right"]
        for frame in events[side]["FCs"]
    ]

    return ({"IC": IC_events, "FC": FC_events}, perspective)


def walk_direction(
    data: np.ndarray, keypoint_mapping: list, min_length: float, fps: float
) -> np.ndarray:
    """
    Determines walking direction (frontal/saggital) based on shoulder coordinates.
    Args:
        data: interpolated and filtered pose data (keypoints x dims x frames)
        keypoint_mapping: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        min_length: minimum segment length in seconds
        fps: sampling frequency of recorded data

    Returns:
        perspective: array of 'frontal', 'sagittal', or None for each frame
    """

    # get shoulder coordinates (y-axis)
    # note: in qualisys data, y-axis is left-right, x is front-back, z is up-down
    shoulder_R = data[keypoint_mapping.index("right_shoulder"), 1, :]
    shoulder_L = data[keypoint_mapping.index("left_shoulder"), 1, :]

    orientation = shoulder_R - shoulder_L
    perspective = np.where(
        orientation > 0, "frontal", np.where(orientation < 0, "sagittal", None)
    ).astype(object)

    # max gap size in frames
    max_gap_size = int(min_length * fps)
    valid_indices = np.where(perspective != None)[0]

    # TODO: figure out what this does, add comments
    for start, end in zip(valid_indices[:-1], valid_indices[1:]):
        if end - start <= max_gap_size:
            perspective[start + 1 : end] = perspective[start]

    changes = np.r_[True, perspective[:-1] != perspective[1:], True]
    segment_starts, segment_ends = np.where(changes[:-1])[0], np.where(changes[1:])[0] - 1

    for start, end in zip(segment_starts, segment_ends):
        if (end - start + 1) / fps < min_length:
            perspective[start : end + 1] = None

    return perspective


def bruening_ridge_detection(
    data: np.ndarray, velocity: float, side: str, kpt_labels: list, properties: dict
) -> tuple:
    """
    Identifies Gait Events (GE) using the velocity of markers on the foot (heel, toe, ankle).

    Args:
        data: interpolated and filtered pose data from qualisys (keypoints x dims x frames)
        velocity: initial walking velocity estimate (m/s)
        side: 'left' or 'right'
        kpt_labels: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        properties: dictionary of static parameters from properties.json

    Returns:
        ICs: list of Initial Contact (IC) frame indices
        FCs: list of Final Contact (FC) frame indices
        velocity: computed walking velocity (m/s)

    """
    fs = properties["fps"]
    heel_thr = properties["heel_thr"] * velocity
    # use ankle marker in case the other 2 are not visible
    ankle_thr = properties["heel_thr"] * velocity
    big_toe_thr = properties["toe_thr"] * velocity

    # Extract trajectories
    heel = data[kpt_labels.index(f"{side}_heel"), :, :]
    big_toe = data[kpt_labels.index(f"{side}_big_toe"), :, :]
    ankle = data[kpt_labels.index(f"{side}_ankle"), :, :]

    # Compute 3D velocities
    heel_vel = np.linalg.norm(np.diff(heel, axis=1), axis=0) * fs
    ankle_vel = np.linalg.norm(np.diff(ankle, axis=1), axis=0) * fs
    big_toe_vel = np.linalg.norm(np.diff(big_toe, axis=1), axis=0) * fs

    # Ground contact detection based on thresholds
    ground_contact = (
        (heel_vel < heel_thr) | (ankle_vel < ankle_thr) | (big_toe_vel < big_toe_thr)
    ).astype(int)

    # Remove short ground contact periods
    min_gc_duration = int(properties["stance_min"] * fs)
    min_no_gc_duration = int(properties["swing_min"] * fs)

    contact_diff = np.diff(np.r_[0, ground_contact, 0])
    starts = np.where(contact_diff == 1)[0]
    ends = np.where(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_gc_duration:
            ground_contact[start:end] = 0

    # Remove short no-contact periods
    contact_diff = np.diff(np.r_[0, ground_contact, 0])
    starts = np.where(contact_diff == 1)[0]
    ends = np.where(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_no_gc_duration:
            ground_contact[start:end] = 1

    # Identify initial contacts (ICs) and final contacts (FCs)
    ground_contact_diff = np.diff(ground_contact)
    ICs = np.where(ground_contact_diff == 1)[0] + 1
    FCs = np.where(ground_contact_diff == -1)[0] + 1

    # Compute walking velocity from stride lengths and durations
    stride_lengths = []
    stride_durations = []

    for i in range(1, len(ICs)):
        stride_duration = (ICs[i] - ICs[i - 1]) / fs
        if properties["stride_min"] <= stride_duration <= properties["stride_max"]:
            stride_length = np.linalg.norm(heel[:, ICs[i]] - heel[:, ICs[i - 1]])
            stride_lengths.append(stride_length)
            stride_durations.append(stride_duration)

    velocity = (
        np.nanmean(np.array(stride_lengths) / np.array(stride_durations))
        if stride_durations
        else np.nan
    )

    return ICs, FCs, velocity


def get_frame_index(gait_events: list, side: str, lower_bound: int, upper_bound: int) -> list:
    """
    Return frame indices from gait_events with the given side from lower_bound to upper_bound frame index.

    Args:
        gait_events: input data
        side: 'left' or 'right'
        lower_bound: lower frame index
        upper_bound: upper frame index
    Returns:
        list of gait events
    """
    return [
        event["frame"]
        for event in gait_events
        if event["side"] == side and lower_bound < event["frame"] < upper_bound
    ]


def compute_pooled_stats(left_values, right_values):
    """
    Calculates
    """
    pooled = np.concatenate([left_values, right_values])
    pooled = pooled[~np.isnan(pooled)]
    mean = np.mean(pooled) if len(pooled) > 0 else np.nan
    # variabilty (std normalized by mean) (coeff of variation)
    cv = 100 * np.std(pooled) / mean if mean != 0 else np.nan
    return {"mean": mean, "CV": cv}


def compute_asymmetry(left_values, right_values):
    """
    Calculates asymmetry between left and right sides
    """
    left, right = map(lambda x: np.array(x)[~np.isnan(x)], [left_values, right_values])
    if len(left) == 0 or len(right) == 0:
        return np.nan
    larger, smaller = max(left.mean(), right.mean()), min(left.mean(), right.mean())
    return 100 * (1 - smaller / larger) if larger > 0 else np.nan


def gait_analysis(data: np.ndarray, events: dict, keypoint_mapping: list, properties: dict) -> dict:

    # static properties for calculations
    fs = properties["fps"]
    stride_min = properties["stride_min"]
    stride_max = properties["stride_max"]

    perspectives = ["all", "straight", "turn"]
    # metrics of interest (for each side)
    moi = {
        metric: []
        for metric in [
            "stime",
            "slen",
            "vel",
            "swing",
            "dsupp",
            "bos",
        ]
    }
    # output data structure
    metrics = {
        perspective: {
            "left": copy.deepcopy(moi),
            "right": copy.deepcopy(moi),
        }
        for perspective in perspectives
    }

    # Initial- and Final-contac gait events
    ICs = events["IC"]
    FCs = events["FC"]

    # iterate Initial Contact gait events
    for i, IC in enumerate(ICs):

        # ipsilateral and contralateral sides for current gait event
        ipsi, contra = IC["side"], "left" if IC["side"] == "right" else "right"

        # perspective value ('straight'/'turn') or 'all' if missing
        perspective = IC.get("perspective", "all")

        if perspective not in perspectives:
            continue

        # get the next ipsilateral event's global index (relative to the whole event list), return None otherwise
        same_foot_next_idx = next(
            (j for j, event in enumerate(ICs[i + 1 :], start=i + 1) if event["side"] == ipsi), None
        )

        # check if event order is correct, filter false positives
        if same_foot_next_idx is not None:

            # stride time = time elapsed between heelstrikes of the same foot
            stime = (ICs[same_foot_next_idx]["frame"] - IC["frame"]) / fs

            # stride time falls in realistic time range
            if stride_min <= stime <= stride_max:
                # frame index of current and next IC event (ipsilateral)
                IC0, IC2 = IC["frame"], ICs[same_foot_next_idx]["frame"]

                # frame index of first contralateral heel strike (between current and next ipsilateral)
                IC1 = get_frame_index(ICs, contra, IC0, IC2)

                FC0, FC1, FC2 = None, None, None

                # if there's a next contralateral GE
                if IC1:
                    IC1 = IC1[0]

                    # see gait_events.png
                    FC0 = get_frame_index(FCs, contra, IC0, IC1)
                    FC1 = get_frame_index(FCs, ipsi, IC1, IC2)
                    FC2 = get_frame_index(FCs, ipsi, IC2, IC2 + int(fs * stride_max))

                    FC0 = FC0[0] if FC0 else None
                    FC1 = FC1[0] if FC1 else None
                    FC2 = FC2[0] if FC2 else None

                # if either of the values is None, ignore it
                if any(x is None for x in [IC0, IC1, IC2, FC0, FC1, FC2]):
                    continue

                # heel point
                HP0 = np.nanmedian(data[keypoint_mapping.index(f"{ipsi}_heel"), :, IC0:FC0], axis=1)
                HP2 = np.nanmedian(data[keypoint_mapping.index(f"{ipsi}_heel"), :, IC2:FC2], axis=1)
                HP1 = np.nanmedian(
                    data[keypoint_mapping.index(f"{contra}_heel"), :, IC1:FC1], axis=1
                )

                # stride lenght on xy plane
                slen = np.linalg.norm(HP2[:2] - HP0[:2])
                vel = slen / stime
                bos = np.linalg.norm(np.cross(HP2 - HP1, HP1 - HP0)) / np.linalg.norm(HP2 - HP1)
                swing = (IC2 - FC1) / fs
                dsupp = ((FC0 - IC0) + (FC1 - IC1)) / fs

                for pers in ["all", perspective]:
                    metrics[pers][ipsi]["stime"].append(stime)
                    metrics[pers][ipsi]["slen"].append(slen)
                    metrics[pers][ipsi]["vel"].append(vel)
                    metrics[pers][ipsi]["swing"].append(swing)
                    metrics[pers][ipsi]["dsupp"].append(dsupp)
                    metrics[pers][ipsi]["bos"].append(bos)

    parameters = {}
    for state, data in metrics.items():
        parameters[state] = {
            metric: {
                **compute_pooled_stats(data["left"][metric], data["right"][metric]),
                "asymmetry": compute_asymmetry(data["left"][metric], data["right"][metric]),
            }
            for metric in data["left"]
        }

    return parameters


def display_results(parameters):
    parameter_order = [
        "stime",
        "slen",
        "vel",
        "swing",
        "dsupp",
        "bos",
    ]
    statistic_order = ["Mean", "CV", "Asymmetry"]

    rows = []
    for segment_type, metrics in parameters.items():
        for param, values in metrics.items():
            rows.extend(
                [
                    {
                        "Parameter": param,
                        "Statistic": "Mean",
                        "Perspective": segment_type,
                        "Value": values["mean"],
                    },
                    {
                        "Parameter": param,
                        "Statistic": "CV",
                        "Perspective": segment_type,
                        "Value": values["CV"],
                    },
                    {
                        "Parameter": param,
                        "Statistic": "Asymmetry",
                        "Perspective": segment_type,
                        "Value": values["asymmetry"],
                    },
                ]
            )
    pd.options.display.float_format = "{:,.1f}".format
    df = pd.DataFrame(rows)
    df["Parameter"] = pd.Categorical(df["Parameter"], categories=parameter_order, ordered=True)
    df["Statistic"] = pd.Categorical(df["Statistic"], categories=statistic_order, ordered=True)
    df = df.sort_values(by=["Parameter", "Statistic"])

    table = df.pivot_table(
        index=["Parameter", "Statistic"],
        columns="Perspective",
        values="Value",
        aggfunc="mean",
        observed=False,
    )
    table = table.reset_index()
    table.columns.name = None

    print(table)
    return table
