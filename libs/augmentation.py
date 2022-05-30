import itertools
import scipy
import numpy as np
import torch

"""
An re-implemention of Tang et al.: https://arxiv.org/abs/2011.11542
"""


def choose_aug(sample, aug_methd, aug_params):
    params = dict(aug_params)
    if aug_methd == 'jitter':
        return jitter(sample, float(params['jitter_ratio']))
    elif aug_methd == 'scale':
        return scaling(sample, float(params['scale_ratio']))
    elif aug_methd == 'jitter_scale':
        return scaling(jitter(sample, float(params['jitter_ratio'])), float(params['scale_ratio']))
    elif aug_methd == 'permutation':
        return permutation(sample, max_segments=aug_params['max_seg'])
    elif aug_methd == 'permutation_jitter':
        return jitter(permutation(sample, max_segments=int(params['max_seg'])), float(params['jitter_ratio']))
    elif aug_methd == 'rotation':
        return rotation(sample)
    elif aug_methd == 'invert':
        return invert(sample)
    elif aug_methd == 'timeflip':
        return time_flip(sample)
    elif aug_methd == 'shuffle':
        return channel_shuffle(sample)
    elif aug_methd == 'warp':
        return warp(sample, sigma=float(params['sigma']), num_knots=int(params['num_knots']))


def DataTransform(sample, aug_method1, aug_method2, aug_params):
    """

    :param sample: shape = (batch_size, seq_len, channel)
    :param aug_method1:
    :param aug_method2:
    :param aug_params:
    :return:
    """
    sample = sample.numpy()
    aug1 = choose_aug(sample, aug_method1, aug_params)
    aug2 = choose_aug(sample, aug_method2, aug_params)
    return torch.from_numpy(aug1), torch.from_numpy(aug2)


def get_cubic_spline_interpolation(x_eval, x_data, y_data):
    cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)


def warp(x, sigma=0.2, num_knots=4):
    time_steps = np.arange(x.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (x.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0] * x.shape[2], num_knots + 2))

    spline_values = np.array(
        [get_cubic_spline_interpolation(time_steps, knot_xs, spline_ys_individual) for spline_ys_individual in
         spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (x.shape[1] - 1)

    X_transformed = np.empty(shape=x.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        X_transformed[i // x.shape[2], :, i % x.shape[2]] = np.interp(time_steps, distorted_time_stamps,
                                                                      x[i // x.shape[2], :, i % x.shape[2]])
    return X_transformed


def channel_shuffle(x):
    # https://arxiv.org/abs/2011.11542
    channels = range(x.shape[2])
    all_channel_permutations = np.array(list(itertools.permutations(channels))[1:])

    random_permutation_indices = np.random.randint(len(all_channel_permutations), size=(x.shape[0]))
    permuted_channels = all_channel_permutations[random_permutation_indices]
    X_transformed = x[np.arange(x.shape[0])[:, np.newaxis, np.newaxis],
                      np.arange(x.shape[1])[np.newaxis, :, np.newaxis],
                      permuted_channels[:, np.newaxis, :]]
    return X_transformed


def time_flip(x):
    # https://arxiv.org/abs/2011.11542
    return x[:, ::-1, :].copy()


def invert(x):
    # https://arxiv.org/abs/2011.11542
    return x * -1


def rotation(x):
    # https://arxiv.org/abs/2011.11542
    axes = np.random.uniform(low=-1, high=1, size=(x.shape[0], x.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(x.shape[0]))
    matrices = axis_angle_to_rotation_matrix(axes, angles)
    acc = np.matmul(x[:, :, :3], matrices)
    gyro = np.matmul(x[:, :, 3:], matrices)

    return np.concatenate((acc, gyro), axis=2)


def axis_angle_to_rotation_matrix(axes, angles):
    # https://arxiv.org/abs/2011.11542
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2, 0, 1))
    return matrix_transposed

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1, x.shape[2]))
    return x * factor


def permutation(x, max_segments=5, seg_mode="random"):
    num_segs = np.random.randint(1, max_segments)

    segment_points_permuted = np.random.choice(x.shape[1], size=(x.shape[0], num_segs))
    segment_points = np.sort(segment_points_permuted, axis=1)

    X_transformed = np.empty(shape=x.shape)
    for i, (sample, segments) in enumerate(zip(x, segment_points)):
        splitted = np.array(np.split(sample, np.append(segments, x.shape[1])))
        np.random.shuffle(splitted)
        concat = np.concatenate(splitted, axis=0)
        X_transformed[i] = concat
    return X_transformed

