import math
import numpy as np
import torch


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def fmat(arr):
    return np.around(arr,3)


def to_tensor(x):
    return torch.tensor(x).cuda()


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.clamp(dists, min=1e-8)
    return dists.float()


def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)
    grouped_inds[dists > radius ** 2] = N
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, min(K, grouped_inds.size(2)))
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    if rt_density:
        return grouped_inds, density
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    assert M < 0
    new_xyz = xyz
    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
                                           rt_density=True)
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, min(K, grouped_inds.size(2)), 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    return new_xyz, new_points, grouped_inds, grouped_xyz


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors
    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0
    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)
    Returns:
    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


# def random_select_points(pc, m):
#     if m < 0:
#         idx = np.arange(pc.shape[0])
#         np.random.shuffle(idx)
#         return pc[idx, :]
#     n = pc.shape[0]
#     replace = False if n >= m else True
#     idx = np.random.choice(n, size=(m, ), replace=replace)
#     return pc[idx, :]


# def generate_rotation_x_matrix(theta):
#     mat = np.eye(3, dtype=np.float32)
#     mat[1, 1] = math.cos(theta)
#     mat[1, 2] = -math.sin(theta)
#     mat[2, 1] = math.sin(theta)
#     mat[2, 2] = math.cos(theta)
#     return mat


# def generate_rotation_y_matrix(theta):
#     mat = np.eye(3, dtype=np.float32)
#     mat[0, 0] = math.cos(theta)
#     mat[0, 2] = math.sin(theta)
#     mat[2, 0] = -math.sin(theta)
#     mat[2, 2] = math.cos(theta)
#     return mat


# def generate_rotation_z_matrix(theta):
#     mat = np.eye(3, dtype=np.float32)
#     mat[0, 0] = math.cos(theta)
#     mat[0, 1] = -math.sin(theta)
#     mat[1, 0] = math.sin(theta)
#     mat[1, 1] = math.cos(theta)
#     return mat


# def generate_random_rotation_matrix(angle1=-45, angle2=45):
#     thetax = np.random.uniform() * np.pi * angle2 / 180.0
#     thetay = np.random.uniform() * np.pi * angle2 / 180.0
#     thetaz = np.random.uniform() * np.pi * angle2 / 180.0
#     matx = generate_rotation_x_matrix(thetax)
#     maty = generate_rotation_y_matrix(thetay)
#     matz = generate_rotation_z_matrix(thetaz)
#     return np.dot(matx, np.dot(maty, matz))


# def generate_random_tranlation_vector(range1=-0.5, range2=0.5):
#     tranlation_vector = np.random.uniform(range1, range2, size=(3, )).astype(np.float32)
#     return tranlation_vector


# def transform(pc, R, t=None):
#     pc = np.dot(pc, R.T)
#     if t is not None:
#         pc = pc + t
#     return pc


# def batch_transform(batch_pc, batch_R, batch_t=None):
#     '''
#     :param batch_pc: shape=(B, N, 3)
#     :param batch_R: shape=(B, 3, 3)
#     :param batch_t: shape=(B, 3)
#     :return: shape(B, N, 3)
#     '''
#     transformed_pc = torch.matmul(batch_pc, batch_R.permute(0, 2, 1).contiguous())
#     if batch_t is not None:
#         transformed_pc = transformed_pc + torch.unsqueeze(batch_t, 1)
#     return transformed_pc


# def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
#     N, C = pc.shape
#     assert(clip > 0)
#     #jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(np.float32)
#     jittered_data = np.clip(
#         np.random.normal(0.0, scale=sigma, size=(N, 3)),
#         -1 * clip, clip).astype(np.float32)
#     jittered_data += pc
#     return jittered_data


# def inv_R_t(R, t):
#     inv_R = R.permute(0, 2, 1).contiguous()
#     inv_t = - inv_R @ t[..., None]
#     return inv_R, torch.squeeze(inv_t, -1)


# def uniform_2_sphere(num: int = None):
#     """Uniform sampling on a 2-sphere
#     Source: https://gist.github.com/andrewbolster/10274979
#     Args:
#         num: Number of vectors to sample (or None if single)
#     Returns:
#         Random Vector (np.ndarray) of size (num, 3) with norm 1.
#         If num is None returned value will have size (3,)
#     """
#     if num is not None:
#         phi = np.random.uniform(0.0, 2 * np.pi, num)
#         cos_theta = np.random.uniform(-1.0, 1.0, num)
#     else:
#         phi = np.random.uniform(0.0, 2 * np.pi)
#         cos_theta = np.random.uniform(-1.0, 1.0)

#     theta = np.arccos(cos_theta)
#     x = np.sin(theta) * np.cos(phi)
#     y = np.sin(theta) * np.sin(phi)
#     z = np.cos(theta)
#     return np.stack((x, y, z), axis=-1)


# def random_crop(pc, p_keep):
#     rand_xyz = uniform_2_sphere()
#     centroid = np.mean(pc[:, :3], axis=0)
#     pc_centered = pc[:, :3] - centroid

#     dist_from_plane = np.dot(pc_centered, rand_xyz)
#     mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
#     return pc[mask, :]


# def shuffle_pc(pc):
#     return np.random.permutation(pc)


# def flip_pc(pc, r=0.5):
#     if np.random.random() > r:
#         pc[:, 1] = -1 * pc[:, 1]
#     return pc