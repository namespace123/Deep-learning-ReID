# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/16
# -------------------------------------------------------------------------------
import numpy as np
import torch

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    # clamp做简单数值处理(为了数值的稳定性)：小于min参数的dist元素值由min值取代。
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def batch_euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [Batch size, Local part, Feature channel]
        y: pytorch Variable, with shape [Batch size, Local part, Feature channel]
    Returns:
        dist: pytorch Variable, with shape [Batch size, Local part, Local part]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    # shape [N, m, n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy

    dist.baddbmm_(x, y.permute(0, 2, 1), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if 'False'(?)
    Return:
         dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
         dist_an: pytorch Variable, distance(anchor, negative); shape [N]
         p_inds: pytorch LongTensor, with shape [N];
           indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
         n_inds: pytorch LongTensor, with shape [N];
           indices of selected hard negative samples; 0 <= p_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples, thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # 'dist_ap' means distance(anchor, positive)
    # both 'dist_ap' and 'relative_p_inds' with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # 'dist_an' means distance(anchor, positive)
    # both 'dist_ap' and 'relative_p_inds' with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]  正负样本的index
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def shortest_dist(dist_mat):
    """Parallel version.
    Args:
        dist_mat: pytorch Variable, avilable shape:
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
        dist: three cases corresponding to 'dist_mat':
        1) scalar
        2) pytorch Variable, with shape [N]
        3) pytorch Variable, with shape [*]
    """

    m, n = dist_mat.size()[:2]
    dist = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                dist[i][j] = dist_mat[i][j]
            elif i == 0 and j > 0:
                dist[i][j] = dist_mat[i][j] + dist[i][j-1]
            elif j == 0 and  i > 0:
                dist[i][j] = dist_mat[i][j] + dist[i-1][j]
            else:
                dist[i][j] = dist_mat[i][j] + torch.min(dist[i][j-1], dist[i-1][j])

    dist = dist[-1][-1]
    return dist

def batch_local_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [N, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [N]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    dist_mat = batch_euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.0) / (torch.exp(dist_mat) + 1.0)  # 为了训练的稳定，进行归一化 0-1
    dist = shortest_dist(dist_mat.permute(1, 2, 0))

    return dist


if __name__ == '__main__':

    x = torch.randn(32, 8, 128)
    y = torch.randn(32, 8, 128)
    # dist_mat = euclidean_dist(x, y)
    # dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, return_inds=True)
    local_dist = batch_local_dist(x, y)  # local_dist.shape: torch.Size([32])
    print(local_dist.shape)

'''
cosine_dist from HOReid
欧式距离是看成坐标系中两个点，来计算两点之间的距离，一般指位置上的差别，即距离；
余弦相似度是看成坐标系中两个向量，来计算两向量之间的夹角，一般指方向上的差别，即所成夹角。
'''
def cosine_dist(x, y):
    '''
    :param x: torch.tensor, 2d
    :param y: torch.tensor, 2d
    :return:
    '''

    bs1 = x.size()[0]
    bs2 = y.size()[0]

    frac_up = torch.matmul(x, y.transpose(0, 1))  # y.transpose(0, 1) 置换
    # torch.sqrt(torch.sum(torch.pow(x, 2), 1)).view(bs1, 1) 平方，按行求和，开根号
    # => (8,) view reshape => (8,1) repeat(1,8)表示第一维不变，第二维扩展为8
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down

    return cosine



def label2similarity(label1, label2):
    '''
    compute similarity matrix of label1 and label2
    :param label1: torch.Tensor, [m]
    :param label2: torch.Tensor, [n]
    :return: torch.Tensor, [m, n], {0, 1}
    '''
    m, n = len(label1), len(label2)
    l1 = label1.view(m, 1).expand([m, n])
    l2 = label2.view(n, 1).expand([n, m]).t()
    similarity = l1 == l2
    return similarity


def low_memory_matrix_op(
        func,
        x, y,
        x_split_axis, y_split_axis,
        x_num_splits, y_num_splits,
        verbose=False, aligned=True):
    """
  For matrix operation like multiplication, in order not to flood the memory
  with huge data, split matrices into smaller parts (Divide and Conquer).

  Note:
    If still out of memory, increase `*_num_splits`.

  Args:
    func: a matrix function func(x, y) -> z with shape [M, N]
    x: numpy array, the dimension to split has length M
    y: numpy array, the dimension to split has length N
    x_split_axis: The axis to split x into parts
    y_split_axis: The axis to split y into parts
    x_num_splits: number of splits. 1 <= x_num_splits <= M
    y_num_splits: number of splits. 1 <= y_num_splits <= N
    verbose: whether to print the progress

  Returns:
    mat: numpy array, shape [M, N]
  """

    if verbose:
        import sys
        import time
        printed = False
        st = time.time()
        last_time = time.time()

    mat = [[] for _ in range(x_num_splits)]
    for i, part_x in enumerate(
            np.array_split(x, x_num_splits, axis=x_split_axis)):
        for j, part_y in enumerate(
                np.array_split(y, y_num_splits, axis=y_split_axis)):
            part_mat = func(part_x, part_y, aligned)
            mat[i].append(part_mat)

            if verbose:
                if not printed:
                    printed = True
                else:
                    # Clean the current line
                    sys.stdout.write("\033[F\033[K")
                print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                    .format(i + 1, j + 1, x_num_splits, y_num_splits,
                            time.time() - last_time, time.time() - st))
                last_time = time.time()
        mat[i] = np.concatenate(mat[i], axis=1)
    mat = np.concatenate(mat, axis=0)
    return mat


def low_memory_local_dist(x, y, aligned=True):
    print('Computing local distance...')
    x_num_splits = int(len(x) / 200) + 1
    y_num_splits = int(len(y) / 200) + 1
    z = low_memory_matrix_op(local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True, aligned=aligned)
    return z





