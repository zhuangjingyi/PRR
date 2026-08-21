import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


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
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False, s_pos=3, s_neg=5):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_pos.fill_diagonal_(False)
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # print('***********************************')
    # print("s_pos:", s_pos)
    # print("s_neg:", s_neg)
    all_ap = dist_mat[is_pos].contiguous().view(N, -1)
    all_ap_sort, _ = torch.sort(all_ap, dim=-1, descending=True)
    all_ap_sort = all_ap_sort[:, 0:s_pos]
    dist_ap = torch.mean(all_ap_sort,dim=-1)

    all_an = dist_mat[is_neg].contiguous().view(N, -1)
    all_an_sort,_ = torch.sort(all_an,dim=-1,descending=False)
    all_an_sort = all_an_sort[:,0:s_neg]
    dist_an = torch.mean(all_an_sort,dim=-1)


    return dist_ap, dist_an


class PRR_TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0, s_pos=3, s_neg=5):
        self.margin = margin
        self.hard_factor = hard_factor
        self.s_pos = s_pos
        self.s_neg = s_neg    #dynamic时唯一参数

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels,s_pos=self.s_pos, s_neg=self.s_neg)
        # dist_ap, dist_an, num = hard_example_mining(dist_mat, labels,s_pos=self.s_pos, s_neg=self.s_neg)    #dynamic

        #  dist_ap *= (1.0 + self.hard_factor)
        #  dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            # print('******')
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            # print('......')

            # print(dist_an.view(1,-1) - dist_ap.view(1,-1))
            score = dist_an - dist_ap
            # score = torch.clamp(score, -50, 50)
            loss = self.ranking_loss(score, y)


        return loss, dist_ap, dist_an


# #AGWTripletLoss
def AGWeuclidean_dist(x, y):
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
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


class AGWTripletLoss(object):

    def __init__(self):
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = AGWeuclidean_dist(global_feat, global_feat)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss
# def hard_example_mining(dist_mat, labels, return_inds=False):
#     """For each anchor, find the hardest positive and negative sample.
#     Args:
#       dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#       labels: pytorch LongTensor, with shape [N]
#       return_inds: whether to return the indices. Save time if `False`(?)
#     Returns:
#       dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#       dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#       p_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#       n_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     NOTE: Only consider the case in which all labels have same num of samples,
#       thus we can cope with all anchors in parallel.
#     """

#     assert len(dist_mat.size()) == 2
#     assert dist_mat.size(0) == dist_mat.size(1)
#     N = dist_mat.size(0)

#     # shape [N, N]
#     is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#     is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

#     # `dist_ap` means distance(anchor, positive)
#     # both `dist_ap` and `relative_p_inds` with shape [N, 1]
#     dist_ap, relative_p_inds = torch.max(
#         dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
#     # `dist_an` means distance(anchor, negative)
#     # both `dist_an` and `relative_n_inds` with shape [N, 1]
#     dist_an, relative_n_inds = torch.min(
#         dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#     # shape [N]
#     dist_ap = dist_ap.squeeze(1)
#     dist_an = dist_an.squeeze(1)

#     if return_inds:
#         # shape [N, N]
#         ind = (labels.new().resize_as_(labels)
#                .copy_(torch.arange(0, N).long())
#                .unsqueeze(0).expand(N, N))
#         # shape [N, 1]
#         p_inds = torch.gather(
#             ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
#         n_inds = torch.gather(
#             ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
#         # shape [N]
#         p_inds = p_inds.squeeze(1)
#         n_inds = n_inds.squeeze(1)
#         return dist_ap, dist_an, p_inds, n_inds

#     return dist_ap, dist_an

# class TripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""

#     def __init__(self, margin=None):
#         self.margin = margin
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()

#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an


#RobustTripletLoss from 26CVPR_Robust Remote Sensing Image–Text Retrieval with Noisy Correspondence
class RRSITRRobustTripletLoss(nn.Module):
    """
    ReID adaptation of the robust triplet term used in RRSITR.

    RRSITR computes a semantic-similarity-aware soft margin as

        sigma_i = base_sigma * (1 + relu(hard_neg_sim - pos_sim))

    and then applies25he

        loss_i = relu(sigma_i - pos_sim + hard_neg_sim).

    The original RRSITR setting has one paired positive on the diagonal of an
    image-text similarity matrix. In person ReID, an identity normally has
    multiple positives in a PK-sampled batch. Therefore, for each anchor we use:
      - hardest positive: the same-ID sample with the LOWEST cosine similarity;
      - hardest negative: the different-ID sample with the HIGHEST cosine similarity.

    This keeps the RRSITR adaptive-margin mechanism while making the positive
    definition compatible with label-based ReID batches.
    """

    def __init__(self, base_sigma= 1.3, normalize_feature=False):
        super().__init__()

        self.base_sigma = float(base_sigma)
        self.normalize_feature = normalize_feature
        if self.normalize_feature:
            assert self.base_sigma < 1


        print('...........', self.base_sigma, self.normalize_feature)


    def forward(self, global_feat, labels):


        if self.normalize_feature:
            feat = F.normalize(global_feat, p=2, dim=1)
        else:
            # print('not do norm')
            feat = global_feat

        dists = euclidean_dist(feat,feat)

        n = dists.size(0)
        same_id = labels.view(n, 1).eq(labels.view(1, n))
        eye = torch.eye(n, dtype=torch.bool, device=dists.device)
        pos_mask = same_id & ~eye
        neg_mask = ~same_id


        pos_dist = dists[pos_mask].contiguous().view(n, -1)
        neg_dist = dists[neg_mask].contiguous().view(n, -1)
        hardest_neg_dist = neg_dist.min(dim=-1,keepdim=True).values

        score = hardest_neg_dist - pos_dist
        adaptive_sigma = self.base_sigma * (1.0 + F.relu(-score))
        per_anchor_loss = F.relu(adaptive_sigma - score)
        loss = per_anchor_loss.mean()

        return loss, pos_dist, neg_dist
