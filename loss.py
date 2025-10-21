import torch
import torch.nn.functional as F
import torch.nn as nn
import ipdb

def orthogonal_loss(B, G):
    # 计算 B 和 G 的点积
    dot_product = torch.sum(B * G, dim=-1)
    
    # 计算 B 和 G 的范数
    B_norm = torch.norm(B, p=2, dim=-1)
    G_norm = torch.norm(G, p=2, dim=-1)
    
    # 计算余弦相似度
    cos_similarity = dot_product / (B_norm * G_norm + 1e-8)  # 加上一个小常数以避免除零
    
    # 计算 L1 损失，使余弦相似度接近零
    loss = F.l1_loss(cos_similarity, torch.zeros_like(cos_similarity))
    
    return loss

def negative_info_nce_loss(x, y, temperature=0.1):
    # x 和 y 是两组特征向量
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    # 计算相似性矩阵
    similarity_matrix = torch.matmul(x, y.T) / temperature  # [batch_size, batch_size]
    
    # 排除对角线元素（正样本对）
    batch_size = similarity_matrix.size(0)
    mask = torch.eye(batch_size, device=x.device).bool()
    negative_samples = similarity_matrix[~mask].view(batch_size, -1)  # 非对角线元素，即负样本对
    
    # 将负样本相似性最小化
    loss = negative_samples.mean()  # 越小越好，表示负样本的相似性越小
    return loss




def parallel_loss(F, prototypes, correct_class_idx):
    """
    实现前景特征 F 与对应类别原型 P_k 平行，同时与其他类别原型 P_i 正交的损失函数。
    
    参数:
    - F: 前景特征向量，形状为 (batch_size, feature_dim)
    - prototypes: 类别原型矩阵，形状为 (num_classes, feature_dim)
    - correct_class_idx: 正确类别的索引，整数或形状为 (batch_size,)
    
    返回:
    - 总损失值
    """
    # 计算 F 与所有类别原型的余弦相似度
    F_norm = torch.norm(F, p=2, dim=-1, keepdim=True)
    prototypes_norm = torch.norm(prototypes, p=2, dim=-1, keepdim=True)
    cos_similarities = (F @ prototypes.T) / (F_norm * prototypes_norm.T + 1e-8)
    
    # 构造与正确类别和其他类别相关的目标值
    batch_size = F.size(0)
    num_classes = prototypes.size(0)
    targets = torch.zeros_like(cos_similarities)
    targets[torch.arange(batch_size), correct_class_idx] = 1  # 正确类别的相似度应该为 1
    
    # 计算 L1 损失
    loss = F.l1_loss(cos_similarities, targets, reduction='mean')
    
    return loss


def PCLoss(features, f_labels, prototypes, t=0.5):
#     ipdb.set_trace()
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0,1)) / t)
    
    pos_sim = torch.exp(torch.diag(torch.mm(a_norm, b_norm[f_labels].transpose(0,1))) / t)
    
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    return loss

class PrototypeContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, prototypes):
        """
        :param features: tensor of shape (batch_size, feature_dim), the features of the samples
        :param labels: tensor of shape (batch_size,), the ground truth labels of the samples
        :param prototypes: tensor of shape (num_classes, feature_dim), the prototype representations of each class
        :return: scalar loss value
        """
        # Normalize features and prototypes
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        # Compute pairwise cosine similarity
        sim_matrix = torch.matmul(features, prototypes.T) / self.temperature  # cosine similarity with temperature

        # Get the target labels in one-hot encoding
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target_sim = torch.gather(sim_matrix, 1, labels)  # similarity to the correct prototype

        # Calculate the contrastive loss
        loss = -torch.log(torch.exp(target_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
        
        return loss.mean()
    
    
class KDLoss(nn.Module):
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits):   
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T, -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()    

        return kl_loss

    
def col_loss(linear_output_1, linear_output_2, args):
    """
    计算两个模型输出的 Col Loss，包括对角和非对角项的损失。
    
    Args:
        linear_output_1: 第一个模型的预测输出 (batch_size, feature_dim)
        linear_output_2: 第二个模型的预测输出 (batch_size, feature_dim)
        off_diag_weight: 非对角项损失的权重，默认 0.005
    
    Returns:
        col_loss: 计算得到的 Col Loss
    """
    # 对两个模型的输出进行标准化 (BatchNorm-like 操作)
    z_1_bn = (linear_output_1 - linear_output_1.mean(0)) / linear_output_1.std(0)
    z_2_bn = (linear_output_2 - linear_output_2.mean(0)) / linear_output_2.std(0)

    # 计算相关性矩阵 C
    c = z_1_bn.T @ z_2_bn
    c.div_(linear_output_1.size(0))  # 归一化 (除以 batch size)

    # 计算对角项损失：目标为 1
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

    # 计算非对角项损失：目标为 0
    off_diag = _off_diagonal(c).add_(1).pow_(2).sum()

    # 总损失
    loss = on_diag + args.yeta * off_diag
    return loss


def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_similarity_matrix(features):
    """
    计算特征的相似性矩阵，去除自身相似性。
    Args:
        features: 输入特征矩阵 (batch_size, feature_dim)
    Returns:
        相似性矩阵 S_i，去除自身相似性 (batch_size, batch_size-1)
    """
    # 归一化特征向量
    features = F.normalize(features, p=2, dim=1)
    # 计算相似性矩阵 (batch_size, batch_size)
    similarity_matrix = torch.mm(features, features.T)  # 余弦相似度
    # 创建一个掩码去除自身相似性
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)  # 自身相似性置零
    return similarity_matrix


def fisl_loss(local_features_list):
    """
    计算 Federated Instance Similarity Learning (FISL) 损失。
    Args:
        local_features_list: 每个客户端的特征输出列表 [features_1, features_2, ..., features_K]，
                             每个 features 的形状是 (batch_size, feature_dim)
    Returns:
        FISL 损失值
    """
    similarity_matrices = []
    
    # 计算每个客户端的相似性矩阵 S_i
    for features in local_features_list:
        S_i = compute_similarity_matrix(features)
        similarity_matrices.append(S_i)

    
    # 计算 KL 散度损失

    kl_div = F.kl_div(
        F.log_softmax(similarity_matrices[0], dim=1), 
        F.softmax(similarity_matrices[1], dim=1), 
        reduction='batchmean'
    )

    return kl_div


class RKdAngle(nn.Module):
    def forward(self, student, teacher, args):
        # N x C
        # N x N x C
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        
        if args.mode_angle == 'L2':
            loss = F.mse_loss(s_angle, t_angle, reduction='mean')
        elif args.mode_angle == 'L1':
            loss = F.l1_loss(s_angle, t_angle, reduction='mean')
        elif args.mode_angle == 'smooth_l1':          
            loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        
        return loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res    



def js_divergence(p, q):
    kl_loss = KDLoss(T=0.5).cuda()
    #     ipdb.set_trace()
    half = torch.div(p + q, 2)
    s1 = kl_loss(p, half)
    s2 = kl_loss(q, half)
    #     ipdb.set_trace()
    return torch.div(s1 + s2, 2)

class RkdDistance(nn.Module):
    def forward(self, student, teacher, args):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
#         ipdb.set_trace()
        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        if args.mode_dis == 'KL':
            KL_loss = KDLoss(T=0.5).to(args.device)   
            loss = KL_loss(d, t_d)
        elif args.mode_dis == 'L2':
            loss = F.mse_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'L1':
            loss = F.l1_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'smooth_l1':
            loss = F.smooth_l1_loss(d, t_d, reduction='mean')            
        elif args.mode_dis == 'JS':
            loss = js_divergence(d, t_d)    
            
        return loss


class Relation_Loss(nn.Module):
    def __init__(self, dist_weight=1, angle_weight=1):
        super(Relation_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dist_weight = dist_weight
        self.angle_weight = angle_weight       
    
    def forward(self, student, teacher, args):    
        
        dis_loss = self.dist_criterion(student, teacher, args)
        angle_loss = self.angle_criterion(student, teacher, args)
        relational_loss = self.dist_weight * dis_loss + self.angle_weight * angle_loss
        
        return relational_loss     

    
    
def prototype_similarity_loss(X, P, target, loss_type="mse"):
    """
    计算基于类别原型的特征相似性损失
    
    参数：
    - X: (batch_size, feature_dim)  -> Batch 特征
    - P: (num_classes, feature_dim) -> 类别原型矩阵
    - target: (batch_size,) -> 每个样本的类别标签
    - loss_type: "mse" 或 "kl" -> 选择损失函数类型
    
    返回：
    - loss: 约束特征学习的损失
    """

    # 归一化类别原型矩阵
    P = P / P.norm(dim=1, keepdim=True)  # (num_classes, feature_dim)

    # 计算类别原型间的相似性矩阵 (num_classes, num_classes)
    S = torch.mm(P, P.T)  # S[i, j] 表示类别 i 和类别 j 的相似性

    # 归一化 batch 特征
    X = X / X.norm(dim=1, keepdim=True)  # (batch_size, feature_dim)

    # 计算 batch 内特征相似性矩阵 (batch_size, batch_size)
    X_sim = torch.mm(X, X.T)  # X_sim[i, j] 表示样本 i 和 j 的特征相似性

    # 获取 batch 内每个样本的类别索引
    c_i = target.unsqueeze(1)  # (batch_size, 1)
    c_j = target.unsqueeze(0)  # (1, batch_size)

    # 通过索引取出类别相似性矩阵 (batch_size, batch_size)
    S_batch = S[c_i, c_j]  # S_batch[i, j] 是样本 i 和样本 j 的类别原型相似性

    # 计算损失
    if loss_type == "mse":
        loss = F.mse_loss(X_sim, S_batch)
    elif loss_type == "kl":
        loss = F.kl_div(F.log_softmax(X_sim, dim=-1), F.softmax(S_batch, dim=-1), reduction='batchmean')
    else:
        raise ValueError("loss_type should be 'mse' or 'kl'")

    return loss


