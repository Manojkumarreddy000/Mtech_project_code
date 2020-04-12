import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class ModifiedMarginLoss(nn.Module):

    def __init__(self, alpha=0.1, gpu=True):
        super(ModifiedMarginLoss, self).__init__()
        self.alpha = alpha
        self.beta = nn.Parameter(torch.tensor([0.5], dtype=torch.float))  
        self.gpu = gpu

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())

        loss = torch.tensor([0], dtype=torch.float)
        if self.gpu:
            loss= loss.cuda()
        
        for i in range(n):
            pos_violation = F.relu(self.alpha + self.beta - torch.masked_select(sim_mat[i], targets==targets[i]))
            neg_violation = F.relu(torch.masked_select(sim_mat[i], targets!=targets[i]) - self.beta + self.alpha)

            pos_denom = pos_violation.sum() + 1e-5
            neg_denom = neg_violation.sum() + 1e-5

            pos_prob = pos_violation / pos_denom
            neg_prob = neg_violation / neg_denom

            pos_prob, pindex = torch.sort(pos_prob, descending=True)
            neg_prob, nindex = torch.sort(neg_prob, descending=True)

            if pos_prob[0] > 0.5:
                loss += pos_violation[0]
            if neg_prob[0] > 0.5:
                loss += neg_violation[0] 

        return loss

class LiftedStructureLoss(nn.Module):
    def __init__(self, hard_mining=None,  **kwargs):
        super(LiftedStructureLoss, self).__init__()         
        self.alpha = nn.Parameter(torch.tensor([9.0], dtype=torch.float))
        #self.beta = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self, score, target):
        loss = 0.
        counter = 0
        target =target.squeeze()
        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))
    
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = F.relu(dist).sqrt()
    
        for i in range(bsz):
            t_i = target[i]
        
            for j in range(i + 1, bsz):
                t_j = target[j]
            
                if t_i == t_j:
                    # Negative component
                    # !! Could do other things (like softmax that weights closer negatives)
                    l_ni = (self.alpha - dist[i][target != t_i]).exp().sum()
                    l_nj = (self.alpha - dist[j][target != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                
                    # Positive component
                    l_p  = dist[i,j]
                
                    loss += F.relu(l_n + l_p) ** 2
                    counter += 1
    
        return loss / (2 * counter)

class DopplegangerLoss(nn.Module):
    def __init__(self):
        super(DopplegangerLoss, self).__init__()
        self.marginloss = ModifiedMarginLoss()
        self.liftedloss = LiftedStructureLoss()
        self.stdloss = nn.NLLLoss()        

    def forward(self, scores, embeddings, targets):
        stdloss = self.stdloss(scores, targets)
        liftedloss = self.liftedloss(embeddings, targets)
        marginloss = self.marginloss(embeddings, targets)
        return stdloss + 0.1*liftedloss + marginloss


def main():
    #breakpoint()
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    x = torch.rand(data_size, input_dim).cuda()
    w = torch.rand((input_dim, output_dim), requires_grad=True).cuda()
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = torch.IntTensor(y_).cuda()    
    loss_fn = ModifiedMarginLoss().cuda()
    print(loss_fn(inputs, targets))


if __name__ == '__main__':
    main()
