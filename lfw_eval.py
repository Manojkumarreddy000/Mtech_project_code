import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os

from torchvision.transforms import functional as F
from PIL import Image

from networks import EmbeddingNet, ClassificationNet

cudnn.benchmark = True

def extractDeepFeature(img, model, is_gray):
    width = 144
    height = 144
    img = img.resize((width, height))
    img_flipped = F.hflip(img)
    img = torch.from_numpy((np.array(img) - 127.5) / 128.0).permute(2, 0, 1).float()
    img_ = torch.from_numpy((np.array(img_flipped) - 127.5) / 128.0).permute(2, 0, 1).float()
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    _, ft1 = model(img)
    _, ft2 = model(img_)
    #ft1 = ft1 / ft1.pow(2).sum(1, keepdim=True).sqrt() #L2 normalization
    #ft2 = ft2 / ft2.pow(2).sum(1, keepdim=True).sqrt() #L2 normalization
    ft = torch.cat((ft1, ft2), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        d = d.split('\t')
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model, model_path=None, is_gray=False):
    predicts = []
    model_dict = torch.load(model_path)
    #del model_dict['model_state_dict']['module.fc1.weight']   
    #del model_dict['model_state_dict']['module.fc1.bias']
    model.load_state_dict(model_dict['model_state_dict'])#, strict=False)
    model.eval()
    root = '/home/data/DB/LFW/images/'
    with open('/home/data/DB/LFW/lfw_pairs.txt') as f:
        pairs_lines = f.readlines()[:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split(' ')
            name1= p[0]
            name2 = p[1]
            sameflag = int(p[2])
            
            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')

            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    #predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    predicts = np.array(predicts)
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    #breakpoint()
    mining_tech = 'RandomSampling_l2_parametrized'
    
    model_eval = nn.DataParallel(ClassificationNet(EmbeddingNet(), 854)).to('cuda')

    _, result = eval(model_eval, model_path=os.path.join('checkpoints', mining_tech, 'best_model.pth'))

    np.savetxt(os.path.join('results', mining_tech, 'result.txt'), result, '%s')
