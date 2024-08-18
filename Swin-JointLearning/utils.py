import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from medpy import metric
from scipy.ndimage import zoom



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target, dim=(1,2))
        y_sum = torch.sum(target * target, dim=(1,2))
        z_sum = torch.sum(score * score, dim=(1,2))
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        #class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            #class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def label_counter(dataloader):
    label0, label1 = 0, 0
    for _, label_batch, _, _ in dataloader:
        for label in label_batch:
            if label == 0:
                label0 += 1
            else:
                label1 += 1
    return label0, label1
    

def calculate_confusion_matrix(cmat):
    TN, FP, FN, TP = cmat[0][0], cmat[0][1], cmat[1][0], cmat[1][1]
    f1_c1 = 2 * TP / (2 * TP + FP + FN)
    f1_c2 = 2 * TN / (2 * TN + FN + FP)

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (f1_c1 + f1_c2) / 2

    return accuracy, precision, recall, f1


def save_metrics_fig(metrics_list, metrics_dict, save_path):
    plt.rcParams["figure.figsize"] = (19.2, 4.8)

    plt.subplot(1,3,1)
    plt.plot(metrics_list[metrics_dict['losses_train']], label='train')
    plt.plot(metrics_list[metrics_dict['losses_valid']], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(metrics_list[metrics_dict['accs_train']], label='train')
    plt.plot(metrics_list[metrics_dict['accs_valid']], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(metrics_list[metrics_dict['f1s_train']], label='train')
    plt.plot(metrics_list[metrics_dict['f1s_valid']], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('f1_score')
    plt.title('F1_score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)