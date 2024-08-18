import argparse
import logging
import numpy as np
import os
import random
import sys
import torch
import torch.backends.cudnn as cudnn

from sklearn import metrics
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from torcheval.metrics import MulticlassConfusionMatrix
from torchvision import transforms

from data.brainct_dataset import BrainctDataset
from data.brainct_dataset import SimpleTransform
from utils import DiceLoss, calculate_seg_loss
from utils import save_output_visualization
from utils import calculate_confusion_matrix
from networks.vision_transformer import MTLSwinUnet
from config import get_config


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                    default='../data/brainCT', help='root dir for data')
parser.add_argument('--output_dir', type=str,
                    default='./work_dirs/brainCT', help='output dir')
parser.add_argument('--ckpt_path', type=str, help='ckpt path')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--weight_cls', type=float, default=0.3, help='loss weight for classification')
parser.add_argument('--weight_seg', type=float, default=0.4, help='loss weight for semantic segmentation')
parser.add_argument('--weight_rec', type=float, default=0.4, help='loss weight for reconstruction')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args(args=['--root_dir', '/home/hirata/BRAIN/MTL-Swin-Unet/dataset_common',
                               '--output_dir', '/home/hirata/BRAIN/MTL-Swin-Unet/src/work_dirs/test',
                               '--ckpt_path', '/home/hirata/BRAIN/MTL-Swin-Unet/src/work_dirs/commondata_CSR_KFold_epoch600_batch64_lr0.02_lossw334_augRG/fold_0_epoch_599.pth',
                               '--num_classes', '2',
                               '--img_size', '224',
                               '--cfg', './configs/swin_tiny_patch4_window7_224_lite.yaml'])


config = get_config(args)


def inference(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log_test_3tasks.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    num_classes = args.num_classes
    weight_cls = args.weight_cls
    weight_seg = args.weight_seg
    weight_rec = args.weight_rec

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    db_test = BrainctDataset(base_dir=args.root_dir, split="test",
                              transform=transforms.Compose([SimpleTransform(output_size=[args.img_size, args.img_size])]))               
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False)
    print("The length of test set is: {}".format(len(db_test)))

    model = MTLSwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load(args.ckpt_path))

    ce_loss_cls = CrossEntropyLoss()
    ce_loss_seg = CrossEntropyLoss(reduction='none')
    dice_loss_seg = DiceLoss(num_classes)
    mse_loss_rec = torch.nn.MSELoss()

    metric_matrix_test = MulticlassConfusionMatrix(num_classes=num_classes)
    loss_test, iou_test = 0, 0
    mask_num = 0
    y_true = []
    y_prob = []

    model.eval()
    with torch.no_grad():
        for i_batch, (image_batch, label_batch, mask_batch, mask_exist_batch) in enumerate(testloader):
            image_batch, label_batch, mask_batch = image_batch.cuda(), label_batch.cuda(), mask_batch.cuda()
            outputs_cls, outputs_seg, outputs_rec = model(image_batch)

            loss_ce_cls = ce_loss_cls(outputs_cls, label_batch[:].long())
            loss_cls = loss_ce_cls
            loss_ce_seg = ce_loss_seg(outputs_seg, mask_batch[:].long())
            loss_dice_seg = dice_loss_seg(outputs_seg, mask_batch, softmax=True)
            image_batch_size = image_batch.size(dim=0)
            loss_seg = calculate_seg_loss(loss_ce_seg, loss_dice_seg, mask_exist_batch, image_batch_size)
            loss_rec = mse_loss_rec(outputs_rec, image_batch)
            loss = weight_cls * loss_cls + weight_seg * loss_seg + weight_rec * loss_rec
            loss_test += loss.item() * image_batch.size(dim=0)

            metric_matrix_test.update(outputs_cls, label_batch)
            p = torch.softmax(outputs_cls, dim=1).detach().cpu().numpy()
            y_true.append(label_batch[0].detach().cpu().numpy())
            y_prob.append(p[0][1])
            if mask_exist_batch[0] == True:
                y = torch.argmax(torch.softmax(outputs_seg, dim=1), dim=1).squeeze(0)
                y = y.to('cpu').detach()
                gt = mask_batch.to('cpu').detach().squeeze(0)
                iou = jaccard_index(y, gt, task="binary")
                iou_test += iou.item()
                mask_num += 1

            label = label_batch.squeeze().to('cpu').detach().numpy().copy()
            predict = torch.argmax(outputs_cls).squeeze().to('cpu').detach().numpy().copy()
            save_dir = os.path.join(snapshot_path, f'inference_3tasks/{label}{predict}')
            save_output_visualization(image_batch, mask_batch, outputs_seg, outputs_rec, save_dir, i_batch)

    cmat_test = metric_matrix_test.compute().tolist()
    metric_matrix_test.reset()
    acc_test, prec_test, rec_test, f1_test = calculate_confusion_matrix(cmat_test)

    auc_test = metrics.roc_auc_score(y_true, y_prob)

    iou_test /= mask_num
    loss_test /= len(db_test)

    logging.info('================================')
    logging.info('{:<4} {:<4}'.format(int(cmat_test[0][0]), int(cmat_test[0][1])))
    logging.info('{:<4} {:<4}'.format(int(cmat_test[1][0]), int(cmat_test[1][1])))
    logging.info(f'loss      : {loss_test}')
    logging.info(f'accuracy  : {acc_test}')
    logging.info(f'precision : {prec_test}')
    logging.info(f'recall    : {rec_test}')
    logging.info(f'f1-score  : {f1_test}')
    logging.info(f'auc       : {auc_test}')
    logging.info(f'iou seg   : {iou_test}')
    logging.info('================================')

    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    inference(args, args.output_dir)