import argparse
import logging
import numpy as np
import os
import random
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from torcheval.metrics import MulticlassConfusionMatrix
from torchvision import transforms
from tqdm import tqdm

from data.brainct_dataset import BrainctDataset
from data.brainct_dataset import RandomGenerator, SimpleTransform
from utils import DiceLoss, calculate_seg_loss
from utils import label_counter, save_metrics_fig
from utils import calculate_confusion_matrix
from networks.vision_transformer import MTLSwinUnet
from config import get_config


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                    default='../data/brainCT', help='root dir for data')
parser.add_argument('--output_dir', type=str,
                    default='./work_dirs/brainCT', help='output dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.02,
                    help='segmentation network learning rate')
parser.add_argument('--weight_cls', type=float, default=0.3, help='loss weight for classification')
parser.add_argument('--weight_seg', type=float, default=0.4, help='loss weight for semantic segmentation')
parser.add_argument('--weight_rec', type=float, default=0.4, help='loss weight for reconstruction')
parser.add_argument('--class_weight', type=bool, default=False, help ='use class weights when the data is unbalance')
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

args = parser.parse_args()


config = get_config(args)


def trainer_brainct(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log_train.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    weight_cls = args.weight_cls
    weight_seg = args.weight_seg
    weight_rec = args.weight_rec

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    db_train = BrainctDataset(base_dir=args.root_dir, split="train",
                              transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_valid = BrainctDataset(base_dir=args.root_dir, split="valid",
                              transform=transforms.Compose([SimpleTransform(output_size=[args.img_size, args.img_size])]))               
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    validloader = DataLoader(db_valid, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False)
    print("The length of train set is: {}".format(len(db_train)))

    model = MTLSwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_from(config)

    if args.class_weight:
        label0, label1 = label_counter(trainloader)
        weights = torch.FloatTensor([label1/len(db_train), label0/len(db_valid)]).cuda()
        ce_loss_cls = CrossEntropyLoss(weight=weights)
    else:
        ce_loss_cls = CrossEntropyLoss()
    ce_loss_seg = CrossEntropyLoss(reduction='none')
    dice_loss_seg = DiceLoss(num_classes)
    mse_loss_rec = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    metric_matrix_train = MulticlassConfusionMatrix(num_classes=num_classes)
    metric_matrix_valid = MulticlassConfusionMatrix(num_classes=num_classes)
    metrics_dict = {'losses_train':0, 'accs_train':1, 'ious_train':2, 'f1s_train':3,
                    'losses_valid':4, 'accs_valid':5, 'ious_valid':6, 'f1s_valid':7}
    metrics_list = [[] for i in range(len(metrics_dict))]

    for epoch_num in iterator:
        loss_train, iou_train = 0, 0
        loss_valid, iou_valid = 0, 0

        model.train()
        for i_batch, (image_batch, label_batch, mask_batch, mask_exist_batch) in enumerate(trainloader):
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
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_train += loss.item() * image_batch.size(dim=0)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            metric_matrix_train.update(outputs_cls, label_batch)
            y = torch.argmax(torch.softmax(outputs_seg, dim=1), dim=1).squeeze(0)
            y = y.to('cpu').detach()
            gt = mask_batch.to('cpu').detach().squeeze(0)
            iou = jaccard_index(y, gt, task="binary")
            iou_train += iou.item() * image_batch.size(dim=0)

            iter_num = iter_num + 1
            logging.info('iteration %d :  loss: %f, loss_cls: %f, loss_seg: %f, loss_mse_rec: %f' % (iter_num, loss.item(), loss_cls.item(), loss_seg.item(), loss_rec.item()))
                
        model.eval()
        with torch.no_grad():
            for i_batch, (image_batch, label_batch, mask_batch, mask_exist_batch) in enumerate(validloader):
                image_batch, label_batch, mask_batch = image_batch.cuda(), label_batch.cuda(), mask_batch.cuda()
                outputs_cls, outputs_seg, outputs_rec = model(image_batch)

                loss_ce_cls = ce_loss_cls(outputs_cls, label_batch[:].long())
                loss_cls = loss_ce_cls
                loss_ce_seg = ce_loss_seg(outputs_seg, mask_batch[:].long())
                loss_dice_seg = dice_loss_seg(outputs_seg, mask_batch, softmax=True)
                image_batch_size = image_batch.size(dim=0)
                loss_seg = calculate_seg_loss(loss_ce_seg, loss_dice_seg, mask_exist_batch, image_batch_size)
                loss_rec = mse_loss_rec(outputs_rec, image_batch)
                loss = args.weight_cls * loss_cls + args.weight_seg * loss_seg + args.weight_rec * loss_rec
                loss_valid += loss.item() * image_batch.size(dim=0)

                metric_matrix_valid.update(outputs_cls, label_batch)
                y = torch.argmax(torch.softmax(outputs_seg, dim=1), dim=1).squeeze(0)
                y = y.to('cpu').detach()
                gt = mask_batch.to('cpu').detach().squeeze(0)
                iou = jaccard_index(y, gt, task="binary")
                iou_valid += iou.item() * image_batch.size(dim=0)

        cmat_train = metric_matrix_train.compute().tolist()
        metric_matrix_train.reset()
        acc_train, prec_train, rec_train, f1_train = calculate_confusion_matrix(cmat_train)

        cmat_valid = metric_matrix_valid.compute().tolist()
        metric_matrix_valid.reset()
        acc_valid, prec_valid, rec_valid, f1_valid = calculate_confusion_matrix(cmat_valid)

        loss_train /= len(db_train)
        iou_train /= len(db_train)

        loss_valid /= len(db_valid)
        iou_valid /= len(db_valid)
        
        metrics_list[metrics_dict['losses_train']].append(loss_train)
        metrics_list[metrics_dict['accs_train']].append(acc_train)
        metrics_list[metrics_dict['ious_train']].append(iou_train)
        metrics_list[metrics_dict['f1s_train']].append(f1_train)

        metrics_list[metrics_dict['losses_valid']].append(loss_valid)
        metrics_list[metrics_dict['accs_valid']].append(acc_valid)
        metrics_list[metrics_dict['ious_valid']].append(iou_valid)
        metrics_list[metrics_dict['f1s_valid']].append(f1_valid)

        logging.info('===========================================================================================')
        logging.info('epoch %d :  accuracy: %f, f1: %f, iou: %f' % (epoch_num, acc_valid, f1_valid, iou_valid))
        logging.info('{:<4} {:<4}'.format(int(cmat_valid[0][0]), int(cmat_valid[0][1])))
        logging.info('{:<4} {:<4}'.format(int(cmat_valid[1][0]), int(cmat_valid[1][1])))
        logging.info('===========================================================================================')

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    save_metrics_fig(metrics_list, metrics_dict, os.path.join(snapshot_path, 'learning_curve.png'))

    return "Training Finished!"


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

    trainer_brainct(args, args.output_dir)