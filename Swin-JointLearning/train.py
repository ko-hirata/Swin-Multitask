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
from torcheval.metrics import MulticlassConfusionMatrix
from torchvision import transforms
from tqdm import tqdm

from data.brainct_dataset import BrainctDataset
from data.brainct_dataset import RandomGenerator, SimpleTransform
from utils import label_counter, save_metrics_fig
from utils import calculate_confusion_matrix
from networks.vision_transformer import SwinTransformerJoint as SwinJoint
from config import get_config


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                    default='../data/brainCT', help='root dir for data')
parser.add_argument('--output_dir', type=str,
                    default='./work_dirs/brainCT', help='output dir')
parser.add_argument('--ckpt_path_rep', type=str, default=None, help='ckpt of representation model')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=128, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.004,
                    help='segmentation network learning rate')
parser.add_argument('--class_weight', type=bool, 
                    default=False, help ='use class weights when the data is unbalance')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--cfg_rep', type=str, required=True, metavar="FILE", help='path to config file for representation model learned by segmentation task', )

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
args.cfg = args.cfg_rep
config_rep = get_config(args)


def trainer_brainct(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log_train.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

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

    model = SwinJoint(config, config_rep, args.ckpt_path_rep, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_from(config)

    if args.class_weight:
        label0, label1 = label_counter(trainloader)
        weights = torch.FloatTensor([label1/len(db_train), label0/len(db_valid)]).cuda()
        ce_loss_cls = CrossEntropyLoss(weight=weights)
    else:
        ce_loss_cls = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    metric_matrix_train = MulticlassConfusionMatrix(num_classes=num_classes)
    metric_matrix_valid = MulticlassConfusionMatrix(num_classes=num_classes)
    metrics_dict = {'losses_train':0, 'accs_train':1, 'f1s_train':2,
                    'losses_valid':3, 'accs_valid':4, 'f1s_valid':5}
    metrics_list = [[] for i in range(len(metrics_dict))]

    for epoch_num in iterator:
        loss_train, loss_valid = 0, 0

        model.train()
        for i_batch, (image_batch, label_batch, _, _) in enumerate(trainloader):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs_cls = model(image_batch)
            
            loss_cls = ce_loss_cls(outputs_cls, label_batch[:].long())
            loss = loss_cls
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_train += loss.item() * image_batch.size(dim=0)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            metric_matrix_train.update(outputs_cls, label_batch)

            iter_num = iter_num + 1
            logging.info('iteration %d :  loss: %f' % (iter_num, loss.item()))
                
        model.eval()
        with torch.no_grad():
            for i_batch, (image_batch, label_batch, _, _) in enumerate(validloader):
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs_cls = model(image_batch)

                loss_cls = ce_loss_cls(outputs_cls, label_batch[:].long())
                loss = loss_cls
                loss_valid += loss.item() * image_batch.size(dim=0)

                metric_matrix_valid.update(outputs_cls, label_batch)

        cmat_train = metric_matrix_train.compute().tolist()
        metric_matrix_train.reset()
        acc_train, prec_train, rec_train, f1_train = calculate_confusion_matrix(cmat_train)

        cmat_valid = metric_matrix_valid.compute().tolist()
        metric_matrix_valid.reset()
        acc_valid, prec_valid, rec_valid, f1_valid = calculate_confusion_matrix(cmat_valid)

        loss_train /= len(db_train)
        loss_valid /= len(db_valid)
        
        metrics_list[metrics_dict['losses_train']].append(loss_train)
        metrics_list[metrics_dict['accs_train']].append(acc_train)
        metrics_list[metrics_dict['f1s_train']].append(f1_train)

        metrics_list[metrics_dict['losses_valid']].append(loss_valid)
        metrics_list[metrics_dict['accs_valid']].append(acc_valid)
        metrics_list[metrics_dict['f1s_valid']].append(f1_valid)

        logging.info('===========================================================================================')
        logging.info('epoch %d :  accuracy: %f, f1: %f' % (epoch_num, acc_valid, f1_valid))
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