'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * Changed from SLIP
 * https://github.com/facebookresearch/SLIP
 * By Le Xue
'''
import argparse
from collections import OrderedDict
import math
import time
import wandb
import open_clip
from itertools import cycle

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.dataset_3d import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn, view_customized_collate_fn

# torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='s3dis', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='s3dis', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
    parser.add_argument('--clip_model', default='hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', type=str)
    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=1, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    return parser

best_acc1 = 0

def main(args):
    # utils.init_distributed_mode(args)
    args.distributed = False

    global best_acc1

    if utils.is_main_process() and args.wandb:
        # wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='ULIP', config=args, reinit=True, entity='wangfatho')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # if args.evaluate_3d:
    #     zero_stats = test_zeroshot_3d(args)
    #     print(zero_stats)
    #     return

    # create model
    print("=> creating CLIP model {}".format(args.clip_model))
    clip_model, clip_preprocessor, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')

    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args, clip_model=clip_model)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)


    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    # 設定哪一些參數需要 weight decay
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            # print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            # best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            # best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    s3dis_entity_train_dataset = get_dataset(clip_preprocessor, tokenizer, args, 's3dis', "entity")
    s3dis_view_train_dataset = get_dataset(clip_preprocessor, tokenizer, args, 's3dis', "view")
    s3dis_scene_train_dataset = get_dataset(clip_preprocessor, tokenizer, args, 's3dis', "scene")
    scannet_entity_train_dataset = get_dataset(clip_preprocessor, tokenizer, args, 'scannet', "entity")
    scannet_view_train_dataset = get_dataset(clip_preprocessor, tokenizer, args, 'scannet', "view")
    scannet_scene_train_dataset = get_dataset(clip_preprocessor, tokenizer, args, 'scannet', "scene")

    s3dis_entity_train_sampler = None
    s3dis_view_train_sampler = None
    s3dis_scene_train_sampler = None
    scannet_entity_train_sampler = None
    scannet_view_train_sampler = None
    scannet_scene_train_sampler = None
    # val_sampler = None

    s3dis_entity_train_loader = torch.utils.data.DataLoader(
        s3dis_entity_train_dataset, batch_size=args.batch_size, shuffle=(s3dis_entity_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=s3dis_entity_train_sampler, drop_last=True,
    )
    s3dis_view_train_loader = torch.utils.data.DataLoader(
        s3dis_view_train_dataset, batch_size=args.batch_size, shuffle=(s3dis_view_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=s3dis_view_train_sampler, drop_last=True,
    )
    s3dis_scene_train_loader = torch.utils.data.DataLoader(
        s3dis_scene_train_dataset, batch_size=args.batch_size, shuffle=(s3dis_scene_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=s3dis_scene_train_sampler, drop_last=True,
    )
    scannet_entity_train_loader = torch.utils.data.DataLoader(
        scannet_entity_train_dataset, batch_size=args.batch_size, shuffle=(scannet_entity_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=scannet_entity_train_sampler, drop_last=True,
    )
    scannet_view_train_loader = torch.utils.data.DataLoader(
        scannet_view_train_dataset, batch_size=args.batch_size, shuffle=(scannet_view_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=scannet_view_train_sampler, drop_last=True,
    )
    scannet_scene_train_loader = torch.utils.data.DataLoader(
        scannet_scene_train_dataset, batch_size=args.batch_size, shuffle=(scannet_scene_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=scannet_scene_train_sampler, drop_last=True,
    )
    
    
    class CustomDataLoader:
        def __init__(self, dataloaders):
            self.dataloaders = dataloaders
            self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
            self.current_dataloader_index = -1
            self.cycle_count = 0
            self.exhausted_dataloaders = set()
        
        def select_dataloader(self):
            if self.current_dataloader_index in self.exhausted_dataloaders or self.cycle_count == 0:
                available_dataloaders = [i for i in range(len(self.dataloaders)) if i not in self.exhausted_dataloaders]
                if not available_dataloaders:
                    return -1
                self.current_dataloader_index = random.choice(available_dataloaders)
                self.cycle_count = 4
            return self.dataloader_iters[self.current_dataloader_index]
        
        def mark_dataloader_exhausted(self):
            self.exhausted_dataloaders.add(self.current_dataloader_index)
        
        def get_data(self):
            dataloader_iter = self.select_dataloader()
            if dataloader_iter == -1:
                return []
            batch_list = []
            try:
                for _ in range(4):
                    batch_list.append(next(dataloader_iter))
                    self.cycle_count -= 1
            except StopIteration:
                self.mark_dataloader_exhausted()
                return self.get_data()  # 如果 dataloader 用完，重新獲取數據
            return batch_list

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        (len(s3dis_entity_train_loader) + len(s3dis_view_train_loader) + len(s3dis_scene_train_loader) + len(scannet_entity_train_loader) + len(scannet_view_train_loader) + len(scannet_scene_train_loader)) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)

    best_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):

        all_in_one_train_loader = CustomDataLoader([s3dis_entity_train_loader, s3dis_view_train_loader, s3dis_scene_train_loader, scannet_entity_train_loader, scannet_view_train_loader, scannet_scene_train_loader])
        print("=> beginning all_in_one training")
        all_in_one_train_stats = train(all_in_one_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        # print("=> beginning s3dis_entity training")
        # s3dis_entity_train_stats = train(s3dis_entity_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 'entity', args)
        # print("=> beginning scannet_entity training")
        # scannet_entity_train_stats = train(scannet_entity_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 'entity', args)
        # print("=> beginning s3dis_view training")
        # s3dis_view_train_stats = train(s3dis_view_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 'view', args)
        # print("=> beginning scannet_view training")
        # scannet_view_train_stats = train(scannet_view_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 'view', args)
        # print("=> beginning s3dis_scene training")
        # s3dis_scene_train_stats = train(s3dis_scene_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 'scene', args)
        # print("=> beginning scannet_scene training")
        # scannet_scene_train_stats = train(scannet_scene_train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 'scene', args)

        # val_stats = {"acc1": -1}

        if epoch % 1 == 0:

            # val_stats = test_zeroshot_3d_core(val_loader, model, tokenizer, args)
            # acc1 = val_stats["acc1"]
            # print(val_stats)

            # is_best = acc1 > best_acc1
            # if is_best:
            #     best_epoch = epoch

            # best_acc1 = max(acc1, best_acc1)

            if epoch % 1 == 0:
                print(f"=> saving {epoch + 1} checkpoint")
                trainable_state_dict = {k: v for k, v in model.state_dict().items() if 'clip_model' not in k}
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': trainable_state_dict,
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        # 'best_acc1': best_acc1,
                        'args': args,
                    }, True, args.output_dir)

            # if epoch + 1 == args.epochs:
            #     print("=> saving last checkpoint")
            #     utils.save_on_master({
            #         'epoch': 'last',
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scaler': scaler.state_dict(),
            #         # 'best_acc1': best_acc1,
            #         'args': args,
            #     }, True, args.output_dir)

        # log_stats = {**{f's3dis_entity_train_{k}': v for k, v in s3dis_entity_train_stats.items()},
        #             **{f's3dis_view_train_{k}': v for k, v in s3dis_view_train_stats.items()},
        #             **{f's3dis_scene_train_{k}': v for k, v in s3dis_scene_train_stats.items()},
        #             **{f'scannet_entity_train_{k}': v for k, v in scannet_entity_train_stats.items()},
        #             **{f'scannet_view_train_{k}': v for k, v in scannet_view_train_stats.items()},
        #             **{f'scannet_scene_train_{k}': v for k, v in scannet_scene_train_stats.items()},
        #              'epoch': epoch,
        #             #  'best_acc1': best_acc1,
        #              'best_epoch': best_epoch}
        
        log_stats = {
            **{f'all_in_one_train_{k}': v for k, v in all_in_one_train_stats.items()},
            'epoch': epoch,
            # 'best_acc1': best_acc1,
            'best_epoch': best_epoch
        }

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
                # wandb.watch(model)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = 0
    for dataloader in train_loader.dataloaders:
        iters_per_epoch += len(dataloader)
    iters_per_epoch //= 4
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))
    outputs = {}

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter in range(iters_per_epoch):
        inputs_list = train_loader.get_data()
        if len(inputs_list) == 0:
            break
        for inputs in inputs_list:
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[it]

            pc = inputs[1]
            texts = inputs[0]

            if train_loader.current_dataloader_index == 1 or train_loader.current_dataloader_index == 4:
                image = inputs[2]
                inputs = [pc, texts, image]
            else:
                inputs = [pc, texts]

            inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                try:
                    output = model(*inputs)
                except Exception as e:
                    print_log("Error: {}".format(e))
                    continue
                for k in output.keys():
                    if k not in outputs:
                        outputs[k] = []
                    outputs[k].append(output[k].to(torch.float32))
                    
        if len(outputs['pc_embed']) % 4 != 0:
            outputs = {}
            continue
        outputs = {k: torch.stack(outputs[k], dim=0).squeeze() for k in outputs}
        loss_dict = criterion(outputs, train_loader.current_dataloader_index)
        loss = loss_dict['loss']
        # loss /= args.update_freq
        outputs = {}

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()
        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        utils.get_model(model).clip_model.logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).clip_model.logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)
        # if data_iter % 12 == 0:
        if args.wandb:
            wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                    'scaler': scaler.get_scale(),
                    'logit': logit_scale})
        progress.display(optim_iter)

    # progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[args.validate_dataset_name]


    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, (pc, target, target_name) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        print(','.join([str(value) for value in top5_accuracy_per_class.values()]))

    progress.synchronize()
    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc5': top5.avg}

# def test_zeroshot_3d(args):
#     ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
#     state_dict = OrderedDict()
#     for k, v in ckpt['state_dict'].items():
#         state_dict[k.replace('module.', '')] = v

#     # create model
#     old_args = ckpt['args']
#     print("=> creating model: {}".format(old_args.model))
#     try:
#         model = getattr(models, old_args.model)(args=args)
#         model.cuda()
#         model.load_state_dict(state_dict, strict=True)
#         print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
#     except:
#         model = getattr(models, args.model)(args=args)
#         model.cuda()
#         model.load_state_dict(state_dict, strict=True)
#         print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

#     tokenizer = SimpleTokenizer()

#     test_dataset = get_dataset(None, tokenizer, args, 'val')
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
#     )
#     results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

#     return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
