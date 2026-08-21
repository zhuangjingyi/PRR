# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import PRR_TripletLoss,AGWTripletLoss,RRSITRRobustTripletLoss
from .center_loss import CenterLoss
from .mmd_loss import MMDLoss, ZJYLoss, FDALoss
from .adasp_standalone import AdaSPLoss
from .Triplet_BH import BatchHardTripletLoss
from .adaptive_triplet import adaptive_TripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    
    center_criterion = FDALoss() #MMDLoss() #CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'adasp' in cfg.MODEL.METRIC_LOSS_TYPE:
        adasp = AdaSPLoss(
            temp=getattr(cfg.SOLVER, 'ADASP_TEMP', 0.05),       # 需要加配置项或用默认值
            loss_type=getattr(cfg.SOLVER, 'ADASP_TYPE', 'adasp')  # 'adasp' / 'sp-h' / 'sp-lh'
        )
        print(f"using AdaSP loss (type={adasp.loss_type}, temp={adasp.temp})")

    elif 'agw' in cfg.MODEL.METRIC_LOSS_TYPE:
        agw = AGWTripletLoss()
        print(f"using AGW loss")

    elif 'adaptive' in cfg.MODEL.METRIC_LOSS_TYPE:
        adaptive = adaptive_TripletLoss(margin=0.3)
        print(f"using adaptive_triplet loss from 25TPAMI-Instruct-ReID++")

    elif 'robust' in cfg.MODEL.METRIC_LOSS_TYPE:
        robust = RRSITRRobustTripletLoss()
        print(f"using robust_triplet loss from 26CVPR-RRSITR")

    elif 'triplet_bh' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet_bh = BatchHardTripletLoss(margin='soft', metric='euclidean')
        print(f"using Batch Hard Triplet Loss")

    elif 'prr_triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            prr_triplet = PRR_TripletLoss(s_pos=cfg.SOLVER.HARD_EXAMPLE_POS_K, s_neg=cfg.SOLVER.HARD_EXAMPLE_NEG_K)
            print("using soft triplet loss for training")
        else:
            prr_triplet = PRR_TripletLoss(cfg.SOLVER.MARGIN,s_pos=cfg.SOLVER.HARD_EXAMPLE_POS_K, s_neg=cfg.SOLVER.HARD_EXAMPLE_NEG_K)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            return F.cross_entropy(score, target)


    elif 'triplet' in sampler:
        # def loss_func(score, feat, target, target_cam): #score/feat not use enhance, the rest do enhance
        #     if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        #         # MMD_LOSS = mmdloss(feat, feat_e)
        #         if cfg.MODEL.IF_LABELSMOOTH == 'on':
        #             ID_LOSS = xent(score, target) #+ xent(score_e, target))*0.5
        #             TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]
        #             # print('xxxxxxxxxxxx')
        #             # print(cfg.MODEL.ID_LOSS_WEIGHT, cfg.MODEL.TRIPLET_LOSS_WEIGHT)
        #             return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS , cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
        #         else:
        #             ID_LOSS = F.cross_entropy(score, target) #+ F.cross_entropy(score_e, target))*0.5
        #             TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]
        #             # print('yyyyyyy')      #choose this
        #             # print(cfg.MODEL.ID_LOSS_WEIGHT, cfg.MODEL.TRIPLET_LOSS_WEIGHT)
        #             return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS , cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
        #     else:
        #         print('expected METRIC_LOSS_TYPE should be triplet'
        #               'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
                
        def loss_func(score, feat, target, target_cam): #score/feat not use enhance, the rest do enhance
            # ========== ID Loss（两种模式通用）==========
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                ID_LOSS = xent(score, target)
            else:
                ID_LOSS = F.cross_entropy(score, target)

            if cfg.MODEL.METRIC_LOSS_TYPE == 'adasp':
                # AdaSP 内部已做 L2 normalize，直接传 feat 和 target
                with torch.cuda.amp.autocast(enabled=False):          # 防 AMP 溢出
                    TRI_LOSS = adasp(feat.float(), target)          # 强制 float32
                # 不再需要 normalize_feature 参数和 [0] 下标
            
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'agw':
                TRI_LOSS = agw(feat, target, normalize_feature=True)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'adaptive':
                TRI_LOSS = adaptive(feat, target, feat.detach())[0]

            elif cfg.MODEL.METRIC_LOSS_TYPE =='robust':
                TRI_LOSS,_,_ = robust(feat, target)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_bh':
                TRI_LOSS = triplet_bh(feat, target)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'prr_triplet':
                TRI_LOSS = prr_triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
                
            return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS , cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion


