import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID,vit_base_patch16_224_TransReID_Tail,vit_base_patch16_224_TransReID_Head,vit_base_patch16_224_TransReID_MultiTail
from .IED import  JQZFeatEnHancer
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).to("cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.INPUT.PIXEL_STD).to("cuda").view(1, 3, 1, 1)
        self.enhance_mean = torch.tensor(cfg.INPUT.ENHANCE_PIXEL_MEAN).to("cuda").view(1, 3, 1, 1)
        self.enhance_std = torch.tensor(cfg.INPUT.ENHANCE_PIXEL_STD).to("cuda").view(1, 3, 1, 1)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        hiddim=64 
        self.enet = JQZFeatEnHancer(hiddim=hiddim)
        self.gray = torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).cuda()

        self.base_i_h = vit_base_patch16_224_TransReID_Head(img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                          camera=camera_num, view=view_num,
                                                          stride_size=cfg.MODEL.STRIDE_SIZE,
                                                          drop_path_rate=cfg.MODEL.DROP_PATH,
                                                          drop_rate=cfg.MODEL.DROP_OUT,
                                                          attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                          gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV,hiddim=hiddim)
        self.base_d1_h = vit_base_patch16_224_TransReID_Head(img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                          camera=camera_num, view=view_num,
                                                          stride_size=cfg.MODEL.STRIDE_SIZE,
                                                          drop_path_rate=cfg.MODEL.DROP_PATH,
                                                          drop_rate=cfg.MODEL.DROP_OUT,
                                                          attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                          gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV,hiddim=hiddim)
        self.multi_tail = vit_base_patch16_224_TransReID_MultiTail(img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                          camera=camera_num, view=view_num,
                                                          stride_size=cfg.MODEL.STRIDE_SIZE,
                                                          drop_path_rate=cfg.MODEL.DROP_PATH,
                                                          drop_rate=cfg.MODEL.DROP_OUT,
                                                          attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                          gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV,model_path=model_path, 
                                                          hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO,interaction_interval=cfg.MODEL.INTERACTION_INTERVAL)

        self.in_planes = self.base_i_h.in_planes
        if pretrain_choice == 'imagenet':
            self.base_i_h.load_param(model_path, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            self.base_d1_h.load_param(model_path, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            # self.base_d2_h.load_param(model_path, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            # self.base_d3_h.load_param(model_path, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier_i = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_i.apply(weights_init_classifier)


        self.bottleneck_i = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_i.bias.requires_grad_(False)
        self.bottleneck_i.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':            self.load_param(model_path)

    def forward(self, x, label=None, cam_label=None, view_label=None):

        g, tail = self.enet(x)

        x = self.base_i_h(x+tail, cam_label=cam_label, view_label=view_label)  # interaction
        # y = self.base_d1_h(g, cam_label=cam_label, view_label=view_label)  # interaction


        global_feat_i = self.multi_tail(x)      #1vit

        feat_i = self.bottleneck_i(global_feat_i)
        feat_cls_i = self.dropout(feat_i)

        if self.training:

            cls_score_i = self.classifier_i(feat_cls_i)

            return cls_score_i, global_feat_i
        else:
            return F.normalize(global_feat_i,dim=1,p=2) + F.normalize(feat_i,dim=1,p=2)



    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
