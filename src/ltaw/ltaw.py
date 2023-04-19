import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .ltaw_module.position_encoding import CA_Block
from .ltaw_module import WindowTransformer, FinePreprocess
from .ltaw_module.coarse_matching import CoarseMatching
from .ltaw_module.fine_matching import FineMatching # 放到build_regression_head
from .ltaw_module.regression_head import build_regression_head


class LTAW(nn.Module):
    """
    LTAW: it consists of
        - backbone: Local Feature CNN,contracting path of feature descriptor
        - pos_encoder: generates relative sine pos encoding
        - match coarse-level: matching on low-resolution feature maps
        - fine-level refinement: matching on high-resolution feature maps
        - regression_head: regresses disparity and occlusion, including optimal transport
    """
    
    
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = CA_Block(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.ltaw_coarse = WindowTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.ltaw_fine = WindowTransformer(config["fine"])
        self.fine_matching = FineMatching() 
        self.regression_head = build_regression_head(config)
        
        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()
        
    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
                
    def _disable_batchnorm_tracking(self):
        """
        disable Batchnorm tracking stats to reduce dependency on dataset (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.inplace = True
                
                
    def forward(self, data):
        """ 
        :param x: input data
        :return:
            a dictionary object with keys
            - "disp_pred" [N,H,W]: predicted disparity
            - "occ_pred" [N,H,W]: predicted occlusion mask
            - "disp_pred_low_res" [N,H//s,W//s]: predicted low res (raw) disparity
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. pos_encoder
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.ltaw_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.ltaw_fine(feat_f0_unfold, feat_f1_unfold)

        # 5.regression_head
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)# 这个和regression_head融合一下
       
        attn_weight = self.WindowTransformer(feat_left, feat_right)
        output = self.regression_head(attn_weight, x)
        return output
        

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
