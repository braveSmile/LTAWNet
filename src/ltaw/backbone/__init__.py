from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 4, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 8, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    else:
        raise ValueError(f"LTAW.BACKBONE_TYPE {config['backbone_type']} not supported.")
