import torch
import logging 

from .pann.pann import Cnn14
from .ast import ASTModel
from .ssast import ASTModel as SSASTModel
from .htsat.htsat import HtsatAudioModel
from .efficientat.MobileNetV3 import get_efficientat_model, NAME_TO_WIDTH
from .m2d.ar_m2d import AR_M2D, TaskNetwork
from audio_cls.utils.registry import MODELS


logger = logging.getLogger(__name__)


@MODELS.register("Cnn14")
def build_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, checkpoint_path):
    if checkpoint_path is None:
        return Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
    else: 
        logger.info("Load Checkpoint from:", checkpoint_path)
        model = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 527)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.fc_audioset = torch.nn.Linear(2048, classes_num, bias=True)
        return model


@MODELS.register("HtsatAudioModel")
def build_HtsatAudioModel(
        spec_size, patch_size, in_chans, num_classes, window_size, depths, 
        embed_dim, patch_stride, num_heads, config, checkpoint_path=None
    ):
    # https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6
        model = HtsatAudioModel(
             spec_size, patch_size, in_chans, num_classes, window_size, 
            depths, embed_dim, patch_stride, num_heads, config
        )
        if checkpoint_path is not None: 
            logger.info(f"Load checkpoint from: {checkpoint_path} successfully.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint["state_dict"].pop("sed_model.head.weight")
            checkpoint["state_dict"].pop("sed_model.head.bias")
            # finetune on the esc and spv2 dataset
            checkpoint["state_dict"].pop("sed_model.tscam_conv.weight")
            checkpoint["state_dict"].pop("sed_model.tscam_conv.bias")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        return model


@MODELS.register("ASTModel")
def build_ASTModel(
        label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, 
        model_size='base384', verbose=True, save_dir='/content', audioset_mdl_url='https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
    ):
    # https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1
    # https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1
    # https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1
    # https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1
    # https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1
    # https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1
    # https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1
    return ASTModel(label_dim, fstride, tstride, input_fdim, input_tdim, imagenet_pretrain, audioset_pretrain, model_size, verbose, save_dir, audioset_mdl_url)


@MODELS.register("EfficientatModel")
def build_EfficientatModel(pretrained_name, head_type, se_dims, num_classes):
    return get_efficientat_model(
        width_mult=NAME_TO_WIDTH(pretrained_name), pretrained_name=pretrained_name, head_type=head_type, 
        se_dims=se_dims, num_classes=num_classes
    )


@MODELS.register('M2DModel')
def build_M2DModel(cfg, n_class):
    ar = AR_M2D(cfg, do_aug=True)
    return TaskNetwork(cfg, ar, n_class)


@MODELS.register("SSASTModel")
def build_SSASTModel(
        label_dim=527, fshape=128, tshape=2, fstride=128, tstride=2,
        input_fdim=128, input_tdim=1024, model_size='base', pretrain_stage=True, 
        load_pretrained_mdl_path=None
    ):
    return SSASTModel(
        label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, 
        pretrain_stage, load_pretrained_mdl_path
    )