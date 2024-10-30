from typing import Union, Type

import torch
from jbag.models.unet import Encoder, StackedConvBlock
from jbag.models.utils import get_matching_conv_transpose_op
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels):
        super().__init__()
        attn_channels = (g_channels + x_channels) // 2
        self.conv_g = nn.Sequential(nn.Conv2d(g_channels, attn_channels, 1),
                                    nn.BatchNorm2d(attn_channels))

        self.conv_x = nn.Sequential(nn.Conv2d(x_channels, attn_channels, 1),
                                    nn.BatchNorm2d(attn_channels))

        self.psi = nn.Sequential(nn.Conv2d(attn_channels, 1, 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g = self.conv_g(g)
        x_g = self.conv_x(x)
        g = self.relu(g + x_g)
        g = self.psi(g)

        return g * x


class GADecoder(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 num_conv_per_stage: Union[int, list[int], tuple[int, ...]],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 conv_bias: bool = None,
                 non_linear_first: bool = False):
        super().__init__()
        num_stages_encoder = len(encoder.output_channels)
        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = [num_conv_per_stage] * (num_stages_encoder - 1)
        assert len(num_conv_per_stage) == num_stages_encoder - 1
        conv_transpose_op = get_matching_conv_transpose_op(encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        non_linear = encoder.non_linear if non_linear is None else non_linear
        non_linear_kwargs = encoder.non_linear_kwargs if non_linear_kwargs is None else non_linear_kwargs

        self.tissue_stages, self.tissue_conv_transpose_ops = self.build_decoder_branch(conv_bias,
                                                                                       conv_transpose_op,
                                                                                       dropout_op,
                                                                                       dropout_op_kwargs,
                                                                                       encoder,
                                                                                       non_linear,
                                                                                       non_linear_first,
                                                                                       non_linear_kwargs,
                                                                                       norm_op,
                                                                                       norm_op_kwargs,
                                                                                       num_conv_per_stage,
                                                                                       num_stages_encoder)

        self.region_stages, self.region_conv_transpose_ops = self.build_decoder_branch(conv_bias,
                                                                                       conv_transpose_op,
                                                                                       dropout_op,
                                                                                       dropout_op_kwargs,
                                                                                       encoder,
                                                                                       non_linear,
                                                                                       non_linear_first,
                                                                                       non_linear_kwargs,
                                                                                       norm_op,
                                                                                       norm_op_kwargs,
                                                                                       num_conv_per_stage,
                                                                                       num_stages_encoder)
        self.ag_start_block = 1
        self.gate_blocks = nn.ModuleList(
            [AttentionGate(encoder.output_channels[-(i + 2)], encoder.output_channels[-(i + 2)])
             for i in range(self.ag_start_block, num_stages_encoder - 1)])

    @staticmethod
    def build_decoder_branch(conv_bias, conv_transpose_op, dropout_op, dropout_op_kwargs, encoder, non_linear,
                             non_linear_first, non_linear_kwargs, norm_op, norm_op_kwargs, num_conv_per_stage,
                             num_stages_encoder):
        stages = []
        conv_transpose_ops = []
        for i in range(1, num_stages_encoder):
            input_features_below = encoder.output_channels[-i]
            input_features_skip = encoder.output_channels[-(i + 1)]
            stride_for_transpose_conv = encoder.strides[-i]
            conv_transpose_ops.append(conv_transpose_op(input_features_below,
                                                        input_features_skip,
                                                        stride_for_transpose_conv,
                                                        stride_for_transpose_conv,
                                                        bias=conv_bias
                                                        ))

            stages.append(StackedConvBlock(
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                num_convs=num_conv_per_stage[i - 1],
                conv_op=encoder.conv_op,
                kernel_size=encoder.kernel_sizes[-(i + 1)],
                initial_stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                non_linear=non_linear,
                non_linear_kwargs=non_linear_kwargs,
                non_linear_first=non_linear_first
            ))
        stages = nn.ModuleList(stages)
        conv_transpose_ops = nn.ModuleList(conv_transpose_ops)
        return stages, conv_transpose_ops

    def forward(self, skips):
        x_tissue = skips[-1]
        x_region = skips[-1]
        for i in range(len(self.tissue_stages)):
            x_tissue = self.tissue_conv_transpose_ops[i](x_tissue)
            x_tissue = torch.cat((x_tissue, skips[-(i + 2)]), dim=1)
            x_tissue = self.tissue_stages[i](x_tissue)

            x_region = self.region_conv_transpose_ops[i](x_region)
            x_region = torch.cat((x_region, skips[-(i + 2)]), dim=1)
            x_region = self.region_stages[i](x_region)

            if i >= self.ag_start_block:
                x_tissue = self.gate_blocks[i - self.ag_start_block](x_tissue, x_region)

        return x_tissue, x_region


class GANet(nn.Module):
    def __init__(self, input_channels: int,
                 num_classes: int,
                 num_stages: int,
                 num_features_per_stage: Union[int, list[int], tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, list[int], tuple[int, ...]],
                 strides: Union[int, list[int], tuple[int, ...]],
                 num_conv_per_stage_encoder: Union[int, list[int], tuple[int, ...]],
                 num_conv_per_stage_decoder: Union[int, list[int], tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 non_linear: Union[None, Type[nn.Module]] = None,
                 non_linear_kwargs: dict = None,
                 non_linear_first: bool = False
                 ):
        super().__init__()

        if isinstance(num_conv_per_stage_encoder, int):
            num_conv_per_stage_encoder = [num_conv_per_stage_encoder] * num_stages

        if isinstance(num_conv_per_stage_decoder, int):
            num_conv_per_stage_decoder = [num_conv_per_stage_decoder] * (num_stages - 1)

        assert len(num_conv_per_stage_encoder) == num_stages
        assert len(num_conv_per_stage_decoder) == num_stages - 1
        self.encoder = Encoder(input_channels=input_channels,
                               num_stages=num_stages,
                               num_features_per_stage=num_features_per_stage,
                               conv_op=conv_op,
                               kernel_sizes=kernel_sizes,
                               strides=strides,
                               num_conv_per_stage=num_conv_per_stage_encoder,
                               conv_bias=conv_bias,
                               norm_op=norm_op,
                               norm_op_kwargs=norm_op_kwargs,
                               non_linear=non_linear,
                               non_linear_kwargs=non_linear_kwargs,
                               return_skips=True,
                               non_linear_first=non_linear_first
                               )
        self.decoder = GADecoder(encoder=self.encoder,
                                 num_conv_per_stage=num_conv_per_stage_decoder,
                                 non_linear_first=non_linear_first
                                 )

        self.tissue_head_1 = StackedConvBlock(input_channels=num_features_per_stage[0],
                                              output_channels=num_features_per_stage[0],
                                              num_convs=num_conv_per_stage_decoder[0],
                                              conv_op=conv_op,
                                              kernel_size=kernel_sizes[0],
                                              initial_stride=1,
                                              conv_bias=conv_bias,
                                              norm_op=norm_op,
                                              norm_op_kwargs=norm_op_kwargs,
                                              non_linear=non_linear,
                                              non_linear_kwargs=non_linear_kwargs,
                                              non_linear_first=non_linear_first)

        self.tissue_head_2 = StackedConvBlock(input_channels=num_features_per_stage[0],
                                              output_channels=num_classes,
                                              num_convs=num_conv_per_stage_decoder[0],
                                              conv_op=conv_op,
                                              kernel_size=kernel_sizes[0],
                                              initial_stride=1,
                                              conv_bias=conv_bias,
                                              norm_op=norm_op,
                                              norm_op_kwargs=norm_op_kwargs,
                                              non_linear=non_linear,
                                              non_linear_kwargs=non_linear_kwargs,
                                              non_linear_first=non_linear_first)

        self.region_head_1 = StackedConvBlock(input_channels=num_features_per_stage[0],
                                              output_channels=num_features_per_stage[0],
                                              num_convs=num_conv_per_stage_decoder[0],
                                              conv_op=conv_op,
                                              kernel_size=kernel_sizes[0],
                                              initial_stride=1,
                                              conv_bias=conv_bias,
                                              norm_op=norm_op,
                                              norm_op_kwargs=norm_op_kwargs,
                                              non_linear=non_linear,
                                              non_linear_kwargs=non_linear_kwargs,
                                              non_linear_first=non_linear_first)

        self.region_head_2 = StackedConvBlock(input_channels=num_features_per_stage[0],
                                              output_channels=num_classes,
                                              num_convs=num_conv_per_stage_decoder[0],
                                              conv_op=conv_op,
                                              kernel_size=kernel_sizes[0],
                                              initial_stride=1,
                                              conv_bias=conv_bias,
                                              norm_op=norm_op,
                                              norm_op_kwargs=norm_op_kwargs,
                                              non_linear=non_linear,
                                              non_linear_kwargs=non_linear_kwargs,
                                              non_linear_first=non_linear_first)

        self.attn_gate1 = AttentionGate(num_features_per_stage[0], num_features_per_stage[0])
        self.attn_gate2 = AttentionGate(num_classes, num_classes)

    def forward(self, x):
        skips = self.encoder(x)
        x_tissue, x_region = self.decoder(skips)

        x_tissue = self.tissue_head_1(x_tissue)
        x_region = self.region_head_1(x_region)

        x_tissue = self.attn_gate1(x_tissue, x_region)

        x_region = self.region_head_2(x_region)
        x_tissue = self.tissue_head_2(x_tissue)

        x_tissue = self.attn_gate2(x_tissue, x_region)

        return x_tissue, x_region
