import torch
import torch.nn as nn
import torch.nn.init as torch_init
import numpy as np
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath
from functools import partial
import math
import seaborn as sns
import matplotlib.pyplot as plt

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
binary_CE_loss = torch.nn.BCELoss(reduction='mean')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv1d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)
        layers.extend([
            nn.Conv1d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x = self.conv(x_conv).permute(0, 2, 1)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, td=None):
        B, N, C = x.shape  # (40 32 512)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if td is not None:
            qkv_td = self.qkv(td).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v = v + qkv_td[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # (40 32 512)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv = LocalityFeedForward(dim, dim, 1, 4.0, "hs+se", 4.0, False, dp_first=False)

    def forward(self, x, td=None):
        x = x + self.conv(x)

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), td)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class Decode_Block(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.linear = nn.Linear(inplanes, inplanes, bias=False)
        self.linear2 = nn.Linear(inplanes, inplanes, bias=False)

    def forward(self, x):
        x = self.linear(x)
        out = self.linear2(x)
        return x, out


class Decode_Block_Geu(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True, add_layer_norm=True):
        super().__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.gating = gating
        self.fc_c = nn.Linear(output_dimension, output_dimension)
        self.add_layer_norm = add_layer_norm
        self.layer_norm = nn.LayerNorm(input_dimension)
        self.linear = nn.Linear(output_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)

        if self.gating:
            x1 = self.fc_c(x)
            if self.add_layer_norm:
                x1 = self.layer_norm(x1)
            x = torch.cat((x, x1), 1)
            x = F.glu(x, 1)

        x = F.normalize(x)
        out = self.linear(x)
        return x, out


class Decode_Block_DU(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.fc1 = nn.Linear(inplanes, 32)
        self.fc2 = nn.Linear(32, inplanes)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(inplanes, inplanes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        out = self.fc(x)
        return x, out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.fusion = args.fusion
        self.batch_size = args.batch_size
        self.feature_group = args.feature_group
        self.aggregate_text = args.aggregate_text
        self.feedback_mode = args.feedback_mode
        if self.feedback_mode == 'concat':
            self.fc_feed = nn.Linear(args.feature_size * 2, args.feature_size)
            self.fc_feed_t = nn.Linear(args.feature_size * 2, args.feature_size)

        if args.dataset == 'ucfcrime' or args.dataset == 'violence':
            self.num_segments = 32
        elif args.dataset == 'shanghai':
            self.num_segments = 32
        elif args.dataset == 'tad':
            self.num_segments = 24

        self.k_abn = self.num_segments // 10  # top k for abnormal snippets
        self.k_nor = self.num_segments // 10  # top k for normal snippets

        if self.feature_group == 'both':
            if args.fusion == 'concat':
                self.fc1 = nn.Linear(args.feature_size + args.feature_size, 512)
            elif args.fusion == 'add' or args.fusion == 'product':
                self.fc0 = nn.Linear(args.feature_size, args.emb_dim)
                # self.fc1 = nn.Linear(args.emb_dim, 512)
                self.fc1 = nn.Linear(args.feature_size, 512)
            elif 'up' in args.fusion:
                self.fc_vis = nn.Linear(args.feature_size, args.feature_size + args.emb_dim)
                self.fc_text = nn.Linear(args.emb_dim, args.feature_size + args.emb_dim)
                self.fc1 = nn.Linear(args.feature_size + args.emb_dim, 512)
            else:
                raise ValueError('Unknown fusion method: {}'.format(args.fusion))
        elif self.feature_group == 'text':
            self.fc1 = nn.Linear(args.emb_dim, 512)
        else:
            self.fc1 = nn.Linear(args.feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.triplet = nn.TripletMarginLoss(margin=1, reduction='mean')
        self.gama = args.gama
        self.apply(weight_init)

 
        self.global_pool = 'token'
        self.class_token = True
        self.num_prefix_tokens = 1 if self.class_token else 0
        self.no_embed_class = False
        self.grad_checkpointing = False
        self.fc_norm = None
        self.fc_norm_t = None
        self.norm_layer = None
        self.act_layer = None
        drop_path_rate = 0.
        attn_drop_rate = 0.
        qkv_bias = True
        depth = args.depth
        mlp_ratio = 4.
        drop_rate = 0.
        num_heads = 8
        init_values = None
        block_fn = Block

        assert self.global_pool in ('', 'avg', 'token')
        assert self.class_token or self.global_pool != 'token'
        use_fc_norm = self.global_pool == 'avg' if self.fc_norm is None else self.fc_norm
        use_fc_norm_t = self.global_pool == 'avg' if self.fc_norm_t is None else self.fc_norm_t
        norm_layer = self.norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = self.act_layer or nn.GELU

        self.feature_align = nn.Linear(args.emb_dim, args.feature_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=args.feature_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.blocks_t = nn.Sequential(*[
            block_fn(
                dim=args.feature_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # prompt set
        self.num_anomaly_cls = args.prompt_num
        self.prompt_dim = args.feature_size  # 512
        self.share_prompt = torch.nn.parameter.Parameter(torch.randn(self.num_anomaly_cls, self.prompt_dim),
                                                       requires_grad=True)

        # prompt loss super-parameter
        self._lambda = args._lambda
        self.margin_prompt = args.margin_prompt

        # var_loss
        self.omega_v = args.omega_v
        self.omega_t = args.omega_t

        # feedback decoders
        self.feedback_aug = args.feedback_aug
        self.decoders_mode = args.decoders_mode

        if self.decoders_mode == 'mlp':
            self.decoders = nn.ModuleList([Decode_Block(self.prompt_dim) for _ in range(depth)])
            self.decoders_t = nn.ModuleList([Decode_Block(self.prompt_dim) for _ in range(depth)])
        elif self.decoders_mode == 'geu':
            self.decoders = nn.ModuleList(
                [Decode_Block_Geu(self.prompt_dim, self.prompt_dim) for _ in range(depth)])
            self.decoders_t = nn.ModuleList(
                [Decode_Block_Geu(self.prompt_dim, self.prompt_dim) for _ in range(depth)])
        elif self.decoders_mode == 'down_up':
            self.decoders = nn.ModuleList([Decode_Block_DU(self.prompt_dim) for _ in range(depth)])
            self.decoders_t = nn.ModuleList([Decode_Block_DU(self.prompt_dim) for _ in range(depth)])
        else:
            raise ValueError('Unknown fusion method: {}'.format(self.decoders_mode))

        self.norm = norm_layer(self.prompt_dim) if not use_fc_norm else nn.Identity()
        self.norm_t = norm_layer(self.prompt_dim) if not use_fc_norm_t else nn.Identity()
        self.fc_norm = norm_layer(self.prompt_dim) if use_fc_norm else nn.Identity()
        self.fc_norm_t = norm_layer(self.prompt_dim) if use_fc_norm_t else nn.Identity()

    def forward_features(self, x, td=None):
        in_var = []
        out_var = []
        for i, blk in enumerate(self.blocks):
            in_var.append(x)
            x = blk(x, td[i] if td is not None else None)
            out_var.append(x)
        x = self.norm(x)
        return x, in_var, out_var

    def forward_features_t(self, x, td=None):
        in_var = []
        out_var = []
        for i, blk in enumerate(self.blocks_t):
            in_var.append(x)
            x = blk(x, td[i] if td is not None else None)
            out_var.append(x)
        x = self.norm(x)
        return x, in_var, out_var

    def feedback(self, x):
        td = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            x, out = self.decoders[depth](x)
            td = [out] + td
        return td

    def feedback_t(self, x):
        td = []
        for depth in range(len(self.decoders_t) - 1, -1, -1):
            x, out = self.decoders_t[depth](x)
            td = [out] + td
        return td

    def forward_score(self, x, bs, ncrops):
        scores = self.relu(self.fc1(x))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)
        return scores

    def forward(self, inputs, text, videoname, is_training=False, return_all_features=False):
        # percent is a dynamic threshold to select abnormal features, accroding to iteration
        k_abn = self.k_abn
        k_nor = self.k_nor

        bs, ncrops, t, f = inputs.size()
        bs2, ncrops2, t2, f2 = text.size()

        inputs = inputs.view(-1, t, f)
        text = text.view(-1, t2, f2)

        text = self.feature_align(text)

        out = inputs
        out2 = text

        # step 1: feedforward
        output_each_iter = []

        '''visual'''
        out, _, __ = self.forward_features(out)
        out = self.drop_out(out)
        '''text'''
        if self.aggregate_text:
            out2, _, __ = self.forward_features_t(out2)
            out2 = self.drop_out(out2)

        # step 2: reweight
        x_norm = F.normalize(out, dim=-1)
        x_norm_t = F.normalize(out2, dim=-1)
        prompt_norm = F.normalize(self.share_prompt[None, ..., None], dim=-1)
        # prompt_norm_t = F.normalize(self.prompt_text[None, ..., None], dim=-1)

        '''visual'''
        cos_sim_all = []
        for pt in prompt_norm[0, :]:
            cos = x_norm @ pt 
            cos_sim_all.append(cos)
        cos_sim, cos_sim_index = torch.topk(torch.cat(cos_sim_all, dim=2), 1, dim=-1)      # (640, 64, 1)

        '''text'''
        cos_sim_all_t = []
        for pt in prompt_norm[0, :]:
            cos = x_norm_t @ pt
            cos_sim_all_t.append(cos)
        cos_sim_t, cos_sim_index_t = torch.topk(torch.cat(cos_sim_all_t, dim=2), 1, dim=-1)

        if self.feedback_mode == 'reweight':
            mask = cos_sim.clamp(0, 1)
            mask_t = cos_sim_t.clamp(0, 1)
            out = out * mask * self.feedback_aug  # (80, 64, 512) * (80, 64, 1)
            out2 = out2 * mask_t * self.feedback_aug
        elif self.feedback_mode == 'concat':
            out = torch.cat([out, np.squeeze(self.share_prompt[cos_sim_index])], dim=2)
            out2 = torch.cat([out2, np.squeeze(self.share_prompt[cos_sim_index_t])], dim=2)
            out = self.fc_feed(out)
            out2 = self.fc_feed_t(out2)
        elif self.feedback_mode == 'add':
            out = out + np.squeeze(self.share_prompt[cos_sim_index])
            out2 = out2 + np.squeeze(self.share_prompt[cos_sim_index_t])
        elif self.feedback_mode == 'product':
            out = out * np.squeeze(self.share_prompt[cos_sim_index])
            out2 = out2 * np.squeeze(self.share_prompt[cos_sim_index_t])
        else:
            raise ValueError('Unknown feedback_mode method: {}'.format(self.feedback_mode))

        # step 3: feedback
        td = self.feedback(out)
        td_t = self.feedback_t(out2)

        # step 4: feedforwar again
        out, in_var, out_var = self.forward_features(inputs, td)
        out2, in_var_t, out_var_t = self.forward_features_t(text, td_t)

   
        if out.shape[1] < out2.shape[1]:  # out(vis)比out2(text)少帧
            # remove the last frame of out2
            out2 = out2[:, :(out.shape[1] - out2.shape[1]), :]
        elif out.shape[1] > out2.shape[1]:  # out(vis)总比out2(text)多1帧
            # padding out2 by repeating the last frame
            out2 = torch.cat((out2, out2[:, (out2.shape[1] - out.shape[1]):, :]), dim=1)
        t = out.shape[1]

        # concat visual features with text features here，
        if self.fusion == 'concat':
            if self.feature_group == 'both':
                out = torch.cat([out, out2], dim=2)
            elif self.feature_group == 'text':
                out, ncrops, f = out2, ncrops2, f2
        elif self.fusion == 'product':
            out = self.relu(self.fc0(out))
            out = self.drop_out(out)
            out = out * out2
        elif self.fusion == 'add':
            # out = self.relu(self.fc0(out))
            # out = self.drop_out(out)
            if out.shape[1] > out2.shape[1]:
                out = out[:, :out2.shape[1], :]
            out = out + out2
        elif self.fusion == 'add_up':
            out = self.relu(self.fc_vis(out))
            out = self.drop_out(out)
            out2 = self.relu(self.fc_text(out2))
            out2 = self.drop_out(out2)
            out = out + out2
        else:
            raise ValueError('Unknown fusion method: {}'.format(self.fusion))

        features = out
        scores = self.forward_score(features, bs, ncrops)

        output_each_iter.append(scores)
        var_loss = self.var_loss(in_var, out_var, scores)
        var_loss_t = self.var_loss_t(in_var_t, out_var_t, scores)

        # features = out
        normal_features = features[0:self.batch_size * ncrops]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size * ncrops:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2,
                                     dim=2)  # tain: feat_magnitudes.shape=[640,32], use l2 norm to compute the feature magnitude
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # train: feat_magnitudes.shape=[64,32]
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # train: shape=[32,32], normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # train: shape=[32,32], abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        #######  process abnormal videos -> select top3 feature magnitude  #######
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]  # [0]为值, [1]为idx, train: shape=[32,3]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])  # train: shape=[32,3,2048]

        abnormal_features = abnormal_features.view(n_size, ncrops, t, -1)  # train: shape=[32,10,32,2048]
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)  # train: shape=[10,32,32,2048]

        device = abnormal_features.device

        # total_select_abn_feature = torch.zeros(0)
        total_select_abn_feature = torch.zeros(0, device=device)

        for abnormal_feature in abnormal_features:  # range(10)
            feat_select_abn = torch.gather(abnormal_feature, 1,
                                           idx_abn_feat)  # train: shape=[32,3,2048], top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  # train: shape=[32,3,1]
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                    dim=1)  # train: shape=[32,3,1]求mean后变为[32,1], top 3 scores in abnormal bag based on the top-3 magnitude

        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        # idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]]).to(device)

        normal_features = normal_features.view(n_size, ncrops, t, -1)
        normal_features = normal_features.permute(1, 0, 2, 3)

        # total_select_nor_feature = torch.zeros(0)
        total_select_nor_feature = torch.zeros(0, device=device)

        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1,
                                              idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        # idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]]).to(device)

        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)  # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature  # train: shape=[320,3,2048]
        feat_select_normal = total_select_nor_feature  # train: shape=[320,3,2048]

       # prompt loss
        prompt_index = self.num_anomaly_cls // 2
        prompt_n, prompt_a = torch.mean(self.share_prompt[:prompt_index], dim=0), torch.mean(self.share_prompt[prompt_index:], dim=0),

        loss_prompt = self.prompt_loss(prompt_n, prompt_a)

        if is_training:
            # score_abnormal, score_normal (shape=[32,1]) are the score of a video, while scores (shape=[64,32,1]) are the score vector of snippets in a video
            # therefore, we use score_abnormal and score_normal during training and scores during inference
            return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, \
                   feat_select_abn, feat_magnitudes, var_loss, var_loss_t, loss_prompt
        else:
            return inputs.squeeze(
                0), features, score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes

    def var_loss(self, in_var, out_var, x):
        # in_var、out_var come from forward_feature
        recon_loss = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            recon, out = self.decoders[depth](out_var[depth].detach())
            target = in_var[depth].detach()
            recon_loss.append(F.mse_loss(recon, target))
        return self.omega_v * sum(recon_loss)

    def var_loss_t(self, in_var, out_var, x):
        # in_var、out_var come from forward_feature
        recon_loss = []
        for depth in range(len(self.decoders_t) - 1, -1, -1):
            recon, out = self.decoders_t[depth](out_var[depth].detach())
            target = in_var[depth].detach()
            recon_loss.append(F.mse_loss(recon, target))
        return self.omega_t * sum(recon_loss)

    def prompt_loss(self, prompt_n, prompt_a):
        loss_abn = torch.abs(self.margin_prompt - torch.norm(prompt_a, p=2))
        loss_nor = torch.norm(prompt_n, p=2)
        loss_prompt = self._lambda * torch.mean((loss_abn + loss_nor) ** 2)
        return loss_prompt
