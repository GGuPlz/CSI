import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet50
import math
from .head import DynamicHead
from collections import namedtuple

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None
def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.tensor(betas, dtype=torch.float32)
    return torch.clip(betas, 0, 0.999)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ResNet50(nn.Module):
    def __init__(self, out_indices=(1, 2, 3), frozen_stages=1):
        super(ResNet50, self).__init__()

        # 加载 torchvision 的 ResNet50
        base_model = resnet50(pretrained=True)

        # stem 部分
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # ResNet 的四个 stage 对应 layer1, 2, 3, 4
        self.layer1 = base_model.layer1  # stage 1
        self.layer2 = base_model.layer2  # stage 2
        self.layer3 = base_model.layer3  # stage 3
        self.layer4 = base_model.layer4  # stage 4

        self.res_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        self.out_indices = out_indices

        # 冻结指定的 stage（不参与训练）
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages):
        if frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in [*self.conv1.parameters(), *self.bn1.parameters()]:
                param.requires_grad = False
        for i in range(1, frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            # print(x.shape)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

class ChannelMapper(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_outs=None):
        super(ChannelMapper, self).__init__()
        assert isinstance(in_channels, list)
        if num_outs is None:
            num_outs = len(in_channels)
        self.num_outs = num_outs

        # 构建 convs，用于 channel mapping
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(self._build_conv(in_ch, out_channels, kernel_size, norm_cfg, act_cfg))

        # 如果 num_outs > len(in_channels)，添加额外的 downsample convs
        self.extra_convs = nn.ModuleList()
        if num_outs > len(in_channels):
            for i in range(num_outs - len(in_channels)):
                in_ch = in_channels[-1] if i == 0 else out_channels
                self.extra_convs.append(self._build_conv(in_ch, out_channels, kernel_size=3,
                                                         norm_cfg=norm_cfg, act_cfg=act_cfg, stride=2))

    def _build_conv(self, in_ch, out_ch, kernel_size, norm_cfg, act_cfg, stride=1):
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)]

        # 支持 GroupNorm
        if norm_cfg is not None and norm_cfg.get("type") == "GN":
            num_groups = norm_cfg.get("num_groups", 32)
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))

        # 支持 ReLU 或其他激活函数
        if act_cfg is not None:
            act_type = act_cfg.get("type", "ReLU")
            if act_type == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif act_type == "LeakyReLU":
                layers.append(nn.LeakyReLU(inplace=True))
            # 可以加其他激活函数支持

        return nn.Sequential(*layers)

    def forward(self, inputs):
        assert len(inputs) == len(self.convs)
        outs = [conv(feat) for conv, feat in zip(self.convs, inputs)]

        if self.extra_convs:
            last_feat = inputs[-1]
            for i, conv in enumerate(self.extra_convs):
                if i == 0:
                    last_feat = conv(last_feat)
                else:
                    last_feat = conv(last_feat)
                outs.append(last_feat)
        return outs

class FeatureMerger(nn.Module):
    def __init__(self):
        super(FeatureMerger, self).__init__()

    def forward(self, inputs):
        # inputs: List of [B, C, H, W]
        merged = []
        for feat in inputs:
            B, C, H, W = feat.shape
            feat = feat.view(B, C, H * W)      # (B, C, HW)
            feat = feat.permute(0, 2, 1)       # (B, HW, C)
            merged.append(feat)
        out = torch.cat(merged, dim=1)         # (B, HW1+HW2+HW3+HW4, C)
        return out   

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta  = nn.Linear(dim, dim)
    
    def forward(self, x, t_emb, f_emb):
        cond = t_emb + f_emb
        scale = self.gamma(cond)
        shift = self.beta(cond)
        return F.relu(self.fc(x) * (1 + scale) + shift)


class DenoiseMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_blocks=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_linear = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(n_blocks)])
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, t, f):
        t_emb = self.time_mlp(t)
        f_emb = self.cond_linear(f.mean(dim=2))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb, f_emb)
        return self.output_proj(h)



class csidiffusion(nn.Module):
    def __init__(self, out_indices=(1, 2, 3), frozen_stages=1, in_channels=[256, 512, 1024], out_channels=128 ,            
                embed_dim=22, num_heads=2, comp_block=16, sel_block=16, win_size=128,
                eps=1e-6, 
                hidden_ratio=4, intermediate_size=None, hidden_act="swish",
                num_block=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.comp_block = comp_block
        self.sel_block = sel_block
        self.win_size = win_size
        #归一化的参数
        self.eps =eps
        #多层感知机的参数
        self.hidden_ratio= hidden_ratio
        self.intermediate_size= intermediate_size
        self.hidden_act= hidden_act
        #block的个数
        self.num_block =num_block

        # 初始化 ResNet50
        self.resnet50 = ResNet50(out_indices=out_indices, frozen_stages=frozen_stages)

        self.channel_mapper = ChannelMapper(
                             in_channels=[512, 1024, 2048],
                             out_channels=256,
                             kernel_size=1,
                             norm_cfg=dict(type='GN', num_groups=32),
                             act_cfg=None,
                             num_outs=4
                            )
        self.combine = FeatureMerger()  #(32 22,256)
        self.head = DynamicHead()
        self.decode = nn.Sequential(
             nn.Linear(256 * 22, 512),
             nn.ReLU(),
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Linear(256, 102)
        ) 
        
        self.code = DenoiseMLP(in_dim=17*2*3, hidden_dim=256, n_blocks=2)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_proposals = 3
        self.num_keypoints = 17
        
        
        # diffusion
        timesteps = 100
        sampling_timesteps = 1
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        self.scale = 2.0
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.box_renewal = True
        self.use_ensemble = True
        self.hw = torch.tensor([86., 42.]).to(self.device)
        

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
    def forward(self, input_csi, input_keypoints):
        csi = input_csi
        resnet_outputs = self.resnet50(csi)
        channel_mapped_outputs = self.channel_mapper(resnet_outputs)

        combined = self.combine(channel_mapped_outputs) #[32,22,256]
        combined = combined.permute(0,2,1)   #[32,256,22]
        #combined = combined.unsqueeze(1).repeat(1, self.num_proposals, 1, 1)  # [32, 10, 256, 22]
        bs = input_csi.shape[0] 
        #combined = combined.view(bs * self.num_proposals, 256, self.embed_dim)     # [320, 256, 22]
        

        # combined = combined.reshape(input_csi.shape[0], -1)
        # csi = self.decode(combined)
        # csi = csi.reshape(-1, 3, 17, 2) 
        # return csi
        if not self.training:
            results = self.ddim_sample(bs, combined)
            return results
        
        if self.training:
            gt_keypoints = input_keypoints.clone()
            gt_keypoints /= self.hw  # 归一化到0-1之间
            x_keypoints, noises, t = self.prepare_targets(gt_keypoints)
            t = t.squeeze(-1)
            x_keypoints *= self.hw  # 还原到原始尺度
            output = self.code(x_keypoints.view(bs, -1), t, combined)
            output = output.view(-1, 3, 17, 2)
            #output = self.head(combined, x_keypoints, t, None)
            return output
        
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        
    def model_predictions(self, backbone_feats, x, t, x_self_cond=None, clip_x_start=False):
        # 假设 x 是归一化到 [-scale, scale] 的关键点坐标 (B, N, K*2)，K 为关键点数量
        
        x_kpts = torch.clamp(x, min=-self.scale, max=self.scale)
        x_kpts = ((x_kpts / self.scale) + 1) / 2  # 映射到 [0, 1]
        x_kpts = x_kpts * self.hw  # 还原到原始尺度
        # 将处理后的关键点送入头部网络进行预测
        #outputs_kpts = self.head(backbone_feats, x_kpts, t, None)
        outputs_kpts = self.code(x_kpts.view(x_kpts.shape[0], -1), t, backbone_feats)
        outputs_kpts = outputs_kpts.view(-1, 3, 17, 2)
        # x_start 是去噪后的预测结果（关键点）
        x_start = outputs_kpts  # (B, N, K*2)，表示 K 个关键点的 (x, y)
        x_start= x_kpts / self.hw

        # 映射回 [-scale, scale]
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-self.scale, max=self.scale)

        # 根据扩散目标预测噪声
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_kpts

    
    def ddim_sample(self, bs,backbone_feats, clip_denoised=True, do_postprocess=True):
        batch = bs
        shape = (batch, self.num_proposals, self.num_keypoints, 2)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=self.device)

        ensemble_kps = []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, output_kpts = self.model_predictions(backbone_feats, img, time_cond, self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start  # [B, N, K, 2]

            if self.box_renewal:
                # 这里可以定义保留某些关键点或姿态框的策略（例如根据关键点中心分布筛选），默认保留全部
                num_remain = self.num_proposals  # 假设全部保留

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if self.box_renewal:
                replenish = torch.randn(batch, self.num_proposals - num_remain, self.num_keypoints, 2, device=img.device)
                img = torch.cat([img[:, :num_remain], replenish], dim=1)

            if self.use_ensemble and self.sampling_timesteps > 1:
                ensemble_kps.append(output_kpts)

        if self.use_ensemble and self.sampling_timesteps > 1:
            all_kps = torch.stack(ensemble_kps, dim=0)  # [T, B, N, K, 2]
            kps = all_kps.mean(dim=0)  # [B, N, K, 2]
        else:
            kps = output_kpts  # [B, N, K, 2]
        return kps
       
        
    def prepare_targets(self, gt_keypoints):
        diffused_keypoints = []
        noises = []
        ts = []

        for gt_keypoint in gt_keypoints:
            keypoint = gt_keypoint
            # CSI 数据下，坐标默认已经归一化，无需 / image_size      
            keypoint = torch.as_tensor(keypoint, dtype=torch.float32, device=self.device) # 强制转成 tensor，防止是 list
            # 加入扩散噪声
            d_keypoint, d_noise, d_t = self.prepare_diffusion_concat(keypoint)  # 你需要定义为支持 [N, K, 2]
            diffused_keypoints.append(d_keypoint)  # [N, K, 2]
            noises.append(d_noise)                 # [N, K, 2]
            ts.append(d_t)                         # [N]

        # 聚合返回
        return torch.stack(diffused_keypoints), torch.stack(noises), torch.stack(ts)
    
    def prepare_diffusion_concat(self, gt_keypoints):
        """
        gt_keypoints: Tensor, shape [N, K, 2], 归一化坐标 (0~1)
            num_proposals: int, 扩散候选关键点组数量默认self.num_proposals
        """

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()  # 单个扩散时间步
        N, K, _ = gt_keypoints.shape
        noise = torch.randn(self.num_proposals, K, 2, device=self.device)  # 采样噪声

        # 如果没有GT关键点，生成一个默认关键点组，全部0.5中间点
        if N == 0:
            gt_keypoints = torch.full((1, K, 2), 0.5, dtype=torch.float, device=self.device)
            N = 1
        # 处理GT数量与proposal数量的关系
        if N < self.num_proposals:
            # 用均值附近的随机噪声填充缺少的proposal
            placeholder = torch.randn(self.num_proposals - N, K, 2, device=self.device) / 6. + 0.5
            # clamp到合理范围避免负值
            placeholder = placeholder.clamp(0., 1.)
            x_start = torch.cat([gt_keypoints, placeholder], dim=0)  # [num_proposals, K, 2]
        elif N > self.num_proposals:
            # 随机选择部分GT关键点组，超出部分丢弃
            indices = torch.randperm(N, device=self.device)[:self.num_proposals]
            x_start = gt_keypoints[indices]
        else:
            x_start = gt_keypoints  # 数量刚好

        # 归一化坐标映射到 [-scale, scale] 区间，方便扩散噪声处理
        x_start = (x_start * 2. - 1.) * self.scale  # [num_proposals, K, 2]

        # 使用q_sample加噪，t为扩散步数，noise是噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # [num_proposals, K, 2]
        x_noisy = torch.clamp(x_noisy, min=-self.scale, max=self.scale)# 限制噪声加权后的范围
        x_noisy = ((x_noisy / self.scale) + 1) / 2.# 映射回归一化区间[0, 1]

        return x_noisy, noise, t
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise    
            


if __name__ == "__main__":
    # 模拟配置对象

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模拟 roi_input_shape
    roi_input_shape = None
    # 初始化 DynamicHead
    model = csidiffusion()
    model = model.to(device)
    model.eval()

    # 模拟输入数据
    
    csi = torch.randn(32, 3, 30, 30).to(device)
    keypoint = torch.randn(32, 3, 17, 2).to(device)
    # 前向传播
    with torch.no_grad():
        pred_bboxes = model(csi, keypoint)


    # 打印输出结果
    print("Predicted bboxes shape:", pred_bboxes.shape)
    print(pred_bboxes.dtype)
    print(pred_bboxes.max())