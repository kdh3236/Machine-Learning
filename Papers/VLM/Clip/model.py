from collections import OrderedDict # 데이터가 입력된 순서를 기억하는 Dictionary
from typing import Tuple, Union 

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ResNet-152의 Bottleneck Block 구현 (Residual Block)
# 3x3 Conv를 가장 작은 채널 수를 갖는 Tensor 위에 처리하여 계산량을 줄이는 구조
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        inpanes: 입력 Channel 수
        planes: 출력 Channel 수 (Expansion 이전)
        stride: Convolution의 Stride
        """
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False) # 1x1 Conv
        self.bn1 = nn.BatchNorm2d(planes) # Per-channel normalization
        self.relu1 = nn.ReLU(inplace=True) # inplace=True : 메모리 절약 / input tensor를 직접 변경하고 새로운 메모리 공간 할당 x

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False) # 채널 수 유지, 3x3, padding=1 Conv
        self.bn2 = nn.BatchNorm2d(planes) # Per-channel normalization
        self.relu2 = nn.ReLU(inplace=True)  

        # stride = 1이라면, Downsampling을 수행하지 않음
        # stride > 1 이라면, avgpool을 통해 downsampling 수행
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity() # stride > 1 일 때만 avgpool 수행 / 입력이 stride만 있다면 Ketnel size도 stride로 설정됨

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False) # Channel 수 확장 / 1x1 Conv
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        # Downsampling이 필요한 경우 또는 input chennel 수와 output channel 수가 다른 경우 
        # Residual Connection을 위해 Downsampling Layer 정의       
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)), # Downsampling 수행
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), # Chennel 수를 맞춰줌
                ("1", nn.BatchNorm2d(planes * self.expansion)) # Per-channel normalization
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x))) # Conv(1x1) -> BN -> ReLU
        out = self.relu2(self.bn2(self.conv2(out))) # Conv(3x3) -> BN -> ReLU
        out = self.avgpool(out) # stride > 1 이라면, avgpool을 통해 downsampling 수행
        out = self.bn3(self.conv3(out)) # Conv(1x1) -> BN 

        if self.downsample is not None:
            identity = self.downsample(x) # Downsampling 수행

        out += identity # Residual Connection (Downsampling이 필요하면 Downsampling된 identity 사용)
        out = self.relu3(out) # 최종 ReLU 활성화 함수 적용
        return out

# ResNet에 적용하는 Attention Pooling Layer
class AttentionPool2d(nn.Module):
    # Special Dimension = H = W
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        special_dim = H = W
        embed_dim: 입력 Feature의 Channel 수
        num_heads: Multi-Head Attention의 Head 수
        output_dim: 출력 Feature의 Channel 수
        """

        super().__init__()
        # Resolution에 따른 Positional Embedding 정의 / Learnable Parameter로 정의
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5) # Random (Xavier/Kaiming) Initialization
        
        # Self-Attention을 위한 Q, K, V, C Projection Layer 정의
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output Projection Layer 정의
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim) # Output Dimension이 주어지지 않으면 embed_dim 사용

        # For Multi-Head Attention
        self.num_heads = num_heads  

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # N*C*H*W -> (HW)*N*C 
        # x.mean(dim=0, keepdim=True): Spatial Average Pooling 수행 / 1*N*C (HW->1)
        # 첫 번째 Token을 통해 이미지 전체적인 정보를 학습하도록 하기 위해 Spatial Average 
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)*N*C / Spatial Average 를 추가하여 첫 번째 Token으로 사용
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # positional_embedding (HW+1) * 1 * C (자동 Broadcasting 되도록) / (HW+1)*N*C
        
        # multi_head_attention_forward: multi_head_attention 
        # Outputs: attention output, weights
        x, _ = F.multi_head_attention_forward(
            query=x[:1], # First Spatial Average Token만 Query로 사용
            key=x, 
            value=x,
            embed_dim_to_check=x.shape[-1], # Embedding Dimension check
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), # Q, K, V Bias를 하나로 합쳐서 전달 (multi_head_attention_forward 함수의 요구사항)
            bias_k=None, # 추가 Leanable Bias 없음
            bias_v=None,
            add_zero_attn=False, # Key, Value에 Zero Attention 추가하지 않음
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, # q_proj_weight, k_proj_weight, v_proj_weight 각각 따로 사용
            training=self.training, # nn.Module의 training 변수 사용
            need_weights=False # attention weight(softmax(QKᵀ/√d))를 같이 리턴하지 않도록 한다.
        )
        return x.squeeze(0) # 1*N*C -> N*C


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
        - Making Convolutional Networks Shift-Invariant Again (Richard Zhang et al., 2019)에서 제안됨
        - Stride > 1인 Convolution 전에 Avgpool을 추가하여, Shift-Invariance 성질 향상
        - Downsampling 시, Blur -> downsample 순서로 처리하며 Anti-Aliasing 효과를 얻는 것과 비슷하다.
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        layers: 각 Layer에서 사용되는 Residual Block 수를 나타내는 튜플
        output_dim: 최종 출력 Feature Dimension
        heads: Attention Pooling Layer의 Head 수
        input_resolution: 입력 이미지 해상도 (H=W)
        width: ResNet의 기본 Channel 수
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution 

        # the 3-layer stem
        # Stem Convolution Layer: 이미지를 처음으로 입력받고 처리하여 Feature Map을 생성하는 부분
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False) # RGB -> width//2, 3x3, stride=2 Conv
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False) # Channel 유지, 3x3 Conv
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False) # width//2 -> width, 3x3 Conv
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2) # 2x2 Avgpool, stride=2

        # residual layers
        # Using Bottleneck class
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0]) 
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension

        # Attention Pooling Layer
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        planes: 출력 Channel 수 (Expansion 이전)
        blocks: Residual Block 수
        stride: Convolution의 Stride
        """

        # 첫 번째 Residual Block에서만 stride 적용하여 Downsampling 수행
        # 같은 Layer에서 Resolution이 일정하도록 유지하기 위함
        # Skip connection과 Resolution을 맞추기 위해서도 Downsampling이 한 번만 일어나야 한다.
        layers = [Bottleneck(self._inplanes, planes, stride)] # Residual Block 추가

        self._inplanes = planes * Bottleneck.expansion 
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        
        # X의 Type을 맞춰줌
        x = x.type(self.conv1.weight.dtype)

        # Stem Convolution
        x = stem(x)

        # Residual Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Attention Pooling
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    
    """ 
    수치적 안정성을 위해 원래의 dtype에서 torch.float32로 변환하여 LayerNorm을 수행한 후, 
    다시 원래의 dtype으로 변환하여 반환
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    # Approximation of GELU Activation Function / BERT류에서 주로 사용됨
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        d_model: Embedding Dimension
        n_head: Multi-Head Attention의 Head 수
        attn_mask: Attention Mask
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        """
        Feed-Forward Network
        OrderedDict를 사용하는 이유는 Checkpoint에서의 이름을 명시하기 위함
        모델 구조가 Seqeuntial(OrederedDict)로 인해 아래와 같이 저장됨.
        (mlp): Sequential(
            (c_fc): Linear(in, out*4)
            (gelu): QuickGELU()
            (c_proj): Linear(out*4, out)
        )
        """
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)), 
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        
        self.ln_2 = LayerNorm(d_model)

        # Clip은 Language Modeling 구조를 유지하기 위해 Transformer encoder임에도 Causal Mask를 사용함
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        # q, k, v, need_weights=False: attention weight를 같이 리턴 x, attn_mask 적용
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        Transformer Encoder Block의 일반적인 구조
            - LayerNorm -> Self-Attention -> Residual Connection -> LayerNorm -> MLP -> Residual Connection
            - Transformer는 Residual Connection 구조가 비교적 많아 Post-LN인 경우에 학습이 불안정해진다.
        Transformer류는 Layernrm이 Attention/Mlp 전에 오는 Pre-LN 구조를 주로 사용함
        """
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        width: Embedding Dimension
        layers: ResidualAttentionBlock 수
        heads: Multi-Head Attention의 Head 수
        attn_mask: Attention Mask
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    # 전체 ResidualAttentionBlock들을 순차적으로 통과시킴
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

"""
ViT 모델 구현
ResNet 대신 Vit를 사용하여 이미지 특징 추출
"""
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        input_resolution: H = W
        patch_size: Patch 크기
        width: Embedding Dimension
        layers: Transformer Block 수
        heads: Multi-Head Attention의 Head 수
        output_dim: 최종 출력 Feature Dimension
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # kennel_size = patch_size, stride = patch_size -> 이미지에서 겹치지 않는 Patch 추출
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5

        # Class token을 위한 Learnable Parameter 정의  
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) # kaiming/xavier initialization

        # N = input_resolution // patch_size이라면 Patch 개수 = N*N
        # N*N Patch + 1 (Class Token)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)) # kaiming/xavier initialization
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)

        # Output dimension을 맞추기 위한 Linear Projection Layer
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) # kaiming/xavier initialization

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [*, grid ** 2 + 1, width] -> [grid ** 2 + 1, *, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [grid ** 2 + 1, *, width] -> [*, grid ** 2 + 1, width]

        x = self.ln_post(x[:, 0, :]) # Class Token만 추출하여 LayerNorm 적용

        # Linear Projection
        if self.proj is not None:
            x = x @ self.proj # [*, width] @ [width, output_dim] -> [*, output_dim]

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)): # layers가 튜플 또는 리스트인 경우 ResNet 사용
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else: # layers가 int인 경우 ViT 사용
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        # clip.py에서 text token을 항상 context_length 길이로 맞춤
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        # Transformer hidden state -> Embedding Dimension 선형 변환을 위한 Projection Layer
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # Consine Similarity 조정을 위한 Temperature Learnable Parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # ResNet을 사용하는 경우
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property # CLIP.dtype()으로 접근하는 대신, CLIP.dtype로 접근할 수 있도록 하는 Decorator
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        Tokenized text by BPE Tokenizer (batch_size, n_ctx)
        -> Token Embedding (batch_size, n_ctx, transformer_width) 
        -> Positional Embedding 추가
        -> Transformer (batch_size, n_ctx, transformer_width) 
        -> LayerNorm
        -> eos token 위치의 Hidden State 추출 (batch_size, transformer_width) 
        -> Text Projection (batch_size, embed_dim)
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  #  [batch_size, n_ctx, d_model] -> [n_ctx, batch_size, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [n_ctx, batch_size, d_model] -> [batch_size, n_ctx, d_model]
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # text.argmax(dim=-1): eos_token_id가 가장 크기 때문에 eos token 위치 인덱스 반환
        # 각 Batch마다 eos token 위치의 Hidden State를 추출하여 Text Projection 수행
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image) # (batch_size, embed_dim)
        text_features = self.encode_text(text) # (batch_size, embed_dim)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t() # (batch_size, batch_size) / dim = 0: image, dim = 1: text
        logits_per_text = logits_per_image.t() # (batch_size, batch_size) / dim = 0: text, dim = 1: image

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to fp16
    For half precision training, convert certain layers to fp16.
    """
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict # state_dict에 visual.proj 키가 있으면 ViT 모델

    if vit: # ViT
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5) # sqrt(N*N - 1) = N
        image_resolution = vision_patch_size * grid_size
    else: # ResNet
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 필요없는 key 제거
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model) # 모델의 가중치를 fp16으로 변환
    model.load_state_dict(state_dict)
    return model.eval()
