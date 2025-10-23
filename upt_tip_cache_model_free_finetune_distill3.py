"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from builtins import Exception
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou
import torch.nn.functional as F

from ops import (
    binary_focal_loss_with_logits,
)
from transformer_layers import (
    TransformerEncoder, 
    TransformerEncoderLayer,
    TransformerDecoder, 
    TransformerDecoderLayer as TripletDecoderLayer,
    SwinTransformer
)
from torchvision.ops import FeaturePyramidNetwork
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list
import sys
from hico_list import hico_verbs_sentence, hico_object_sentence, hico_verbs_sentence_reb_1, hico_verbs_sentence_reb_2
from vcoco_list import vcoco_verbs_sentence
sys.path.append('detr')
from detr.models import build_model
from detr.models.position_encoding import PositionEmbeddingSine
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
import CLIP_models_adapter_prior2
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle, random
from tqdm import tqdm
from torch.cuda import amp
from hico_text_label import hico_unseen_index
import hico_text_label
from HICO_utils import HOI_IDX_TO_ACT_IDX

from ops import compute_sinusoidal_pe, compute_spatial_encodings
from collections import defaultdict


_tokenizer = _Tokenizer()
class GaussianSampler(nn.Module):
    def __init__(self, origin_text_embeddings, zs_type="rare_first", base_path="./data/distribution"):
        super().__init__()

        self.n_cls, self.dim = origin_text_embeddings.shape

        if self.dim == 512: 
            base_path="./data/distribution"
        elif self.dim == 768: 
            base_path="./data/distribution_L"
        
        load_path = os.path.join(base_path, zs_type, "vdrp_group_cov.pt")
        group_cov_dict = torch.load(load_path)

        cov_diag = torch.stack([torch.diagonal(group_cov_dict[v]) for v in range(self.n_cls)], dim=0)
        visual_std = cov_diag.sqrt()  

        text_std = origin_text_embeddings.std(dim=1, keepdim=True)  
        visual_std_scaled = (visual_std / visual_std.mean(dim=1, keepdim=True)) * text_std  

        self.register_buffer("visual_std_scaled", visual_std_scaled)
        self.register_buffer("origin_text_embeddings", origin_text_embeddings) 

    def forward(self, learned_embeddings, noise_scale=0.1, apply_noise=True):
        """
        Args:
            learned_embeddings: [n_cls, dim] prompt after training
            noise_scale: float, multiplicative factor to control perturbation strength
            apply_noise: whether to apply perturbation (False = pass-through)
        Returns:
            perturbed_embeddings: [n_cls, dim]
        """
        if apply_noise:
            eps = torch.randn_like(learned_embeddings)
            perturbed = learned_embeddings + eps * self.visual_std_scaled * noise_scale
        else:
            perturbed = learned_embeddings

        return perturbed


def sparsemax(logits, dim=-1, temperature=10.0):
    logits = logits * temperature
    logits = logits - logits.max(dim=dim, keepdim=True)[0] 
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cumulative_sum = torch.cumsum(sorted_logits, dim=dim)
    rhos = torch.arange(1, logits.size(dim) + 1, device=logits.device).view([1]* (logits.dim()-1) + [-1])
    rhos = rhos.expand_as(logits)
    
    support = (sorted_logits * rhos) > (cumulative_sum - 1)
    support_size = support.sum(dim=dim, keepdim=True)
    tau = (cumulative_sum - 1) / support_size
    tau = tau.gather(dim, support_size.long() - 1)
    return torch.clamp(logits - tau, min=0.0)

class CueSelectorModule(nn.Module):
    def __init__(self, cue_bank, selection='sparsemax', temperature=10.0):
        super().__init__()
        self.register_buffer("cue_bank", cue_bank) 
        self.selection = selection
        self.temperature = temperature

    def forward(self, region_feat):
        B, D = region_feat.shape
        V, N, _ = self.cue_bank.shape 

        region_exp = region_feat[:, None, None, :]
        cue_bank_exp = self.cue_bank[None, :, :, :]

        sim = F.cosine_similarity(region_exp, cue_bank_exp, dim=-1) 

        if self.selection == 'sparsemax':
            w = sparsemax(sim, dim=-1, temperature=self.temperature) 
        elif self.selection == 'softmax':
            w = F.softmax(sim, dim=-1)
        elif self.selection == 'topk':
            topk_val, topk_idx = sim.topk(3, dim=-1) 
            w = torch.zeros_like(sim)  
            w.scatter_(dim=-1, index=topk_idx, src=topk_val)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)  

        c_sel = torch.sum(w.unsqueeze(-1) * cue_bank_exp, dim=2)
        return c_sel, w

class PromptUpdater(nn.Module):
    def __init__(self, cue_scale=0.2):
        super().__init__()
        self.cue_scale = cue_scale

    def forward(self, base_prompt_emb, c_sel):
        B, V, D = c_sel.shape

        z_base = base_prompt_emb.unsqueeze(0).expand(B, V, D)  
        z = z_base + self.cue_scale * c_sel

        return z

class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Weight_Pred(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = MLP(input_dim=input_dim, hidden_dim=512, output_dim=128, num_layers=2)
        self.drop1 = nn.Dropout()
        self.linear2 = MLP(input_dim=128, hidden_dim=32, output_dim=3, num_layers=2)
    
    def forward(self, x):
        x = self.drop1(self.linear1(x))
        x = self.linear2(x)
        return F.sigmoid(x)
    
def isin_tensor(values, test_elements):
    return (values[..., None] == test_elements).any(-1)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, prior=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, prior)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = args.CLASS_TOKEN_POSITION
        self.dtype = dtype
        
        if args.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
            
        self.use_cov_net = args.use_cov_net
        if self.use_cov_net:
            if ctx_dim == 512: 
                load_path = f"./data/distribution/{args.zs_type}"
            elif ctx_dim == 768: 
                load_path = f"./data/distribution_L/{args.zs_type}"

            file_name = "vdrp_group_cov.pt"
            group_cov = torch.load(os.path.join(load_path, file_name))  
            
            cov_feats = []
            for v in range(self.n_cls):
                cov = group_cov[v]
                cov_diag = torch.diagonal(cov)
                cov_feats.append(cov_diag)

            self.cov_feats = torch.stack(cov_feats, dim=0).to(dtype=dtype)
            self.cov_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(ctx_dim, ctx_dim // 4)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(ctx_dim // 4, ctx_dim))
            ])).to(dtype=dtype)
            self.delta_norm = nn.LayerNorm(ctx_dim)
            self.delta_scale = args.delta_scale if hasattr(args, 'delta_scale') else 0.02
        
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.tokenized_prompts = tokenized_prompts

    def forward(self):            
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.use_cov_net:
            cov_feats = self.cov_feats.to(ctx.device)  # [117, 512]
            delta = self.delta_norm(self.cov_net(cov_feats))            # [117, 512]
            
            delta = delta.unsqueeze(1).expand(-1, ctx.size(1), -1)  # [117, 24, 512]
            ctx = ctx + self.delta_scale * delta

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model,):
        super().__init__()
        if args.prompt_learning:
            self.prompt_learner = PromptLearner(args, classnames, clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.classnames = classnames
            self.text_encoder = TextEncoder(clip_model)

        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.patch_size = self.image_encoder.patch_size
        self.human_idx = args.human_idx        
            
    def forward(self, image):
        bs, c, h, w = image.shape
        visual_feats = self.image_encoder.conv1(image.type(self.dtype))
        visual_feats = visual_feats.reshape(bs, visual_feats.shape[1], -1).permute(0, 2, 1)
        visual_feats = torch.cat([self.image_encoder.class_embedding + torch.zeros(bs, 1, visual_feats.shape[-1], device=visual_feats.device), visual_feats], dim=1)
        visual_feats = visual_feats + self.image_encoder.positional_embedding
        visual_feats = self.image_encoder.ln_pre(visual_feats)
        visual_feats = visual_feats.permute(1, 0, 2)


class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module 
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action/interaction classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        origin_text_embeddings: torch.tensor,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model
        self.origin_text_embeddings = origin_text_embeddings
        self.object_embedding = object_embedding
        self.visual_output_dim = model.image_encoder.output_dim
        self.detector_output_dim = detector.transformer.d_model
        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.num_anno = kwargs["num_anno"]
        self.num_classes = num_classes

        if args.zs:
            self.zs_type = args.zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type = None

        self.seen_verb_idxs = list(set([HOI_IDX_TO_ACT_IDX[idx] for idx in range(600) if idx not in self.filtered_hoi_idx]))
        
        if self.num_classes == 117:
            self.unseen_verb_idxs = [i for i in range(self.num_classes) if i not in self.seen_verb_idxs]
        elif self.num_classes == 600:
            self.seen_hoi_idxs = [i for i in range(self.num_classes) if i not in self.filtered_hoi_idx]
        

        self.individual_norm = True
        self.logits_type = args.logits_type 
        self.consist = True
        self.evaluate_type = 'detr' 
        
        self.beta_cache = torch.tensor(10)
        self.alpha_cache = torch.tensor(1.0)

        self.prior_type = args.prior_type
        self.finetune_adapter = True
        self.prior_dim = self.visual_output_dim
        
        if self.prior_type == 'cbe':
            self.priors_initial_dim = self.prior_dim+5
        elif self.prior_type == 'cb':
            self.priors_initial_dim = 5
        elif self.prior_type == 'ce':
            self.priors_initial_dim = self.prior_dim+1
        elif self.prior_type == 'be':
            self.priors_initial_dim = self.prior_dim+4
        elif self.prior_type == 'c':
            self.priors_initial_dim = 1
        elif self.prior_type == 'b':
            self.priors_initial_dim = 4
        elif self.prior_type == 'e':
            self.priors_initial_dim = self.prior_dim
        else:
            raise NotImplementedError
        

        self.prompt_learning = args.prompt_learning
        
        if self.finetune_adapter:
            if 'T' in self.logits_type:
                if not self.prompt_learning:
                    self.adapter_union_weight = self.origin_text_embeddings.clone().detach().to(args.device)
                self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.7))
            
            if 'H' in self.logits_type:
                self.adapter_H_weight = nn.Parameter(torch.empty(self.visual_output_dim, self.visual_output_dim))
                nn.init.xavier_normal_(self.adapter_H_weight)
                self.logit_scale_H = nn.Parameter(torch.ones([]) * np.log(1 / 0.7))
                self.adapter_H_bias = nn.Parameter(-torch.ones(self.num_classes))
            
            if 'O' in self.logits_type:
                self.adapter_O_weight = nn.Parameter(torch.empty(self.visual_output_dim, self.visual_output_dim))
                nn.init.xavier_normal_(self.adapter_O_weight)
                self.logit_scale_O = nn.Parameter(torch.ones([]) * np.log(1 / 0.7))
                self.adapter_O_bias = nn.Parameter(-torch.ones(self.num_classes))
            
        
        if args.use_insadapter: 
            if args.prior_method == 0:
                self.priors_downproj = MLP(self.priors_initial_dim, 128, args.bottleneck, 3) 

        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]
        
        self.HOI_IDX_TO_OBJ_IDX = [
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                40, 40, 22, 22, 22, 22, 22
            ]

        self.obj_to_no_interaction = torch.as_tensor([169, 23, 75, 159, 9, 64, 193, 575, 45, 566, 329, 505, 417, 246,
                                                        30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
                                                        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
                                                        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
                                                        91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
                                                        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        self.epoch = 0
        self.COCO_CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
                    'fire hydrant','N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
                    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', \
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', \
                    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', \
                    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
                    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self.hico_hoi_to_idx = {}
        for idx, (vo_pair, text) in enumerate(hico_text_label.hico_text_label.items()):
            self.hico_hoi_to_idx[vo_pair] = idx
        
        self.reserve_indices = [idx for (idx, name) in enumerate(self.COCO_CLASSES) if name != 'N/A']
        self.reserve_indices = self.reserve_indices + [91]
        self.reserve_indices = torch.as_tensor(self.reserve_indices)
        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda
        self.featmap_dropout = nn.Dropout(kwargs["featmap_dropout_rate"])
        self.feat_mask_type = args.feat_mask_type
        self.use_insadapter = args.use_insadapter
        self.prior_method = args.prior_method
        self.box_proj = args.box_proj
        if self.box_proj:
            self.box_proj_mlp = MLP(8, 128, self.visual_output_dim, num_layers=3)

        self.use_text_adapter = args.use_text_adapter

        if 'S' in self.logits_type:
            self.spatial_head = nn.Sequential(
                nn.Linear(36, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, self.visual_output_dim), nn.ReLU(),
            )
            self.mmf = MultiModalFusion(self.visual_output_dim * 2, self.visual_output_dim, self.visual_output_dim) 
            self.spatial_fusion = nn.Linear(self.visual_output_dim * 2, self.visual_output_dim)
        
        
        self.use_verb_cue_selector = args.use_verb_cue_selector
        if self.use_verb_cue_selector: 
            load_path = './data/verb_concepts/'
            
            if self.visual_output_dim == 512: 
                human_cue_file = 'human_concepts.pt'
                object_cue_file = 'object_concepts.pt'
                context_cue_file = 'context_concepts.pt'
            elif self.visual_output_dim == 768: 
                if self.zs_type == 'vcoco':
                    human_cue_file = 'human_concepts_vcoco.pt'
                    object_cue_file = 'object_concepts_vcoco.pt'
                    context_cue_file = 'context_concepts_vcoco.pt'                
                else: 
                    human_cue_file = 'human_concepts_L.pt'
                    object_cue_file = 'object_concepts_L.pt'
                    context_cue_file = 'context_concepts_L.pt'

            self.branch = args.branch
            self.temperature = args.temperature
            
            self.top_cue = args.top_cue
            if 'human' in self.branch:
                human_cue_bank = torch.load(os.path.join(load_path, human_cue_file)).to(args.device)                
                self.human_cue_selector = CueSelectorModule(human_cue_bank[:, :self.top_cue, :])
            if 'object' in self.branch:
                object_cue_bank = torch.load(os.path.join(load_path, object_cue_file)).to(args.device)
                self.object_cue_selector = CueSelectorModule(object_cue_bank[:, :self.top_cue, :])

            if 'context' in self.branch:
                context_cue_bank = torch.load(os.path.join(load_path, context_cue_file)).to(args.device)
                self.context_cue_selector = CueSelectorModule(context_cue_bank[:, :self.top_cue, :])
            
            self.prompt_updater = PromptUpdater(cue_scale=args.cue_scale)
        
        self.use_gaussian_sampler = args.use_gaussian_sampler
        
        if self.use_gaussian_sampler:
            self.gaussian_sampler = GaussianSampler(self.origin_text_embeddings, zs_type=args.zs_type)
            self.noise_scale = args.noise_scale
        
        self.visual_features = defaultdict(list)
        
        
            
    @torch.no_grad()
    def refresh_unseen_verb_cache_mem(self, ):
        assert self.num_classes == 117
        seen_idxs = self.seen_verb_idxs
        unseen_idxs = self.unseen_verb_idxs
        text_embedding = self.origin_text_embeddings
        for i in unseen_idxs:
            cur_logits = (text_embedding[i] @ text_embedding[torch.as_tensor(seen_idxs)].T)
            cur_logits = F.softmax(cur_logits)
            if 'H' in self.logits_type:
                self.adapter_H_weight[i] = cur_logits @ self.adapter_H_weight[torch.as_tensor(seen_idxs)]
            if 'O' in self.logits_type:
                self.adapter_O_weight[i] = cur_logits @ self.adapter_O_weight[torch.as_tensor(seen_idxs)]

    def get_clip_feature(self,image): 
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2) 
        
        local_feat = self.clip_model.visual.transformer.resblocks[:11]((x,None))[0]
        return local_feat
    
    def _reset_parameters(self):
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self, x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor):
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        target_cls_idx = []
        for obj in object_class[y]:
            idx = obj.item()
            if 0 <= idx < len(self.object_class_to_target_class):
                target_cls_idx.append(self.object_class_to_target_class[idx])
            else:
                target_cls_idx.append([])


        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    @torch.no_grad()
    def init_adapter_union_weight(self, device):
        if self.prompt_learning and self.use_text_adapter:
            classnames = self.clip_head.classnames
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in classnames]).to(device)
            with torch.no_grad():
                embedding = self.clip_head.text_encoder.token_embedding(tokenized_prompts).type(self.clip_head.text_encoder.dtype)
            ctx = self.clip_head.prompt_learner.ctx[:, :64] 
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.clip_head.prompt_learner.n_cls, -1, -1)
            prior_mask = None
            prior = (ctx, prior_mask)
            self.adapter_union_weight = self.clip_head.text_encoder(embedding, tokenized_prompts, prior)
        else:
            prompts = self.clip_head.prompt_learner()
            tokenized_prompts = self.clip_head.tokenized_prompts
            self.adapter_union_weight = self.clip_head.text_encoder(prompts, tokenized_prompts)
            self.adapter_union_weight = self.adapter_union_weight.to(device)

    def compute_roi_embeddings(self, features: OrderedDict, image_size: Tensor, region_props: List[dict], feat_global):
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        attn_maps_collated = []
        all_logits = []

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        gt_feats_collated = []
        pair_feats_collated = []
        gt_all_logits = []
        pair_logits = []
        pair_prior = []
        gt_labels = []        
        human_features_collated = []
        object_features_collated = []
        union_features_collated = []

        if self.prompt_learning and self.training and self.use_text_adapter:
            classnames = self.clip_head.classnames
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in classnames]).to(device)
            with torch.no_grad():
                embedding = self.clip_head.text_encoder.token_embedding(tokenized_prompts).type(self.clip_head.text_encoder.dtype)
            ctx = self.clip_head.prompt_learner.ctx[:, :64] ## quick hack: the prior-dim should be 64
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.clip_head.prompt_learner.n_cls, -1, -1)
            prior_mask = None
            prior = (ctx, prior_mask)
            self.adapter_union_weight = self.clip_head.text_encoder(embedding, tokenized_prompts, prior)
        elif self.prompt_learning and self.training and self.prior_method != 4:
            prompts = self.clip_head.prompt_learner()
            tokenized_prompts = self.clip_head.tokenized_prompts
            self.adapter_union_weight = self.clip_head.text_encoder(prompts, tokenized_prompts)
        
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            global_feature = feat_global[b_idx]
            
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                human_features_collated.append(torch.zeros(0, self.visual_output_dim, device=device))
                object_features_collated.append(torch.zeros(0, self.visual_output_dim, device=device))
                union_features_collated.append(torch.zeros(0, self.visual_output_dim, device=device))
                continue

            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            
            if len(x_keep) == 0:
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()
            
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)

            spatial_scale = 1 / (image_size[0,0]/local_features.shape[1])
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            
            if self.feat_mask_type == 0:
                union_features = self.featmap_dropout(union_features).flatten(2).mean(-1)
                single_features = self.featmap_dropout(single_features).flatten(2).mean(-1)
            elif self.feat_mask_type == 1:
                union_features = union_features.flatten(2).mean(-1)
                single_features = single_features.flatten(2).mean(-1)
                                    
            human_features = single_features[x_keep]
            object_features = single_features[y_keep]
            
            if 'S' in self.logits_type:
                img_size = image_size[b_idx]
                pairwise_spatial = compute_spatial_encodings(
                    [boxes[x], ], [boxes[y], ], [img_size, ]
                )
                pairwise_spatial = self.spatial_head(pairwise_spatial)
                pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)
                
                spatial_query = self.mmf(
                    torch.cat([human_features, object_features], dim=1),
                    pairwise_spatial_reshaped[x_keep, y_keep]
                )
            
            if self.individual_norm: 
                human_features = human_features / human_features.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)

            else:
                concat_feat = torch.cat([human_features,object_features, union_features],dim=-1) 
                concat_feat = concat_feat/concat_feat.norm(dim=-1,keepdim=True)             

            if self.logits_type == 'H+O+T':
                logits_H = human_features @ self.adapter_H_weight @ self.adapter_union_weight.T
                logits_O = object_features @ self.adapter_O_weight @ self.adapter_union_weight.T    
                logits_text = union_features @ self.adapter_union_weight.T
                logits = logits_H * self.logit_scale_H + logits_O * self.logit_scale_O + logits_text * self.logit_scale_text 
                
            elif self.logits_type == 'S+T':
                context_features = self.spatial_fusion(torch.cat([union_features, spatial_query], dim=1))

                if self.use_verb_cue_selector:
                    feature_dict = {
                        'human': human_features,
                        'object': object_features,
                        'context': context_features,
                    }
                    selector_dict = {
                        'human': getattr(self, 'human_cue_selector', None),
                        'object': getattr(self, 'object_cue_selector', None),
                        'context': getattr(self, 'context_cue_selector', None),
                    }

                    prompt_outputs = {}
                    weight = self.gaussian_sampler(self.adapter_union_weight, self.noise_scale) if self.use_gaussian_sampler and self.training else self.adapter_union_weight

                    self.last_cue_weights = {}
                    
                    for branch in ['human', 'object', 'context']:
                        selector = selector_dict[branch]
                        feature = feature_dict[branch]

                        if selector is not None:
                            sel, w = selector(feature)
                            self.last_cue_weights[branch] = w.detach().cpu()
                            prompt_outputs[branch] = self.prompt_updater(weight, sel)
                            
                        else:
                            prompt_outputs[branch] = weight 
                            
                    logits_H = torch.sum(human_features.unsqueeze(1) * prompt_outputs['human'], dim=-1)
                    logits_O = torch.sum(object_features.unsqueeze(1) * prompt_outputs['object'], dim=-1)
                    logits_C = torch.sum(context_features.unsqueeze(1) * prompt_outputs['context'], dim=-1)
            
                    logits = (logits_H + logits_O + logits_C) / 3.0
                    
                    self.last_union_boxes = union_boxes.detach().cpu()
                    self.last_h_boxes = boxes[x_keep].detach().cpu()
                    self.last_o_boxes = boxes[y_keep].detach().cpu()
                    
                    self.last_logits = logits.detach().cpu()
                    

                    
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits.append(logits)
            union_features_collated.append(union_features)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated
    
    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)
        
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        
        iou_h = box_iou(boxes_h, gt_bx_h)
        iou_o = box_iou(boxes_o, gt_bx_o)
        match = torch.min(iou_h, iou_o) >= self.fg_iou_thresh
        
        if match.sum() == 0:
            return labels   

        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1
        else:
            labels[x, targets['hoi'][y]] = 1

        return labels
    
    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, gt_feats, objects,):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = torch.cat(logits) 
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        
        objects = torch.cat(objects); objects = objects[x]

        if self.filtered_hoi_idx:
            v_list = y.tolist()
            o_list = objects.tolist()
            pair_ids = list(zip(v_list, o_list))

            hoi_ids = [self.hico_hoi_to_idx.get(pair, -1) for pair in pair_ids]
            hoi_ids = torch.tensor(hoi_ids, device=logits.device)

            filtered = torch.tensor(self.filtered_hoi_idx, device=logits.device)
            mask = ~isin_tensor(hoi_ids, filtered)

            if (~mask).any():
                print(f"Filtered HOI pairs for {self.zs_type}")
            
            logits = logits[mask]
            labels = labels[mask]
            prior = prior[mask]

        n_p = len(torch.nonzero(labels))
        if n_p == 0:
            n_p = 1e9

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier() 
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
            )

        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for b_idx, res in enumerate(results):
            sc, lb, bx = res.values()
            hs = hidden_states[b_idx]

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human

            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
            ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections

    def get_prior(self, region_props, image_size, prior_method): 
        
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            hs = props['hidden_states']

            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)
            
            object_embs = self.object_embedding.to(labels.device)[labels]
            mask[b_idx,:n] = False
            
            if self.prior_type == 'cbe':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
                priors[b_idx,:n,5:self.visual_output_dim+5] = object_embs
            elif self.prior_type == 'cb':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            elif self.prior_type == 'ce':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
                priors[b_idx,:n,1:self.visual_output_dim+1] = object_embs
            elif self.prior_type == 'be':
                priors[b_idx,:n,:4] = boxes
                priors[b_idx,:n,4:self.visual_output_dim+4] = object_embs
            elif self.prior_type == 'c':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
            elif self.prior_type == 'b':
                priors[b_idx,:n,:4] = boxes
            elif self.prior_type == 'e':
                priors[b_idx,:n,:self.visual_output_dim] = object_embs
            else:
                raise NotImplementedError

        if prior_method == 0:
            priors = self.priors_downproj(priors)
        return (priors, mask)
    
    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections

    def prepare_target_hois(self, targets, device):
        unique_hois, cnt = {}, 0
        tgt_ids = []
        for t in targets:
            for hoi in t["hoi"]:
                hoi_id = hoi.item()
                if self.training:
                    if hoi_id not in unique_hois:
                        unique_hois[hoi_id] = cnt
                        cnt += 1
                    tgt_ids.append(unique_hois[hoi_id])
                else:
                    tgt_ids.append(hoi_id)
        tgt_ids = torch.as_tensor(tgt_ids, dtype=torch.int64, device=device)
        return unique_hois
    
    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()  
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe


    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        batch_size = len(images)
        
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]                  
        
        device = images_clip[0].device

        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
        
        with torch.no_grad():
            features, pos = self.detector.backbone(images_orig)
            
            src, mask = features[-1].decompose()            
            hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])

            outputs_class = self.detector.class_embed(hs)
            outputs_coord = self.detector.bbox_embed(hs).sigmoid()  

        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)
        
        region_props = self.prepare_region_proposals(results, hs[-1])
        priors = None

        images_clip = nested_tensor_from_tensor_list(images_clip)
        
        priors = self.get_prior(region_props, image_sizes, self.prior_method)
        feat_global, feat_local = self.clip_head.image_encoder(images_clip.decompose()[0], priors)

        logits, prior, bh, bo, objects, gt_feats, pair_feats = self.compute_roi_embeddings(feat_local, image_sizes, region_props, feat_global)
        boxes = [r['boxes'] for r in region_props]
        
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, gt_feats, objects)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )

            return loss_dict
        
        if len(logits) == 0:
            print(targets)
            return None

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

def get_multi_prompts(classnames):   ## https://github.com/openai/CLIP/blob/main/data/prompts.md, 
    templates = ['a photo of a person {}.',
                'a video of a person {}.',
                'a example of a person {}.',
                'a demonstration of a person {}.',
                'a photo of the person {}.',
                'a video of the person {}.',
                'a example of the person {}.', 
                'a demonstration of the person {}.',
                ]
    hico_texts = [' '.join(name.split(' ')[5:]) for name in classnames]
    all_texts_input = []
    for temp in templates:
        texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
        all_texts_input.append(texts_input)
    all_texts_input = torch.stack(all_texts_input,dim=0)
    return all_texts_input

@torch.no_grad()
def get_origin_text_emb(args, clip_model, tgt_class_names, obj_class_names):
    use_templates = args.use_templates
    if use_templates == False:
        text_inputs = torch.cat([clip.tokenize(classname) for classname in tgt_class_names])
    elif use_templates:
        text_inputs = get_multi_prompts(tgt_class_names)
        bs_t, nums, c = text_inputs.shape
        text_inputs = text_inputs.view(-1, c)

    with torch.no_grad():
        origin_text_embedding = clip_model.encode_text(text_inputs)
    if use_templates:
        origin_text_embedding = origin_text_embedding.view(bs_t, nums, -1).mean(0)

    origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(dim=-1, keepdim=True) 

    obj_text_inputs = torch.cat([clip.tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
    return origin_text_embedding, object_embedding


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, num_anno, verb2interaction=None):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model'], strict=False) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'], strict=False)
    
    clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    clip_model = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, 
                                        adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers,
                                        use_text_adapter=args.use_text_adapter, bottleneck=args.bottleneck)
    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError
    
    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]    
    origin_text_embeddings, object_embedding = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=classnames, obj_class_names=obj_class_names)
    origin_text_embeddings = origin_text_embeddings.clone().detach()
    
    object_embedding = object_embedding.clone().detach()
    
    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)
    detector = UPT(args,
        detr, postprocessors['bbox'], model, origin_text_embeddings, object_embedding,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        num_anno = num_anno,
        featmap_dropout_rate=args.featmap_dropout_rate,
    )
    return detector


