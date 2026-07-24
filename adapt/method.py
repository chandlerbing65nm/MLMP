import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ovss import load_ovss
from optim import get_optimizer
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class METHOD:
    """
    Configurable TTA method for OVSS models.

    Extends TENT-style entropy minimization with selective parameter training:
    optionally train LayerNorm and/or attention layers in the visual encoder,
    restricted to the last K transformer blocks.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=10,
                 prompt_dir=None, runtime_calculation=False, optimizer='adam',
                 reset_mode='episodic', device='cpu',
                 train_imag_norm=True, last_imag_k_norm=0,
                 train_imag_attn=False, last_imag_k_attn=0,
                 train_text_norm=False, last_text_k_norm=0,
                 loss_ent=True, lamb_ent=1.0,
                 loss_div=False, lamb_div=1.0,
                 loss_aug_cons=False, lamb_aug_cons=1.0,
                 loss_src_cons=False, lamb_src_cons=1.0,
                 updownsample=1.0,
                 prompt_average=False,
                 cons_type='sym_kl'):

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")

        self.prompt_dir = prompt_dir
        self.steps = steps
        self.runtime = runtime_calculation
        self.optimizer_name = optimizer
        self.reset_mode = reset_mode
        self.device = device

        self.train_imag_norm = train_imag_norm
        self.last_imag_k_norm = last_imag_k_norm
        self.train_imag_attn = train_imag_attn
        self.last_imag_k_attn = last_imag_k_attn
        self.train_text_norm = train_text_norm
        self.last_text_k_norm = last_text_k_norm
        self.loss_ent = loss_ent
        self.lamb_ent = lamb_ent
        self.loss_div = loss_div
        self.lamb_div = lamb_div
        self.loss_aug_cons = loss_aug_cons
        self.lamb_aug_cons = lamb_aug_cons
        self.loss_src_cons = loss_src_cons
        self.lamb_src_cons = lamb_src_cons
        self.updownsample = updownsample
        self.prompt_average = prompt_average
        self.cons_type = cons_type
        self.eval_size = None

        self.source_model = None

        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        if self.prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(self.prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # Freeze text encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        # Freeze visual encoder entirely, then selectively unfreeze
        self.model.visual.requires_grad_(False)

        params = []
        num_blocks = len(self.model.visual.transformer.resblocks)

        if self.train_imag_norm:
            ln_params = self._collect_ln_params(self.model.visual, num_blocks)
            for p in ln_params:
                p.requires_grad_(True)
            params.extend(ln_params)

        if self.train_imag_attn:
            attn_params = self._collect_attn_params(self.model.visual, num_blocks)
            for p in attn_params:
                p.requires_grad_(True)
            params.extend(attn_params)

        if self.train_text_norm:
            text_num_blocks = len(self.model.transformer.resblocks)
            text_ln_params = self._collect_text_ln_params(text_num_blocks)
            for p in text_ln_params:
                p.requires_grad_(True)
            params.extend(text_ln_params)

        if len(params) == 0:
            raise ValueError("No trainable parameters selected. Enable --train_imag_norm, --train_imag_attn, or --train_text_norm.")

        print_clip_parameters(self.model)
        self.optimizer = get_optimizer(params, optimizer_name=self.optimizer_name, lr=self.lr)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        if self.loss_src_cons:
            self.source_model = copy.deepcopy(self.model)
            self.source_model.requires_grad_(False)
            self.source_model.eval()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def _collect_ln_params(self, visual, num_blocks):
        params = []
        for nm, m in visual.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        if self.last_imag_k_norm > 0:
                            block_idx = self._get_block_index(nm, num_blocks)
                            if block_idx is not None and block_idx < num_blocks - self.last_imag_k_norm:
                                continue
                        params.append(p)
        return params

    def _collect_attn_params(self, visual, num_blocks):
        params = []
        for nm, m in visual.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                for np_name, p in m.named_parameters():
                    if np_name in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                        if self.last_imag_k_attn > 0:
                            block_idx = self._get_block_index(nm, num_blocks)
                            if block_idx is not None and block_idx < num_blocks - self.last_imag_k_attn:
                                continue
                        params.append(p)
        return params

    def _collect_text_ln_params(self, num_blocks):
        params = []
        for nm, m in self.model.transformer.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        if self.last_text_k_norm > 0:
                            block_idx = self._get_text_block_index(nm, num_blocks)
                            if block_idx is not None and block_idx < num_blocks - self.last_text_k_norm:
                                continue
                        params.append(p)
        if self.last_text_k_norm == 0 or num_blocks <= self.last_text_k_norm:
            for np_name, p in self.model.ln_final.named_parameters():
                if np_name in ['weight', 'bias']:
                    params.append(p)
        return params

    @staticmethod
    def _get_text_block_index(module_name, num_blocks):
        for i in range(num_blocks):
            if f'resblocks.{i}.' in module_name:
                return i
        return None

    @staticmethod
    def _get_block_index(module_name, num_blocks):
        for i in range(num_blocks):
            if f'transformer.resblocks.{i}.' in module_name:
                return i
        return None

    def adapt(self, x):
        if self.reset_mode == 'episodic':
            self.reset()
        loss_report = self.perform_adaptation(x)
        return loss_report

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        text_features = self.extract_text_embeddings(self.classes, self.prompt_templates, average=self.prompt_average).squeeze()
        logits, _, _ = self.model(x, text_features, True, interpolate=False)
        # logits: [#templates, batch, #classes, native_h, native_w]

        native_h, native_w = logits.shape[-2], logits.shape[-1]
        input_h, input_w = x.shape[-2], x.shape[-1]
        target_h = round(native_h + (input_h - native_h) * self.updownsample)
        target_w = round(native_w + (input_w - native_w) * self.updownsample)
        self.eval_size = target_h

        if target_h != native_h or target_w != native_w:
            temp_dim, b_dim, c_dim = logits.shape[0], logits.shape[1], logits.shape[2]
            logits = logits.reshape(-1, c_dim, native_h, native_w)
            logits = F.interpolate(logits, size=(target_h, target_w), mode='bilinear', align_corners=False)
            logits = logits.view(temp_dim, b_dim, c_dim, target_h, target_w)

        logits = logits[0]  # [batch, #classes, target_h, target_w]
        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2 - t1)
        return logits

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('Cannot reset without saved model/optimizer state')
        self.load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            text_features = self.extract_text_embeddings(self.classes, self.prompt_templates, average=self.prompt_average).squeeze()
            logits, _, _ = self.model(x, text_features, True, interpolate=False)

            loss = torch.tensor(0.0, device=x.device)

            if self.loss_ent:
                entropy_per_pixel = self.softmax_entropy(logits)
                loss = loss + self.lamb_ent * entropy_per_pixel.mean()

            if self.loss_div:
                div_loss = self.diversity_loss(logits)
                loss = loss + self.lamb_div * div_loss

            if self.loss_aug_cons:
                x_flip = torch.flip(x, dims=[-1])
                logits_flip, _, _ = self.model(x_flip, text_features, True, interpolate=False)
                logits_flip = torch.flip(logits_flip, dims=[-1])
                cons_loss = self.consistency_loss(logits, logits_flip)
                loss = loss + self.lamb_aug_cons * cons_loss

            if self.loss_src_cons:
                with torch.no_grad():
                    logits_src, _, _ = self.source_model(x, text_features, True, interpolate=False)
                src_cons_loss = self.consistency_loss(logits, logits_src)
                loss = loss + self.lamb_src_cons * src_cons_loss

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        t2 = time.time()
        if self.runtime:
            self.adapt_times.append(t2 - t1)
        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

    @staticmethod
    def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
        """Class-wise diversity loss to prevent model collapse.

        Maximizes the entropy of the marginal class distribution across pixels,
        preventing all pixels from collapsing to a single class.

        Args:
            logits: [#templates, batch, #classes, W, H]
        Returns:
            scalar loss (minimize to maximize class diversity)
        """
        p = logits.softmax(dim=-3)
        marginal = p.mean(dim=(1, 3, 4))  # [#templates, #classes]
        entropy = -(marginal * torch.log(marginal + 1e-8)).sum(dim=-1)  # [#templates]
        return -entropy.mean()

    def consistency_loss(self, logits_orig: torch.Tensor, logits_aug: torch.Tensor) -> torch.Tensor:
        """Pixel-wise consistency loss via KL divergence.

        Enforces prediction invariance between original and augmented/source
        predictions, stabilizing continual TTA by preventing drift.

        Args:
            logits_orig: [#templates, batch, #classes, W, H]
            logits_aug: [#templates, batch, #classes, W, H] (already de-augmented)
        Returns:
            scalar loss
        """
        p = logits_orig.softmax(dim=-3)
        q = logits_aug.softmax(dim=-3)
        log_p = logits_orig.log_softmax(dim=-3)
        log_q = logits_aug.log_softmax(dim=-3)
        kl_pq = (p * (log_p - log_q)).sum(dim=-3)
        kl_qp = (q * (log_q - log_p)).sum(dim=-3)
        if self.cons_type == 'for_kl':
            return kl_pq.mean()
        elif self.cons_type == 'rev_kl':
            return kl_qp.mean()
        else:
            return 0.5 * (kl_pq + kl_qp).mean()
