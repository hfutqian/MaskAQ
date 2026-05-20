import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import resolve_data_config

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from einops import rearrange

import random
import numpy as np

from image_gen_aug_utils import ColorJitter, GaussianNoise


class hook_class():
    def __init__(self):
        self.matmul_output = []
        
    def hook_fn_forward(self,module, input, output):
        self.matmul_output.append(output)

def attention_entropy_loss_similarity(att_maps, is_swin=False,
                                      patch_subsample_k=64,
                                      eps=1e-8):

    if not att_maps:
        return torch.tensor(0.0).cuda()

    device = att_maps[0].device
    dtype = att_maps[0].dtype
    two_pi_e_log = torch.log(torch.tensor(2.0 * np.pi * np.e, device=device, dtype=dtype))

    total_h = 0.0
    layer_count = 0

    for att in att_maps:
        if is_swin:
            if att.dim() == 4:
                att_mean = att.mean(dim=1)
                attention_p = att_mean
                if attention_p.dim() == 2:
                    attention_p = attention_p.unsqueeze(-1)
            else:
                attention_p = att
        else:
            if att.dim() == 4 and att.shape[2] == att.shape[3]:
                att_mean = att.mean(dim=1)
                if att_mean.shape[1] > 1:
                    attention_p = att_mean[:, 1:, 1:]
                else:
                    attention_p = att_mean
            else:
                attention_p = att

        if attention_p.dim() == 3:
            B, N, _ = attention_p.shape
            for b in range(B):
                vecs = attention_p[b]
                if vecs.numel() == 0:
                    continue
                n = vecs.shape[0]
                if n > patch_subsample_k:
                    idx = torch.randperm(n, device=device)[:patch_subsample_k]
                    vecs = vecs[idx]
                    n = vecs.shape[0]
                vecs = F.normalize(vecs, p=2, dim=-1, eps=1e-6)
                sims = torch.matmul(vecs, vecs.t())
                sims_flat = sims.view(-1)
                var = torch.var(sims_flat, unbiased=False) + eps
                h = 0.5 * (two_pi_e_log + torch.log(var))
                total_h += h
                layer_count += 1

        elif attention_p.dim() == 2:
            B, N = attention_p.shape
            for b in range(B):
                vecs = attention_p[b].unsqueeze(-1)
                if vecs.numel() == 0:
                    continue
                n = vecs.shape[0]
                if n > patch_subsample_k:
                    idx = torch.randperm(n, device=device)[:patch_subsample_k]
                    vecs = vecs[idx]
                    n = vecs.shape[0]
                vecs = F.normalize(vecs, p=2, dim=-1, eps=1e-6)
                sims = torch.matmul(vecs, vecs.t())
                sims_flat = sims.view(-1)
                var = torch.var(sims_flat, unbiased=False) + eps
                h = 0.5 * (two_pi_e_log + torch.log(var))
                total_h += h
                layer_count += 1
        else:
            flat = attention_p.view(attention_p.shape[0], -1)
            B, _ = flat.shape
            for b in range(B):
                vecs = flat[b].unsqueeze(-1)
                if vecs.numel() == 0:
                    continue
                n = vecs.shape[0]
                if n > patch_subsample_k:
                    idx = torch.randperm(n, device=device)[:patch_subsample_k]
                    vecs = vecs[idx]
                    n = vecs.shape[0]
                vecs = F.normalize(vecs, p=2, dim=-1, eps=1e-6)
                sims = torch.matmul(vecs, vecs.t())
                sims_flat = sims.view(-1)
                var = torch.var(sims_flat, unbiased=False) + eps
                h = 0.5 * (two_pi_e_log + torch.log(var))
                total_h += h
                layer_count += 1

    if layer_count == 0:
        return torch.tensor(0.0, device=device)
    avg_h = total_h / float(layer_count)
    return -avg_h

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3)/np.sqrt(2) + torch.norm(diff4)/np.sqrt(2)
    return loss_var_l2


def stochastic_mask_drop(mask, drop_prob=0.3, min_keep=1, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
    device = mask.device
    dtype = mask.dtype
    B, _ = mask.shape
    new_mask = mask.clone().to(device)
    for b in range(B):
        m = new_mask[b]
        sel_idx = torch.nonzero(m > 0.5).squeeze(-1)
        if sel_idx.numel() == 0:
            continue
        keep_num = max(min_keep, int(round(sel_idx.numel() * (1.0 - drop_prob))))
        keep_num = min(keep_num, sel_idx.numel())
        if keep_num == sel_idx.numel():
            continue
        perm = torch.randperm(sel_idx.numel(), device=device)
        keep_idx = sel_idx[perm[:keep_num]]
        m[sel_idx] = 0.0
        m[keep_idx] = 1.0
        new_mask[b] = m
    return new_mask.type(dtype)

    
def generate_attention_mask(att_maps, ratio=0.3, is_swin=False):
    if is_swin:
        combined_att = torch.stack(att_maps).mean(0)
        if len(combined_att.shape) == 3:
            combined_att = combined_att.mean(1)
        else:
            combined_att = combined_att.mean(1)
    else:
        if len(att_maps) > 0 and len(att_maps[0].shape) == 4:
            combined_att = torch.stack(att_maps).mean(0)
            combined_att = combined_att.mean(1)
            combined_att = combined_att[:, 0, 1:]
        else:
            combined_att = torch.stack(att_maps).mean(0)
            combined_att = combined_att.flatten(1)
    
    B = combined_att.shape[0]
    masks = []
    
    for b in range(B):
        att_b = combined_att[b]
        if len(att_b.shape) > 1:
            att_b = att_b.flatten()
        
        k = max(1, int(len(att_b) * ratio))
        _, top_indices = torch.topk(att_b, k)
        
        mask = torch.zeros_like(att_b)
        mask[top_indices] = 1.0
        masks.append(mask)
    
    return torch.stack(masks)

def masked_attention_difference_loss(att_maps_full, att_maps_quant, mask, is_swin=False):
    total_loss = 0.0
    
    for att_f, att_q in zip(att_maps_full, att_maps_quant):
        if is_swin:
            att_f_flat = att_f.flatten(1)
            att_q_flat = att_q.flatten(1)
            
            if mask.shape[1] != att_f_flat.shape[1]:
                mask_resized = F.interpolate(
                    mask.unsqueeze(1).float(), 
                    size=att_f_flat.shape[1], 
                    mode='nearest'
                ).squeeze(1)
            else:
                mask_resized = mask
        else:
            if len(att_f.shape) == 4:
                att_f_mean = att_f.mean(1)
                att_q_mean = att_q.mean(1)
                
                att_f_cls = att_f_mean[:, 0, 1:]
                att_q_cls = att_q_mean[:, 0, 1:]
                
                if mask.shape[1] != att_f_cls.shape[1]:
                    mask_resized = F.interpolate(
                        mask.unsqueeze(1).float(),
                        size=att_f_cls.shape[1],
                        mode='nearest'
                    ).squeeze(1)
                else:
                    mask_resized = mask
                
                masked_diff = torch.abs(att_f_cls - att_q_cls) * mask_resized
                total_loss += masked_diff.sum() / (mask_resized.sum() + 1e-8)
            else:
                att_f_flat = att_f.flatten(1)
                att_q_flat = att_q.flatten(1)
                
                if mask.shape[1] != att_f_flat.shape[1]:
                    mask_resized = F.interpolate(
                        mask.unsqueeze(1).float(),
                        size=att_f_flat.shape[1],
                        mode='nearest'
                    ).squeeze(1)
                else:
                    mask_resized = mask
                
                masked_diff = torch.abs(att_f_flat - att_q_flat) * mask_resized
                total_loss += masked_diff.sum() / (mask_resized.sum() + 1e-8)
    
    return total_loss / len(att_maps_full)

def update_mask_ratio(current_iter, total_iter, initial_ratio=0.5, final_ratio=0.1):
    progress = current_iter / total_iter
    current_ratio = initial_ratio * (1 - progress) + final_ratio * progress
    return max(current_ratio, final_ratio)

def ssim_loss_att_map_swin(att_hook_output, mode='full_map'):
    
    ssim_map = []
    
    for att in att_hook_output:
        att_map = rearrange(att, "B H N d -> (B N) H d")
        att_map = att_map - att_map.min(-1, keepdim=True)[0]
        att_map = att_map / (att_map.max(-1, keepdim=True)[0] + 1e-8)
        
        att_map = rearrange(att_map, "B H (w h) -> B H w h", w=7, h=7)
        mu = att_map.mean([-2,-1])
        
        mu_sq = (mu.pow(2))
        mu1_mu2 = att_map.mean([-2,-1]).unsqueeze(1)*att_map.mean([-2,-1]).unsqueeze(2)

        sigma_sq = (att_map*att_map).mean([-2,-1])-mu_sq
        sigma12 = (att_map.unsqueeze(1)*att_map.unsqueeze(2)).mean([-2,-1])-mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2
        
        one_layer_ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu_sq.unsqueeze(1) + mu_sq.unsqueeze(2) + C1)*(sigma_sq.unsqueeze(1) + sigma_sq.unsqueeze(2) + C2))
        
        if mode == "full_map":
            ssim_map.append(one_layer_ssim_map)
        elif mode == 'loss':
            ssim_map.append((1-one_layer_ssim_map**2).mean())
        else:
            assert('invalid mode for SSIM')
    return ssim_map
        
def ssim_loss_att_map(att_hook_output, mode='full_map', self_zero=False):
    hook_stack = torch.stack(att_hook_output)[:,:,:,1:,1:197]
    hook_stack = rearrange(hook_stack, "L B H N d -> L (B N) H d")
    hook_stack = hook_stack - hook_stack.min(-1, keepdim=True)[0]
    hook_stack = hook_stack / (hook_stack.max(-1, keepdim=True)[0] + 1e-8)

    
    att_map = rearrange(hook_stack, "L B H (w h) -> L B H w h", w=14,h=14)
    mu = att_map.mean([-2,-1])

    mu_sq = (mu.pow(2))
    mu1_mu2 = att_map.mean([-2,-1]).unsqueeze(2)*att_map.mean([-2,-1]).unsqueeze(3)
    
    sigma_sq = (att_map*att_map).mean([-2,-1])-mu_sq
    sigma12 = (att_map.unsqueeze(2)*att_map.unsqueeze(3)).mean([-2,-1])-mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu_sq.unsqueeze(2) + mu_sq.unsqueeze(3) + C1)*(sigma_sq.unsqueeze(2) + sigma_sq.unsqueeze(3) + C2))
    if mode == 'loss':
        return ssim_map.mean() 
    elif mode == 'full_map':
        if self_zero:
            ssim_map = ssim_map - torch.eye(ssim_map.shape[-1], device=ssim_map.device).unsqueeze(0).unsqueeze(0)
        return ssim_map
    else:
        assert('invalid mode for SSIM')
        
        

def generate(epoch, model, model_origin, model_quant, batch_size, iter, num_images, save_dir, 
             ssim_coef=1.0, class_coef=1.0, tv_coef=2.5e-5, 
             att_diff_coef=0.1,
             initial_mask_ratio=0.5, final_mask_ratio=0.1,
             similarity_entropy_coef=1.0):
    
    model_stats = resolve_data_config(dict(),model=model_origin, verbose=True)
    
    mu = torch.tensor(model_stats["mean"],device='cuda')
    std = torch.tensor(model_stats["std"],device='cuda')
    
    
    IMG_SIZE = model_origin.patch_embed.img_size[0]
    aug_list = []
    args_no_gs = False
    args_no_cj = False
    if not args_no_gs:
        aug_list.append(GaussianNoise(batch_size, True, 0.5, iter))
    if not args_no_cj:
        aug_list.append(ColorJitter(batch_size, True))
        
    if not aug_list:
        aug_list.append(nn.Identity())
        
    aug = torch.nn.Sequential(*aug_list)
    
    is_swin = 'swin' in model
    
    hook = hook_class()
    hook_quant = hook_class()

    head_hook_handles = []
    head_hook_handles_quant = []
    
    head_hook_handles.clear()
    head_hook_handles_quant.clear()
    
    for m in model_origin.modules():
        if isinstance(m, Attention):
            handle = m.matmul1.register_forward_hook(hook.hook_fn_forward)
            head_hook_handles.append(handle)
        if isinstance(m, WindowAttention):
            handle = m.matmul1.register_forward_hook(hook.hook_fn_forward)
            head_hook_handles.append(handle)
    
    for m in model_quant.modules():
        if isinstance(m, Attention):
            handle = m.matmul1.register_forward_hook(hook_quant.hook_fn_forward)
            head_hook_handles_quant.append(handle)
        if isinstance(m, WindowAttention):
            handle = m.matmul1.register_forward_hook(hook_quant.hook_fn_forward)
            head_hook_handles_quant.append(handle)

    for batch in range(num_images//batch_size):
        img_train = torch.randn((batch_size,3,IMG_SIZE,IMG_SIZE)).cuda()
        img_train.requires_grad = True
        opt = torch.optim.AdamW([img_train], lr=0.1, betas=[0.9,0.999], eps=1e-8, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iter, 0.001)

        class_label = torch.randint(0,1000,(batch_size,)).cuda() 
        
        NAN_FLAG = False
        
        for i in range(iter):
            
            if i < iter/2:
                JITTER = 8
            else:
                JITTER = 8*2
            hook.matmul_output.clear() 
            hook_quant.matmul_output.clear()


            opt.zero_grad()

            current_mask_ratio = update_mask_ratio(i, iter, initial_mask_ratio, final_mask_ratio)
            
            lim_0, lim_1 = JITTER, JITTER
            
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            img_jit = torch.roll(img_train, shifts=(off1, off2), dims=(2, 3))
            
            args_flip = False
            if args_flip and random.random() > 0.5:
                img_jit = torch.flip(img_jit, dims=(3,))
                
            img_jit = aug(img_jit)
            
            output = model_origin(img_jit)
            model_quant(img_jit)

            class_loss = torch.zeros(1).cuda()
            ssim_loss = torch.zeros(1).cuda()
            att_diff_loss = torch.zeros(1).cuda()
            similarity_entropy_loss = torch.zeros(1).cuda() 
            
            if "swin" in model:
                ssim_loss = torch.stack(ssim_loss_att_map_swin(hook.matmul_output, mode='loss')).mean()
            else:
                ssim_map = ssim_loss_att_map(hook.matmul_output, mode="full_map")
                
                ssim_loss = (1-(ssim_map)**2).mean()

            attention_mask = generate_attention_mask(hook.matmul_output, ratio=current_mask_ratio, is_swin=is_swin)
            attention_mask = stochastic_mask_drop(attention_mask, drop_prob=0.30, min_keep=1)

            att_diff_loss = masked_attention_difference_loss(
                hook.matmul_output, hook_quant.matmul_output, attention_mask, is_swin=is_swin
            )
            
            class_loss = F.cross_entropy(output,class_label)
            
            loss_tv = get_image_prior_losses(img_train)
            similarity_entropy_loss = attention_entropy_loss_similarity(hook.matmul_output, is_swin=is_swin)

            total_loss = (ssim_coef*ssim_loss + 
                         class_coef*class_loss + 
                         tv_coef*loss_tv + 
                         att_diff_coef*att_diff_loss +
                         similarity_entropy_coef*similarity_entropy_loss)
                
            total_loss.backward()
            opt.step()
            lr_scheduler.step()
            
            clamp_training = True
            if clamp_training:
                img_train.data = torch.clamp(img_train.data, (-mu/std).view(-1,1,1), ((1-mu)/std).view(-1,1,1))
            
            if torch.isnan(img_train).sum()>0:
                NAN_FLAG=True
                break

            del img_jit, output, total_loss, ssim_loss, class_loss, loss_tv, att_diff_loss, similarity_entropy_loss, attention_mask
                
        save_prefix = 0
        if not NAN_FLAG:
            torch.save(img_train.detach().cpu(), os.path.join(save_dir,f"{save_prefix+batch}_images.pt"))
            torch.save(class_label.detach().cpu(), os.path.join(save_dir,f"{save_prefix+batch}_class_labels.pt"))



    for h in head_hook_handles:
        h.remove()
    for h in head_hook_handles_quant:
        h.remove()

    hook.matmul_output.clear() 
    hook_quant.matmul_output.clear()




    
