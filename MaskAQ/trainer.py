import os

import torch.nn as nn
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from einops import rearrange

from torchvision import transforms
from gaussian_blur import GaussianBlur


from hydra_image_gen_ssim_att_map import generate
from hydra_image_gen_merge import gen_merge

from torch.optim.lr_scheduler import CosineAnnealingLR



__all__ = ["Trainer"]


def ssim_dist_loss_swin(att_hook_output_teacher, att_hook_output_quant, mode='loss',distance='ssim'):

    ssim_map = []

    for att_t, att_q in zip(att_hook_output_teacher, att_hook_output_quant):
        att_map_t = rearrange(att_t, "B H (w h) d -> B d H (w h)", w=7, h=7)
        att_map_q = rearrange(att_q, "B H (w h) d -> B d H (w h)", w=7, h=7)
        
        if distance == 'ssim':
            mu_t = att_map_t.mean(-1)
            mu_q = att_map_q.mean(-1)
            
            mu_t_sq = mu_t.pow(2)
            mu_q_sq = mu_q.pow(2)
            
            mut_muq = mu_t*mu_q
            
            sigma_sq_t = (att_map_t*att_map_t - mu_t_sq.unsqueeze(-1)).mean(dim=-1)
            sigma_sq_q = (att_map_q*att_map_q - mu_q_sq.unsqueeze(-1)).mean(dim=-1)
            
            sigma_tq = (att_map_t*att_map_q - mut_muq.unsqueeze(-1)).mean(dim=-1)
        
            C2 = 0.03**2
            
            ssim_map_single_layer = ((2*sigma_tq + C2)) / ((sigma_sq_t+sigma_sq_q +C2))
            
            if mode == 'loss':
                ssim_map.append((1-ssim_map_single_layer.mean()))
            elif mode == 'full_map':
                ssim_map.append(ssim_map_single_layer)
        elif distance == 'mse':
            ssim_map.append(F.mse_loss(att_map_t,att_map_q))
        elif distance == 'kl':
            ssim_map.append(F.kl_div(F.log_softmax(att_map_q,dim=-1), F.softmax(att_map_t)))
        elif distance == 'l1':
            return F.l1_loss(att_map_t,att_map_q)
                
    return torch.stack(ssim_map).mean()
 

def ssim_dist_loss(att_hook_output_teacher, att_hook_output_quant, mode='loss',distance='ssim'):
    hook_stack_teacher = torch.stack(att_hook_output_teacher)[:,:,:,1:]
    att_map_teacher = rearrange(hook_stack_teacher, "L B H N d -> L B d H N")
    
    hook_stack_quant = torch.stack(att_hook_output_quant)[:,:,:,1:]
    att_map_quant = rearrange(hook_stack_quant, "L B H N d -> L B d H N")
    
    if distance == 'ssim':
        mu_t = att_map_teacher.mean(-1)
        mu_q = att_map_quant.mean(-1)
        
        mu_t_sq = mu_t.pow(2)
        mu_q_sq = mu_q.pow(2)
        
        mut_muq = mu_t*mu_q
        
        sigma_sq_t = (att_map_teacher*att_map_teacher - mu_t_sq.unsqueeze(-1)).mean(dim=-1)
        sigma_sq_q = (att_map_quant*att_map_quant - mu_q_sq.unsqueeze(-1)).mean(dim=-1)
        
        sigma_tq = (att_map_teacher*att_map_quant - mut_muq.unsqueeze(-1)).mean(dim=-1)
        
        C2 = 0.03**2
        
        ssim_map = ((2*sigma_tq + C2)) / ((sigma_sq_t+sigma_sq_q +C2))

        if mode == 'loss':
            return 1-ssim_map.mean() 
        elif mode == 'full_map':
            return 1-ssim_map
        else:
            assert('invalid mode for SSIM')
    elif distance == 'mse':
        return F.mse_loss(att_map_teacher,att_map_quant)
    elif distance == 'kl':
        return F.kl_div(F.log_softmax(att_map_quant,dim=-1),F.softmax(att_map_teacher,dim=-1))
    elif distance == 'l1':
        return F.l1_loss(att_map_teacher,att_map_quant)


_eps = 1e-6

def make_attention_mask_from_teacher(att_hook_output_teacher, proportion=0.2, mode='topk', agg_heads='mean'):

    if isinstance(att_hook_output_teacher, list):
        stacked = torch.stack(att_hook_output_teacher)
    else:
        stacked = att_hook_output_teacher

    L, B, _, N, _ = stacked.shape
    importance = stacked.abs().mean(dim=-1)

    if agg_heads == 'mean':
        importance_pos = importance.mean(dim=2)
    elif agg_heads == 'max':
        importance_pos, _ = importance.max(dim=2)
    else:
        importance_pos = importance.mean(dim=2)

    device = importance_pos.device
    mask = torch.zeros_like(importance_pos, device=device)

    if mode == 'topk':
        k = max(1, int(round(N * float(proportion))))
        for li in range(L):
            for bi in range(B):
                vals = importance_pos[li, bi]
                if k >= N:
                    mask[li, bi] = 1.0
                    continue
                topk_idx = torch.topk(vals, k=k, largest=True, sorted=False).indices
                mask[li, bi, topk_idx] = 1.0
    elif mode == 'threshold':
        thresh = torch.quantile(importance_pos.view(-1), 1.0 - proportion)
        mask = (importance_pos >= thresh).float()
    else:
        raise ValueError("unknown mode for make_attention_mask_from_teacher")

    return mask


def _compute_weighted_stats(att_map, weight_map, dim=-1):
    weight = weight_map + _eps
    w_sum = weight.sum(dim=dim, keepdim=True)
    mean = (att_map * weight).sum(dim=dim, keepdim=True) / w_sum
    ex2 = ((att_map * att_map) * weight).sum(dim=dim, keepdim=True) / w_sum
    return mean.squeeze(dim), ex2.squeeze(dim)


def ssim_dist_loss_masked(att_hook_output_teacher, att_hook_output_quant,
                          mask_proportion=0.2, weight_high=2.0, weight_low=1.0,
                          mode='loss', distance='ssim', mask_mode='topk'):

    if isinstance(att_hook_output_teacher, list):
        t_stack = torch.stack(att_hook_output_teacher)
        q_stack = torch.stack(att_hook_output_quant)
    else:
        t_stack = att_hook_output_teacher
        q_stack = att_hook_output_quant

    t_stack = t_stack[:, :, :, 1:, :]
    q_stack = q_stack[:, :, :, 1:, :]

    _, _, H, N, d = t_stack.shape

    mask_binary = make_attention_mask_from_teacher(t_stack, proportion=mask_proportion, mode=mask_mode, agg_heads='mean')
    device = t_stack.device
    mask_binary = mask_binary.to(device)
    weight_vals = mask_binary * (weight_high - weight_low) + weight_low
    hook_stack_teacher = t_stack
    att_map_teacher = rearrange(hook_stack_teacher, "L B H N d -> L B d H N")
    hook_stack_quant = q_stack
    att_map_quant = rearrange(hook_stack_quant, "L B H N d -> L B d H N")
    weight_map = weight_vals[:, :, None, None, :].expand(-1, -1, d, H, -1)

    if distance == 'ssim':
        mu_t, ex2_t = _compute_weighted_stats(att_map_teacher, weight_map, dim=-1)
        mu_q, ex2_q = _compute_weighted_stats(att_map_quant, weight_map, dim=-1)
        sigma_sq_t = ex2_t - mu_t * mu_t
        sigma_sq_q = ex2_q - mu_q * mu_q
        ex_tq = ((att_map_teacher * att_map_quant) * weight_map).sum(dim=-1) / (weight_map.sum(dim=-1) + _eps)
        sigma_tq = ex_tq - mu_t * mu_q

        C2 = 0.03 ** 2

        ssim_map = ((2.0 * sigma_tq + C2)) / ((sigma_sq_t + sigma_sq_q + C2) + _eps)

        if mode == 'loss':
            return (1.0 - ssim_map.mean())
        elif mode == 'full_map':
            return 1.0 - ssim_map
        else:
            raise AssertionError("invalid mode for masked SSIM")
    elif distance == 'mse':
        diff2 = (att_map_teacher - att_map_quant) ** 2
        mse_weighted = (diff2 * weight_map).sum(dim=-1) / (weight_map.sum(dim=-1) + _eps)
        return mse_weighted.mean()
    elif distance == 'kl':
        t_logits = att_map_teacher
        q_logits = att_map_quant
        t_prob = F.softmax(t_logits, dim=-1)
        q_logprob = F.log_softmax(q_logits, dim=-1)
        kl_pos = F.kl_div(q_logprob, t_prob, reduction='none')
        kl_weighted = (kl_pos * weight_map).sum(dim=-1) / (weight_map.sum(dim=-1) + _eps)
        return kl_weighted.mean()
    elif distance == 'l1':
        l1 = (att_map_teacher - att_map_quant).abs()
        l1_weighted = (l1 * weight_map).sum(dim=-1) / (weight_map.sum(dim=-1) + _eps)
        return l1_weighted.mean()
    else:
        raise ValueError("unknown distance for masked ssim loss")


def ssim_dist_loss_swin_masked(att_hook_output_teacher, att_hook_output_quant,
                               mask_proportion=0.2, weight_high=2.0, weight_low=1.0,
                               mode='loss', distance='ssim', mask_mode='topk'):

    L = len(att_hook_output_teacher)
    layer_ssim_losses = []
    for li in range(L):
        att_t = att_hook_output_teacher[li]
        att_q = att_hook_output_quant[li]
        mask = make_attention_mask_from_teacher([att_t], proportion=mask_proportion, mode=mask_mode, agg_heads='mean')
        mask = mask[0].to(att_t.device)

        att_map_t = rearrange(att_t, "B H N d -> B d H N")
        att_map_q = rearrange(att_q, "B H N d -> B d H N")
        _, d, H, N = att_map_t.shape

        weight_vals = mask * (weight_high - weight_low) + weight_low
        weight_map = weight_vals[:, None, None, :].expand(-1, d, H, -1)

        if distance == 'ssim':
            mu_t, ex2_t = _compute_weighted_stats(att_map_t, weight_map, dim=-1)
            mu_q, ex2_q = _compute_weighted_stats(att_map_q, weight_map, dim=-1)
            sigma_sq_t = ex2_t - mu_t * mu_t
            sigma_sq_q = ex2_q - mu_q * mu_q
            ex_tq = ((att_map_t * att_map_q) * weight_map).sum(dim=-1) / (weight_map.sum(dim=-1) + _eps)
            sigma_tq = ex_tq - mu_t * mu_q

            C2 = 0.03 ** 2
            ssim_map = ((2.0 * sigma_tq + C2)) / ((sigma_sq_t + sigma_sq_q + C2) + _eps)
            layer_loss = (1.0 - ssim_map).mean()
        elif distance == 'mse':
            diff2 = (att_map_t - att_map_q) ** 2
            mse_weighted = (diff2 * weight_map).sum(dim=-1) / (weight_map.sum(dim=-1) + _eps)
            layer_loss = mse_weighted.mean()
        else:
            layer_loss = F.mse_loss(att_map_t, att_map_q)
        layer_ssim_losses.append(layer_loss)

    return torch.stack(layer_ssim_losses).mean()


    
class Trainer(object):
    
    def __init__(self, model, model_teacher, lr_master_S, 
                train_loader, test_loader, settings, logger, tensorboard_logger=None,
                opt_type="SGD", optimizer_state=None, run_count=0):
        self.settings = settings
        
        self.model = utils.data_parallel(
            model, self.settings.nGPU, self.settings.GPU)
        self.model_teacher = utils.data_parallel(
            model_teacher, self.settings.nGPU, self.settings.GPU)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tensorboard_logger = tensorboard_logger
        self.lr_master_S = lr_master_S
        self.opt_type = opt_type
        if opt_type == "SGD":
            self.optimizer_S = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,
            )
        elif opt_type == "RMSProp":
            self.optimizer_S = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum
            )
        elif opt_type == "Adam":
            self.optimizer_S = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay
            )
        else:
            assert False, "invalid type: %d" % opt_type
        if optimizer_state is not None:
            self.optimizer_S.load_state_dict(optimizer_state)

        self.scheduler = CosineAnnealingLR(
            self.optimizer_S, 
            T_max=self.settings.nEpochs,  
            eta_min=self.lr_master_S.lr * 0.01,    
            last_epoch=-1                 
        )


        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.matmul2_output = []
        self.matmul2_output_quant = []
        self.teacher_hooks = []
        self.student_hooks = []

        self.fix_G = False
        
        denormalize = transforms.Compose([ transforms.Normalize(mean=torch.tensor([0.,0.,0.,]),
                                                                std=1/torch.tensor(self.settings.data_config["std"])),
                                          transforms.Normalize(mean=-torch.tensor(self.settings.data_config["mean"]),
                                                               std=torch.tensor([1.,1.,1.]))
        ])
        normalize = transforms.Compose([
            transforms.Normalize(mean=torch.tensor(self.settings.data_config["mean"]),
                                 std=torch.tensor(self.settings.data_config["std"]))
        ])
        
        s=1
        color_jitter = transforms.Compose([
            denormalize,
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            normalize
        ])
        if self.settings.real_data:
            transform_list = [
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * self.settings.img_size)).cuda()
            ]
        else:
            transform_list = [
                transforms.RandomResizedCrop(size=self.settings.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * self.settings.img_size)).cuda()
            ]
            
        self.data_transforms = transforms.Compose(transform_list)

    def update_lr(self, epoch):

        old_lr = self.optimizer_S.param_groups[0]['lr']
        
        self.scheduler.step()
        
        current_lr = self.optimizer_S.param_groups[0]['lr']
        
        print(f"Epoch {epoch + 1}: Learning rate updated from {old_lr:.6f} to {current_lr:.6f}")
        
        if self.tensorboard_logger is not None:
            self.scalar_info['learning_rate'] = current_lr
            
        return current_lr

    def get_current_lr(self):
        return self.optimizer_S.param_groups[0]['lr']
    
    def print_lr_schedule(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.settings.nEpochs
            
        print("Learning Rate Schedule:")
        print("-" * 40)
        temp_scheduler = CosineAnnealingLR(
            torch.optim.SGD([torch.tensor(1.0)], lr=self.lr_master_S.lr),
            T_max=num_epochs,
            eta_min=self.lr_master_S.lr * 0.01
        )
        
        for epoch in range(num_epochs):
            lr = temp_scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1:3d}: {lr:.6f}")
            temp_scheduler.step()
        print("-" * 40)

    
    def loss_fn_kd(self, output, labels, teacher_outputs):
        criterion_d = nn.CrossEntropyLoss().cuda()
        kdloss = nn.KLDivLoss(reduction="batchmean").cuda()

        alpha = self.settings.alpha
        T = self.settings.temperature
        a = F.log_softmax(output / T, dim=1)
        b = F.softmax(teacher_outputs / T, dim=1)
        c = (alpha * T * T)
        d = criterion_d(output, labels)

        KD_loss = self.settings.kd_scale*kdloss(a,b)*c + self.settings.ce_scale*d
        return KD_loss

    
    def forward(self, images, teacher_outputs, labels=None):
        output = self.model(images)
        if labels is not None:
            loss = self.loss_fn_kd(output, labels, teacher_outputs)
            return output, loss
        else:
            return output, None

    def backward_S(self, loss_S):
        self.optimizer_S.zero_grad()
        loss_S.backward()
        self.optimizer_S.step()

        
    def hook_fn_forward(self,module, input, output):
        self.matmul2_output.append(output)
        
    def hook_fn_forward_quant(self,module, input, output):
        self.matmul2_output_quant.append(output)





    
    def train(self, epoch):
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = (self.settings.num_samples // self.settings.batchSize)+1 

        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        ssim_coef = 1.0
        class_coef = 1.0
        tv_coef = 2.5e-5
        save_dir = os.path.join('./gen_images_raw',f"{self.settings.network}_ssim_{ssim_coef}_class_{class_coef}_tv_{tv_coef}")
        os.makedirs(save_dir, exist_ok=True)
        dataset_path = os.path.join(self.settings.dataset_path, f"{self.settings.network}_ssim_{self.settings.img_opt_ssim}_class_{self.settings.img_opt_cls}_tv_{self.settings.img_opt_tv}_merged.pt")
        if epoch % 30 == 0:
            generate(epoch=epoch, model=self.settings.network, model_origin=self.model_teacher, model_quant=self.model, batch_size=self.settings.batchSize, iter=2000, num_images=10000, save_dir=save_dir, ssim_coef=1.0, class_coef=1.0, tv_coef=2.5e-5)         
            gen_merge(model=self.settings.network, SAVE_PATH='./gen_images_raw')

        dataset_full = torch.load(dataset_path)
                
        if len(dataset_full) > self.settings.num_samples:
            dataset, _ = torch.utils.data.random_split(dataset_full,[self.settings.num_samples, len(dataset_full)-self.settings.num_samples])
        else:
            dataset = dataset_full
                            
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size = self.settings.batchSize,
                                                                num_workers=self.settings.nThreads, shuffle=True)

        self.optimizer_S.zero_grad()
        
        if self.settings.head_dist_coef > 0.0:
            for m in self.model_teacher.modules():
                if isinstance(m, Attention) or isinstance(m, WindowAttention):
                    self.teacher_hooks.append(m.matmul2.register_forward_hook(self.hook_fn_forward))
        
        if self.settings.head_dist_coef > 0.0:
            for m in self.model.modules():
                if isinstance(m, Attention) or isinstance(m, WindowAttention):
                    self.student_hooks.append(m.matmul2.register_forward_hook(self.hook_fn_forward_quant))
        
        for i, (images, class_labels) in enumerate(self.train_loader):
            
            if self.settings.real_data:
                if i > iters:
                    break
                
            if self.settings.random_samples:
                images = torch.rand_like(images).cuda()
            else:
                images = images.cuda()
                    
            images = self.data_transforms(images)
                
                    
            if torch.isnan(images).sum()>0:
                print("nan occur in train image, continue",torch.isnan(images).sum())
                continue
                    
            class_labels = class_labels.cuda()

            self.matmul2_output.clear()
            self.matmul2_output_quant.clear()

            with torch.no_grad():
                output_teacher_batch = self.model_teacher(images)

            output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), class_labels)

            head_dist_loss = torch.zeros(1).cuda()

            if self.settings.head_dist_coef > 0.0:
                mask_prop = 0.1
                w_high = 2.0
                w_low  = 1.0

                if 'swin' in self.settings.network:
                    head_dist_loss = ssim_dist_loss_swin_masked(
                        self.matmul2_output, self.matmul2_output_quant,
                        mask_proportion=mask_prop, weight_high=w_high, weight_low=w_low
                    )
                else:
                    head_dist_loss = ssim_dist_loss_masked(
                        self.matmul2_output, self.matmul2_output_quant,
                        mask_proportion=mask_prop, weight_high=w_high, weight_low=w_low,
                        distance=self.settings.head_dist_distance
                    )
            

            loss_S_total = loss_S + self.settings.head_dist_coef*head_dist_loss

            if self.settings.real_data or self.settings.aq_mode == "lsq":
                LESS_UNFREEZE=3
            else:
                LESS_UNFREEZE=0
                
            loss_S_total = loss_S_total / self.settings.grad_acc
            loss_S_total.backward()

            if epoch>= self.settings.warmup_epochs-LESS_UNFREEZE:
                if ((i + 1) % self.settings.grad_acc == 0) or (i + 1 == len(self.train_loader)):
                    self.optimizer_S.step()
                    self.optimizer_S.zero_grad()

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=class_labels,
                loss=loss_S_total, top5_flag=True, mean_flag=True)
            
            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))
            
            gt = class_labels.data.cpu().numpy()
            d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

            fp_acc.update(d_acc)

            del images, output_teacher_batch, output
            
        print(
            "[Epoch %d/%d] [Batch %d/%d] [T acc: %.4f%%] [Q acc: %.4f%%] [S loss: %f] [HD loss: %f]"
            % (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, 100-top1_error.avg,
            loss_S.item(), head_dist_loss.item())
        )
        
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()

        self.scalar_info['accuracy every epoch'] = 100 * d_acc
        self.scalar_info['S loss every epoch'] = loss_S

        self.scalar_info['training_top1error'] = top1_error.avg
        self.scalar_info['training_top5error'] = top5_error.avg
        self.scalar_info['training_loss'] = top1_loss.avg
        
        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        return top1_error.avg, loss_S_total,top5_error.avg 
    
    
    def train_random(self, epoch):
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = (self.settings.num_samples // self.settings.batchSize)+1 
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        
        self.optimizer_S.zero_grad()
        
        if self.settings.head_dist_coef > 0.0:
            for m in self.model_teacher.modules():
                if isinstance(m, Attention) or isinstance(m, WindowAttention):
                    self.teacher_hooks.append(m.matmul2.register_forward_hook(self.hook_fn_forward))
        
        if epoch==0 and self.settings.head_dist_coef > 0.0:
            for m in self.model.modules():
                if isinstance(m, Attention) or isinstance(m, WindowAttention):
                    m.matmul2.register_forward_hook(self.hook_fn_forward_quant)

        
        for i in range(iters): 
                
            images = torch.randn((self.settings.batchSize,self.settings.channels,self.settings.img_size,self.settings.img_size),device='cuda')

                    
            images = self.data_transforms(images)
                    
            if torch.isnan(images).sum()>0:
                print("nan occur in train image, continue",torch.isnan(images).sum())
                continue
                    

            class_labels = torch.randint(0,high=self.settings.nClasses, size=(self.settings.batchSize,),device='cuda')

            self.matmul2_output.clear()
            self.matmul2_output_quant.clear()
            with torch.no_grad():
                output_teacher_batch = self.model_teacher(images)

            output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), class_labels)

            head_dist_loss = torch.zeros(1).cuda()
            if self.settings.head_dist_coef > 0.0:
                if 'swin' in self.settings.network:
                    head_dist_loss = ssim_dist_loss_swin(self.matmul2_output, self.matmul2_output_quant)
                else:
                    head_dist_loss = ssim_dist_loss(self.matmul2_output, self.matmul2_output_quant)
                

            loss_S_total = loss_S + self.settings.head_dist_coef*head_dist_loss 

            if self.settings.real_data or self.settings.aq_mode == "lsq":
                LESS_UNFREEZE=3
            else:
                LESS_UNFREEZE=0
                
            loss_S_total = loss_S_total / self.settings.grad_acc
            loss_S_total.backward()

            if epoch>= self.settings.warmup_epochs-LESS_UNFREEZE:
                if ((i + 1) % self.settings.grad_acc == 0) or (i + 1 == len(self.train_loader)):
                    self.optimizer_S.step()
                    self.optimizer_S.zero_grad()

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=class_labels,
                loss=loss_S_total, top5_flag=True, mean_flag=True)
            
            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))
            
            gt = class_labels.data.cpu().numpy()
            d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

            fp_acc.update(d_acc)
            
        print(
            "[Epoch %d/%d] [Batch %d/%d] [T acc: %.4f%%] [Q acc: %.4f%%] [S loss: %f] [HD loss: %f]"
            % (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, 100-top1_error.avg,
            loss_S.item(), head_dist_loss.item())
        )
        
        for hook in self.teacher_hooks:
            hook.remove()

        self.scalar_info['accuracy every epoch'] = 100 * d_acc
        self.scalar_info['S loss every epoch'] = loss_S

        self.scalar_info['training_top1error'] = top1_error.avg
        self.scalar_info['training_top5error'] = top5_error.avg
        self.scalar_info['training_loss'] = top1_loss.avg
        
        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        return top1_error.avg, loss_S_total,top5_error.avg 

    def test(self, epoch):
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        
        self.model.eval()
        self.model_teacher.eval()
        
        iters = len(self.test_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                labels = labels.cuda()
                images = images.cuda()
                output = self.model(images)

                loss = torch.ones(1)
                self.matmul2_output_quant.clear()

                single_error, single_loss, single5_error = utils.compute_singlecrop(
                    outputs=output, loss=loss,
                    labels=labels, top5_flag=True, mean_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
        )
        
        self.scalar_info['testing_top1error'] = top1_error.avg
        self.scalar_info['testing_top5error'] = top5_error.avg
        self.scalar_info['testing_loss'] = top1_loss.avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test_teacher(self, epoch):
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model_teacher.eval()

        iters = len(self.test_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                labels = labels.cuda()
                if self.settings.tenCrop:
                    image_size = images.size()
                    images = images.view(
                        image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.cuda()
                        temp_output = self.model_teacher(img)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = utils.compute_tencrop(
                        outputs=output, labels=labels)
                else:
                    if self.settings.nGPU == 1:
                        images = images.cuda()

                    output = self.model_teacher(images)

                    loss = torch.ones(1)
                    self.matmul2_output.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                        outputs=output, loss=loss,
                        labels=labels, top5_flag=True, mean_flag=True)
                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

        print(
                "Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
                % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
        )

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg
