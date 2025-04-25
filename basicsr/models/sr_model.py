import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr_dat.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import sys
import os

# Get the absolute path to the directory two levels above
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add this directory to Python's search path
sys.path.append(parent_dir)

# Now you can import your module directly
from models_KAT.Umass import Umass, DifferentiableRGBtoVel, Dissipation
import matplotlib.pyplot as plt
import numpy as np

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.opt_train=opt['train']


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
        
        # Initialize the Umass loss function if needed
        if opt['train']['use_umass_loss']:
            self.RGBtoVel= DifferentiableRGBtoVel(vmin=0,vmax=0.56)
            self.umass_loss_fn = Umass().to(self.device)
            self.umass_loss_weight = opt['train']['umass_loss_weight']

        # Initialize the Umass loss function if needed
        if opt['train']['use_momentum_loss']:
            #self.momentum_loss_fn = MomentumLoss().to(self.device)
            self.momentum_loss_weight = opt['train']['momentum_loss_weight']

        # Initialize the Dissipation loss function if needed
        if  opt['train']['use_dissipation_loss']:
            self.dissipation_loss_fn = Dissipation().to(self.device)
            self.dissipation_loss_weight = opt['train']['dissipation_loss_weight']
        


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_csv = data['csv'].to(self.device)  # Ground truth velocity
            #self.gt_geometry = data['geometry'].to(self.device)  # Ground truth geometry
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # Enforce zero values in the output where obstacles are present
        if hasattr(self, 'gt_geometry'):
            #print(self.gt_geometry.shape)
            # Ensure gt_geometry is binary (obstacle: 1, non-obstacle: 0)
            mask = (self.gt_geometry ==1).float().to(self.device)
            mask = torch.rot90(mask, k=3, dims=(2, 3))

            self.output=  self.output.to(self.device)
            self.output = self.output * (1 - mask)  # Apply the mask to the output


        l_total = 0
        loss_dict = OrderedDict()
        
        # Pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        # Perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        # Assume that the following boolean flags are defined elsewhere in your class:
        # self.use_rgb_to_vel, self.use_umass_loss, self.use_dissipation_loss

        # Umass Loss
        if self.opt_train['use_umass_loss']:
            umass_loss = self.umass_loss_weight * self.umass_loss_fn(self.RGBtoVel(self.output), self.gt_csv)
            l_total += umass_loss
            loss_dict['umass_loss'] = umass_loss


        l_total.backward()
        self.optimizer_g.step()

        # Record Total Loss
        loss_dict['total_loss'] = l_total
        
        # Update log
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        # Initialize metric results
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # Initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        
        # Initialize mass_loss and dissipation_loss
        self.metric_results['mass_loss'] = 0
        #self.metric_results['dissipation_loss'] = 0
        #self.metric_results['momentum_loss'] = 0

        # Zero self.metric_results for metrics
        if with_metrics:
            for metric in self.opt['val']['metrics'].keys():
                self.metric_results[metric] = 0

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            # Get model output and ground truth
            visuals = self.get_current_visuals()
            sr_img = visuals['result']  # Model output tensor
           
            if 'gt' in visuals:
                gt_img = visuals['gt']  # Ground truth tensor
#                gt_velocity = self.gt_velocity  # Ground truth velocity

                # Compute velocity from sr_img
               # E_velocity = self.rgb_to_vel(sr_img * 256)  # Convert to velocity

                # Compute mass loss and dissipation loss
                # Umass Loss
                if self.opt_train['use_umass_loss']:
                    umass_loss = self.umass_loss_weight * self.umass_loss_fn(self.RGBtoVel(self.output), self.gt_csv)
                    self.metric_results['mass_loss'] += umass_loss.item()

                # Compute mass loss and dissipation loss
                if self.opt_train.get('use_momentum_loss', False):
                    momentum_loss = self.momentum_loss_weight * self.momentum_loss_fn(sr_img.to(self.device), gt_img.to(self.device))
                    self.metric_results['momentum_loss'] += momentum_loss.item()

                if self.opt_train.get('use_dissipation_loss', False):
                    dissipation_loss = self.dissipation_loss_weight * self.dissipation_loss_fn(E_velocity, gt_velocity)
                    self.metric_results['dissipation_loss'] += dissipation_loss.item()
            # Enforce zero values in the output where obstacles are present
        # Enforce zero values in the output where obstacles are present
            if hasattr(self, 'gt_geometry'):
                mask = (self.gt_geometry >0).float().to(self.device)
                # Flip the mask horizontally (mirror effect)
                mask = torch.flip(mask, dims=[3])
                # Rotate the mask 90 degrees to the right
                mask = torch.rot90(mask, k=3, dims=(2, 3))
                sr_img = sr_img.to(self.device)
                sr_img = sr_img * (1 - mask)

            # Convert tensors to images for metric calculation and saving
            sr_img_np = tensor2img([sr_img])
            metric_data['img'] = sr_img_np
            if 'gt' in visuals:
                gt_img_np = tensor2img([gt_img])
                metric_data['img2'] = gt_img_np
                del self.gt  # Remove gt to free memory

            # Tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png'
                    )
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.png',
                        )
                imwrite(sr_img_np, save_img_path)

            if with_metrics:
                # Calculate other metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        if use_pbar:
            pbar.close()

        # Average the metrics
        for metric in self.metric_results.keys():
            self.metric_results[metric] /= (idx + 1)
            if with_metrics and metric not in ['mass_loss', 'dissipation_loss','momentum_loss']:
                # Update the best metric result for other metrics
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

        # Log validation metric values
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)



    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation [{dataset_name}] \tIter: {current_iter}\t'
        for metric, value in self.metric_results.items():
            log_str += f'{metric}: {value:.4f}\t'
            # If using TensorBoard logger
            if tb_logger:
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
        logger = get_root_logger()
        logger.info(log_str)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)