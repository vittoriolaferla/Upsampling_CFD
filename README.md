## Requirements
```
git clone
conda create -n Upsampling python= 3.8
conda activate Upsampling
pip install -r requirements.txt
python setup.py develop
```

### Training
To train SwinIR, run the following commands. You may need to change the `dataroot_H`, `dataroot_L`, `scale factor`, `noisel level`, `JPEG level`, `G_optimizer_lr`, `G_scheduler_milestones`, etc. in the json file for different settings. 

```python
# SwinIR model
##X2 magnification factor
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 main_train_SwinIR.py --opt options/options_SwinIR/train_swinir_sr_classicalx2.json  --dist True 
##X4 magnification factor
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 main_train_SwinIR.py --opt options/options_SwinIR/train_swinir_sr_classicalx4.json  --dist True 
# DAT model
##X2 magnification factor
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/options_DAT/Train/train_DAT_2_x2.yml --launcher pytorch
##X4 magnification factor
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/options_DAT/Train/train_DAT_2_x4.yml --launcher pytorch

#ResShift model
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 --nnodes=1 main.py --cfg_path configs/realsr_swinunet_64.yaml --save_dir  output_dir
```

### Testing
To thest the models use this commands:
```python
# SwinIR model
python main_test_SwinIR.py --task classical_sr --scale 2 --training_patch_size 32 --model_path superresolution
/swinir_sr_x2_Obstacles/models/80000_G.pth --folder_lq datasets/dataset_obstacles/dataset_csv_Y_all_cases/test/LW --folder_gt datasets/dataset_obstacles/dataset_csv_Y_al
l_cases/test/HR
# DAT model
##X2 magnification factor
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/options_DAT/Test/test_DAT_2_x2.yml --launcher pytorch
#ResShift model
python inference_customModel.py    --in_path  data/dataset_vectors/validation/lw     --out_path infereceResut_physics/no_physics_Vector_x2_time    --ckpt_path /home/vittorio/Scrivania/ResShift_4_scale/models_trained_no_physcs/no_physics_Vector_x2/ckpts/model_75000.pth     --config_path configs/realsr_swinunet_realsrgan48.yaml     --scale 2    --chop_size 64     --chop_stride 64     --bs 1     --task realsr```

