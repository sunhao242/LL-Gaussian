
voxel_size=0.001
update_init_factor=16
appearance_residual_dim=32
prune_ratio=0.1
gpu=1
kernel_size=0.1
warmup=False
use_residual=True
scene_names=( "chair" "stone" "path" "staircase"  "firehydrant" " pole")
scene='./dataset/LLRS-sRGB'    
exp_name='test'

for scene_name in ${scene_names[@]}; do
    scripts/train.sh -d ${scene}/${scene_name} -l ${scene_name}${exp_name} --gpu ${gpu} --warmup ${warmup} --use_residual ${use_residual} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_residual_dim ${appearance_residual_dim} --prune_ratio ${prune_ratio} --kernel_size ${kernel_size} 
done


