function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

lod=0
iterations=8_000
iterations_static=8_000
update_until=5_000
feat_dim=32
densify_grad_threshold=0.0002
success_threshold=0.8
# offset_lr_init=0.01
# offset_lr_final=0.0001
# mlp_color_lr_init=0.008 
# mlp_color_lr_final=0.00005
offset_lr_init=0.001 #edit
offset_lr_final=0.00001 #edit
mlp_color_lr_init=0.04
mlp_color_lr_final=0.00025
pose_lr_init=0.0001
pose_lr_final=0.00001
update_from=1000

position_lr_max_steps=${iterations_static}
offset_lr_max_steps=${iterations_static}
mlp_opacity_lr_max_steps=${iterations_static}
mlp_cov_lr_max_steps=${iterations_static}
mlp_color_lr_max_steps=${iterations_static}
mlp_featurebank_lr_max_steps=${iterations_static}
appearance_lr_max_steps=${iterations_static}
pose_lr_max_steps=${iterations_static}




while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --use_residual) use_residual="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_residual_dim) appearance_residual_dim="$2"; shift ;;
        --prune_ratio) prune_ratio="$2"; shift ;;
        --kernel_size) kernel_size="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")
mkdir -p outputs/${logdir}/$time
if [ "$warmup" = "True" ]; then
    echo "warmup"
    if [ "$use_residual" = "True" ]; then
        echo "use_residual"
        CUDA_VISIBLE_DEVICES=${gpu} python train.py --eval -s ${data} --lod ${lod} \
        --iterations ${iterations} \
        --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor}\
        --appearance_residual_dim ${appearance_residual_dim}  \
        --kernel_size ${kernel_size} --port $port -m outputs/${logdir}/$time --use_wandb --warmup  \
        --update_until ${update_until} --feat_dim ${feat_dim} \
        --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}\
        --position_lr_max_steps ${position_lr_max_steps} --offset_lr_max_steps ${offset_lr_max_steps}\
        --mlp_opacity_lr_max_steps ${mlp_opacity_lr_max_steps} --mlp_cov_lr_max_steps ${mlp_cov_lr_max_steps}\
        --mlp_color_lr_max_steps ${mlp_color_lr_max_steps} --mlp_featurebank_lr_max_steps ${mlp_featurebank_lr_max_steps}\
        --mlp_color_lr_init ${mlp_color_lr_init} --mlp_color_lr_final ${mlp_color_lr_final}\
        --pose_lr_max_steps ${pose_lr_max_steps} \
        --offset_lr_init ${offset_lr_init} --offset_lr_final ${offset_lr_final} --pose_lr_init ${pose_lr_init} --pose_lr_final ${pose_lr_final}\
        --update_from ${update_from}\
        --appearance_lr_max_steps ${appearance_lr_max_steps}  --use_residual   --use_3D_filter --prune_ratio ${prune_ratio} \
        # >outputs/${logdir}/$time/${logdir}.log 2>&1 
        wait
    else
        CUDA_VISIBLE_DEVICES=${gpu} python train.py --eval -s ${data} --lod ${lod} \
        --iterations ${iterations} \
        --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor}\
        --kernel_size ${kernel_size} --port $port -m outputs/${logdir}/$time --use_wandb  \
        --update_until ${update_until} --feat_dim ${feat_dim}  \
        --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}\
        --position_lr_max_steps ${position_lr_max_steps} --offset_lr_max_steps ${offset_lr_max_steps}\
        --mlp_opacity_lr_max_steps ${mlp_opacity_lr_max_steps} --mlp_cov_lr_max_steps ${mlp_cov_lr_max_steps}\
        --mlp_color_lr_max_steps ${mlp_color_lr_max_steps} --mlp_featurebank_lr_max_steps ${mlp_featurebank_lr_max_steps}\
        --mlp_color_lr_init ${mlp_color_lr_init} --mlp_color_lr_final ${mlp_color_lr_final}\
        --pose_lr_max_steps ${pose_lr_max_steps} \
        --offset_lr_init ${offset_lr_init} --offset_lr_final ${offset_lr_final} --pose_lr_init ${pose_lr_init} --pose_lr_final ${pose_lr_final}\
        --update_from ${update_from}\
        --appearance_lr_max_steps ${appearance_lr_max_steps}  --use_3D_filter --prune_ratio ${prune_ratio}\
        # >outputs/${logdir}/$time/${logdir}.log 2>&1
    fi
else
    
    if [ "$use_residual" = "True" ]; then
        echo "use_residual"
        CUDA_VISIBLE_DEVICES=${gpu} python train.py --eval -s ${data} --lod ${lod} \
        --iterations ${iterations} \
        --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor}\
        --appearance_residual_dim ${appearance_residual_dim} \
        --kernel_size ${kernel_size} --port $port -m outputs/${logdir}/$time --use_wandb   \
        --update_until ${update_until} --feat_dim ${feat_dim} \
        --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}\
        --position_lr_max_steps ${position_lr_max_steps} --offset_lr_max_steps ${offset_lr_max_steps}\
        --mlp_opacity_lr_max_steps ${mlp_opacity_lr_max_steps} --mlp_cov_lr_max_steps ${mlp_cov_lr_max_steps}\
        --mlp_color_lr_max_steps ${mlp_color_lr_max_steps} --mlp_featurebank_lr_max_steps ${mlp_featurebank_lr_max_steps}\
        --mlp_color_lr_init ${mlp_color_lr_init} --mlp_color_lr_final ${mlp_color_lr_final}\
        --pose_lr_max_steps ${pose_lr_max_steps} \
        --offset_lr_init ${offset_lr_init} --offset_lr_final ${offset_lr_final} --pose_lr_init ${pose_lr_init} --pose_lr_final ${pose_lr_final}\
        --update_from ${update_from}\
        --appearance_lr_max_steps ${appearance_lr_max_steps}  --use_residual --use_3D_filter --prune_ratio ${prune_ratio}\
        # >outputs/${logdir}/$time/${logdir}.log 2>&1
    else
        CUDA_VISIBLE_DEVICES=${gpu} python train.py --eval -s ${data} --lod ${lod} \
        --iterations ${iterations} \
        --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor}\
        --appearance_residual_dim ${appearance_residual_dim} \
        --kernel_size ${kernel_size} --port $port -m outputs/${logdir}/$time --use_wandb  \
        --update_until ${update_until} --feat_dim ${feat_dim} \
        --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}\
        --position_lr_max_steps ${position_lr_max_steps} --offset_lr_max_steps ${offset_lr_max_steps}\
        --mlp_opacity_lr_max_steps ${mlp_opacity_lr_max_steps} --mlp_cov_lr_max_steps ${mlp_cov_lr_max_steps}\
        --mlp_color_lr_max_steps ${mlp_color_lr_max_steps} --mlp_featurebank_lr_max_steps ${mlp_featurebank_lr_max_steps}\
        --mlp_color_lr_init ${mlp_color_lr_init} --mlp_color_lr_final ${mlp_color_lr_final}\
        --pose_lr_max_steps ${pose_lr_max_steps} \
        --offset_lr_init ${offset_lr_init} --offset_lr_final ${offset_lr_final} --pose_lr_init ${pose_lr_init} --pose_lr_final ${pose_lr_final}\
        --update_from ${update_from} \
        --appearance_lr_max_steps ${appearance_lr_max_steps}  --use_3D_filter --prune_ratio ${prune_ratio} \
        # >outputs/${logdir}/$time/${logdir}.log 2>&1
    fi
fi


