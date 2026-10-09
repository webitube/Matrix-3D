import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
import argparse
import cv2
        
def main(args):
    device = args.device
    step1_output_dir = os.path.abspath(args.inout_dir)

    generated_dir = os.path.join(step1_output_dir, "generated")
    condition_dir = os.path.join(step1_output_dir, "condition")

    if not args.train_only:
        prompt_path = os.path.join(args.inout_dir, 'prompt.txt')

        with open(os.path.abspath(prompt_path),"r",encoding="utf-8") as f:
            prompt=f.read()
            print(f"prompt is {prompt}")

        generated_video_path = os.path.join(generated_dir,"generated.mp4")
        if args.resolution == 720:
            width_following = 1440
            height_following = 720
        else:
            os.system(f"cd code/VideoSR && python scripts/enhance_video_pipeline.py --version v2 --up_scale 2 --target_fps 20 --noise_aug 100 --solver_mode 'fast' --steps 15 --input_path {generated_video_path} --prompt \'{prompt}\' --save_dir {generated_dir} --suffix enhancement")
            generated_video_path = os.path.join(generated_dir,"generated_resize_enhance.mp4")
            width_following = 1920
            height_following = 960

        camera_path = os.path.join(condition_dir,"cameras.npz")
        os.system(f"python code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py \
            --device {device} \
            --camera_path {camera_path} \
            --video_path {generated_video_path} \
            --anchor_frame_depth_paths \'{os.path.join(condition_dir,'firstframe_depth.exr')}\' \
            --anchor_frame_mask_paths \'{os.path.join(condition_dir,'firstframe_mask.png')}\' \
            --anchor_frame_indices 0 \
            --output_dir {os.path.join(step1_output_dir,'geom_optim')} \
            --depth_estimation_interval 10 \
            --width {width_following} \
            --height {height_following} \
        ")
        # cut everything into perspective images;


        os.system(
            f"python code/utils_3dscene/gs_optim_datagen.py \
                --optimized_depth_dir {os.path.join(step1_output_dir,'geom_optim/data/optimized_depths')} \
                --camera_path {os.path.join(step1_output_dir,'condition/cameras.npz')} \
                --output_dir {os.path.join(step1_output_dir,'geom_optim/data')} \
            "
        )
        cmd_rename = f"mv {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')} {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')}"
        os.system(cmd_rename)
        #cmd = f"cd StableSR && python scripts/sr_val_ddpm_text_T_vqganfin_old.py --init-img {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')} --outdir {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')}"
        cmd = f"cd code/StableSR && python scripts/sr_val_ddpm_text_T_vqganfin_old.py --init-img {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')} --outdir {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')} --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt {os.path.abspath('./checkpoints/StableSR/stablesr_turbo.ckpt')} --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples 1 --vqgan_ckpt {os.path.abspath('./checkpoints/StableSR/vqgan_cfw_00011.ckpt')} --colorfix_type wavelet"
        os.system(cmd)
    else:
        print("[train_only] skipping VideoSR/depth/datagen/StableSR; training directly on existing geom_optim/data")

    # apply gs optimization;
    #for i in range(N):
    gs_input_dir = os.path.join(step1_output_dir,'geom_optim/data')
    gs_output_dir = os.path.join(step1_output_dir,'geom_optim/output')
    os.system(f"cd ./code/Pano_GS_Opt && python train.py -s {gs_input_dir} -m {gs_output_dir} -r 1 --use_decoupled_appearance --save_iterations {args.iterations} --test_iterations {args.iterations} \
    --sh_degree 0 --densify_from_iter {args.densify_from_iter} --densify_until_iter {args.densify_until_iter} --iterations {args.iterations} --eval \
    --img_sample_interval 1 --num_views_per_view 3 --num_of_point_cloud {args.num_of_point_cloud} --device {device} --distortion_from_iter 6500 --depth_normal_from_iter 6500 \
    --densify_grad_threshold {args.densify_grad_threshold} --log_memory_interval {args.log_memory_interval}\
    ")

    # gather results;
    all_output_dir = step1_output_dir
    gs_path = os.path.join(gs_output_dir,f"point_cloud/iteration_{args.iterations}/point_cloud.ply")
    os.system(f"cp {gs_path} {os.path.join(all_output_dir, 'generated_3dgs_opt.ply')}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="the device on which the 3d scene generation runs")
    parser.add_argument("--inout_dir", type=str, default="./output/example1", help="the directory storing the input and output result")
    parser.add_argument("--resolution", type=int, default=720, help="the working resolution of the 3D scene generation")
    parser.add_argument("--train_only", "--train-only", action="store_true",
                        help="skip VideoSR/depth/datagen/StableSR and run 3DGS training directly on an existing geom_optim/data dir (for fast densification experiments)")
    parser.add_argument("--iterations", type=int, default=3000, help="3DGS training iterations")
    parser.add_argument("--densify_from_iter", type=int, default=500, help="start densification at this iteration")
    parser.add_argument("--densify_until_iter", type=int, default=1501, help="stop densification at this iteration (lower = fewer Gaussians = less VRAM)")
    parser.add_argument("--num_of_point_cloud", type=int, default=3000000, help="initial number of Gaussians")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002, help="gradient threshold for densification (higher = fewer new points)")
    parser.add_argument("--log_memory_interval", type=int, default=100, help="log VRAM + system RAM + point count every N training iterations (0 disables)")
    # parser.add_argument("--step1_output_dir", type=str, default="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output_step1/A_cherry_blossom_forest_with_petals_falling_gently,_a_wooden_bridge_over_a_stream,_and_a_shrine_in_the_background,_anime_style,_ultra-detailed,_soft_pastel_colors,_serene_ambiance_superres")
    args = parser.parse_args()
    

    main(args)