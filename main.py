# Standard
import os, time, argparse, copy

# Third-party
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
from adapt import get_method
from utils import segmentation_datasets
from utils.metrics import intersect_and_union, process_metrics, total_area_to_metrics
from utils.misc import set_global_seeds, save_configuration, aggregate_pred_patches


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise argparse.ArgumentTypeError("Expected 'True' or 'False'")


"""TODO List:
- end of main_segmentation is necessary?


- datasets
    we can have download datasets script? not sure
    but we can have a dataset.md
- repo:
    - add a section (supported methods=> list them and add reference to each of them)
    - we can talk about how to perform all methods (including No Adapt)
    - in acknowledgements, we can say athat we modified the original CLIP code to "ovss/clip/model.py" to be able to perform segmentation 
"""



def argparser():
    parser = argparse.ArgumentParser(
        description="Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation"
    )
    
    # ----------------------------------------
    # I/O Directories
    # ----------------------------------------
    parser.add_argument(
        '--save_dir',
        type=str,
        default='save/',
        help='Directory to save model weights and results'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.data/',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='',
        help='Path to the YAML file containing prompt templates'
    )
    
    # ----------------------------------------
    # Dataset Settings
    # ----------------------------------------
    parser.add_argument(
        '--dataset',
        type=str,
        default='COCOStuffDataset',
        choices=(
            'COCOStuffDataset', 'COCOObjectDataset', 'CityscapesDataset',
            'PascalVOC20Dataset', 'PascalVOC21Dataset',
            'PascalContext59Dataset', 'PascalContext60Dataset', 'SUIM6Dataset', 'SUIM5Dataset',
            'DUTUSEG5Dataset', 'DUTUSEG4Dataset'
        ),
        help='Which dataset to load'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of data-loading workers'
    )
    parser.add_argument(
        '--init_resize',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Resize images before patch extraction. '
            'Order doesn’t matter (e.g., (560,448) same as (448,560)). '
            'If None, use original size (batch_size must be 1).'
        )
    )
    parser.add_argument(
        '--patch_size',
        nargs='+',
        type=int,
        default=None,
        help='Size of each image patch after resize (model input size)'
    )
    parser.add_argument(
        '--patch_stride',
        type=int,
        default=None,
        help='Stride for extracting patches'
    )
    parser.add_argument(
        '--corruptions_list',
        nargs='+',
        type=str,
        default=None,
        help='List of corruptions to apply for robustness (e.g., gaussian, motion_blur)'
    )
    
    # ----------------------------------------
    # Model Settings
    # ----------------------------------------
    parser.add_argument(
        '--ovss_type',
        type=str,
        default='ncalip',
        help='Open-Vocabulary Semantic Segmentation type (e.g., nacalip, clip, clip, etc.)'
    )
    parser.add_argument(
        '--ovss_backbone',
        type=str,
        default='ViT-B/32',
        help='CLIP vision backbone (e.g., ViT-B/32, ViT-L/14)'
    )
    parser.add_argument(
        '--class_extensions',
        action='store_true',
        help='Enable dataset-specific class extensions if available'
    )
    
    # ----------------------------------------
    # Adaptation / Training Settings
    # ----------------------------------------
    parser.add_argument(
        '--adapt',
        action='store_true',
        help='Enable test-time adaptation'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tent',
        help='Adaptation method name (e.g., mlmp watt, tent)'
    )
    parser.add_argument(
        '--reset_mode',
        type=str,
        default='episodic',
        choices=('episodic', 'normal', 'continual'),
        help='Reset behavior for TTA'
    )
    parser.add_argument(
        '--lifelong',
        type=str,
        default='None',
        choices=('None', 'shuffle_domain_pround', 'shuffle_domain_pbatch'),
        help='Lifelong domain scheduling mode'
    )
    parser.add_argument(
        '--lifelong_rnds',
        type=int,
        default=3,
        help='Number of lifelong rounds'
    )
    parser.add_argument(
        '--domain_gen',
        type=str2bool,
        default=False,
        help='If True, adapt on all but the last domain_gen_num domains and directly evaluate the last domains with adapted weights'
    )
    parser.add_argument(
        '--domain_gen_num',
        type=int,
        default=5,
        help='Number of last domains to hold out from adaptation for domain generalization evaluation'
    )
    parser.add_argument(
        '--batch_size', '--batch-size',
        type=int,
        default=1,
        dest='batch_size',
        help='Batch size for adaptation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for adaptation optimizer'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=('adam', 'adamw', 'sgd'),
        help='Optimizer for adaptation'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1,
        help='Number of adaptation iterations per batch'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=1,
        help='Number of experimental repetitions'
    )
    
    # ----------------------------------------
    # Debug / Misc
    # ----------------------------------------
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--plot_loss',
        action='store_true',
        help='Plot the loss curve (averaged over batches and seeds)'
    )
    parser.add_argument(
        '--runtime_calculation',
        action='store_true',
        help='Calculate the runtime of adaptation and evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'mlmp':
        parser.add_argument(
            '--vision_outputs',
            nargs='+',
            type=int,
            default=(-1,),
            help='Indices of vision layers to extract outputs from'
        )
        parser.add_argument(
            '--prompt_integration',
             type=str, default='loss', 
             help='If we have different prompt templates, how to integrate them (loss-level or text-level). MLMP uses loss-level integration by default.'
             )
        parser.add_argument(
            '--alpha_cls', 
            type=float, 
            default=1.0, 
            help='Weight for the classification loss in MLMP'
            )
    
    elif method == 'watt':
        parser.add_argument(
            '--watt_l', 
            default=2, 
            type=int, 
            help='Number of adaptation iterations for each text embedding before weight averaging'
            )
        parser.add_argument('--watt_m', 
            default=5, 
            type=int, 
            help='Number of repetitions of the adaptation and weight averaging process'
            )

    elif method == 'clipartt':
        parser.add_argument(
            '--clipartt_k', 
            default=3, 
            type=int, 
            help='Number of classes taken to build the area pseudo label'
            )

    elif method == 'tpt':
        parser.add_argument(
                '--n_ctx', 
                default=4, 
                type=int,
            )

    elif method == 'method':
        parser.add_argument(
            '--train_imag_norm',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Train LayerNorm layers in the visual encoder (True/False)'
        )
        parser.add_argument(
            '--last_imag_k_norm',
            type=int,
            default=0,
            help='Train only the last K visual transformer blocks LN layers. 0 = all LN layers.'
        )
        parser.add_argument(
            '--train_imag_attn',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Train attention layers in the visual encoder (True/False)'
        )
        parser.add_argument(
            '--last_imag_k_attn',
            type=int,
            default=0,
            help='Train only the last K visual transformer blocks attn layers. 0 = all attn layers.'
        )
        parser.add_argument(
            '--train_text_norm',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Train LayerNorm layers in the text encoder (True/False)'
        )
        parser.add_argument(
            '--last_text_k_norm',
            type=int,
            default=0,
            help='Train only the last K text transformer blocks LN layers. 0 = all LN layers.'
        )
        parser.add_argument(
            '--loss_ent',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Enable entropy minimization loss (True/False)'
        )
        parser.add_argument(
            '--lamb_ent',
            type=float,
            default=1.0,
            help='Lambda multiplier for entropy minimization loss'
        )
        parser.add_argument(
            '--loss_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable class-wise diversity loss to prevent model collapse (True/False)'
        )
        parser.add_argument(
            '--lamb_div',
            type=float,
            default=1.0,
            help='Lambda multiplier for diversity loss'
        )
        parser.add_argument(
            '--loss_aug_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable pixel-wise augmentation consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_aug_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for augmentation consistency loss'
        )
        parser.add_argument(
            '--loss_src_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable source model consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_src_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for source model consistency loss'
        )
        parser.add_argument(
            '--updownsample',
            type=float,
            default=1.0,
            help='Control prediction upsampling vs GT downsampling ratio (0.0=native token res, 1.0=full patch res, 0.5=halfway)'
        )
        parser.add_argument(
            '--prompt_average',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable averaging of prompt encodings during adaptation and evaluation (True/False). Only effective when --prompt_dir is set.'
        )
        parser.add_argument(
            '--cons_type',
            type=str,
            default='sym_kl',
            choices=['sym_kl', 'for_kl', 'rev_kl'],
            help='Consistency loss type: symmetric KL (sym_kl), forward KL (for_kl), or reverse KL (rev_kl). Applied to both aug_cons and src_cons losses.'
        )

    
    return parser


def main(args):

    # Save the configuration settings
    save_configuration(args)

    # Start the timer
    start_time = time.time()

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the save directory if it doesn't exist
    all_results_path = os.path.join(args.save_dir, "results.txt")
    os.makedirs(os.path.dirname(all_results_path), exist_ok=True)

    if args.domain_gen and args.lifelong != 'None':
        raise ValueError("--domain_gen only works when --lifelong is None")

    if args.domain_gen:
        run_domain_gen(args, device, start_time, all_results_path)
        return

    if args.lifelong != 'None':
        run_lifelong(args, device, start_time, all_results_path)
        return

    # create necessary variables
    all_results = dict()
    headers = "mIoU, mDice, mAcc"
    adapt_time_all_corr = []
    eval_time_all_corr = []
    continual_methods = None
    domain_summary = []
    
    for c_idx, corruption in enumerate(args.corruptions_list):

        data_loader, org_classes = segmentation_datasets.prepare_data(args.dataset, args.data_dir, args.init_resize,
                                                                  args.patch_size, args.patch_stride, corruption=corruption, 
                                                                  batch_size=args.batch_size, num_workers=args.workers)
        
        # Check if the extensions of classes should be used
        if args.class_extensions and data_loader.dataset.class_extensions is not None:
            ext_classes = data_loader.dataset.class_extensions
            args.classes = ext_classes
            print(f"\n+++ Using class extensions")
            print(f"+++ The number of classes [no extension]: {len(org_classes)}")
            print(f"+++ The number of classes after extension:  {len(ext_classes)}")

        else:
            args.classes = org_classes
            print(f"\n+++ The number of classes [no extension]: {len(org_classes)}")

        num_org_classes = len(org_classes)
        ignore_index = data_loader.dataset.ignore_index # the index of the ignore label in the segmentation map

        if args.reset_mode == 'episodic':
            adapt_method = get_method(args, device)
        elif args.reset_mode == 'continual' and continual_methods is None:
            continual_methods = [get_method(args, device) for _ in range(args.trials)]

        # Results path
        c_results_path = os.path.join(args.save_dir, f"{c_idx:02}_{corruption}", "results.txt")
        os.makedirs(os.path.dirname(c_results_path), exist_ok=True)

        miou_seeds = []
        dice_seeds = []
        acc_seeds = []
        loss_seed_report = []

        for t in range(args.trials):
            if args.reset_mode == 'normal':
                adapt_method = get_method(args, device)
            elif args.reset_mode == 'continual':
                adapt_method = continual_methods[t]

            results = []
            loss_batch_report = []
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):

                if args.debug and batch_idx == 10: 
                    break

                inputs = data['img_patches'] 
                labels = data['gt_patches']  
                original_gts = data['gt'] 

                patch_grid_shape = data['meta']['patch_grid_shape'] 
                image_shapes = data['meta']['img_shape']
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                if args.reset_mode == 'episodic':
                    adapt_method.reset()
                
                # perform adaptation
                if args.adapt:
                    loss_iter_report = adapt_method.adapt(inputs)
                    loss_batch_report.append(loss_iter_report)

                # perform evaluation 
                with torch.no_grad():
                    patch_preds = adapt_method.evaluate(inputs)

                # compute eval scale for updownsample support
                eval_size = getattr(adapt_method, 'eval_size', args.patch_size[0])
                eval_scale = eval_size / args.patch_size[0]

                # aggregate the predictions to construct the final segmentation map for each image in the batch
                if args.init_resize:
                    if eval_scale < 1.0:
                        scaled_patch_size = (round(args.patch_size[0] * eval_scale), round(args.patch_size[1] * eval_scale))
                        scaled_patch_stride = round(args.patch_stride * eval_scale)
                        scaled_img_shapes = [(round(h * eval_scale), round(w * eval_scale)) for h, w in image_shapes]
                        reconstructed_preds = aggregate_pred_patches(patch_preds, patch_grid_shape, scaled_img_shapes, scaled_patch_size, scaled_patch_stride)
                    else:
                        reconstructed_preds = aggregate_pred_patches(patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
                else:
                    reconstructed_preds = patch_preds

                
                # calculate the metrics for each image in the batch (since the images may have different sizes)
                for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):

                    # get the predictions
                    pd = pd.softmax(dim=0) # [num_org_classes, H, W]

                    # fix the extensions indices
                    if args.class_extensions and data_loader.dataset.class_extensions is not None:
                        ext_to_real_cls_indx = torch.Tensor(data_loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
                        num_cls, num_queries = max(ext_to_real_cls_indx) + 1, len(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = torch.nn.functional.one_hot(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = ext_to_real_cls_indx.T.view(num_cls, num_queries, 1, 1)
                        pd = pd.unsqueeze(0)
                        pd = (pd * ext_to_real_cls_indx).max(1)[0]


                    pd = pd.argmax(dim=0)  # [H, W]
                    pd = pd.to(gt.device)  

                    # get the ground truth
                    gt = gt[0]             # [H, W]
                    if eval_scale < 1.0:
                        target_h, target_w = scaled_img_shapes[idx]
                        gt = torch.nn.functional.interpolate(
                            gt.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest'
                        ).squeeze(0).squeeze(0).long()
                    # metric calculation
                    results.append(intersect_and_union(pd, gt, num_org_classes, ignore_index))
               
            
            # Convert the batch report to a numpy array for easier averaging
            loss_batch_report = np.array(loss_batch_report)

            # Average loss over batches for each iteration
            avg_loss_per_iter = np.mean(loss_batch_report, axis=0)  # Shape: [10] (for 10 iterations)
            loss_seed_report.append(avg_loss_per_iter)

            
            metrics = process_metrics(results, org_classes)
            miou_seeds.append(metrics['mIoU'])
            dice_seeds.append(metrics['mDice'])
            acc_seeds.append(metrics['mAcc'])
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")


            # Saving the weights if self.weights_track list is not empty
            if adapt_method.model.weights_track:
                weights_path = os.path.join(args.save_dir, "weights")
                
                weights = adapt_method.model.weights_track
                weights = np.hstack(weights)
                os.makedirs(weights_path, exist_ok=True)
                
                # save to a file
                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                # plot and save the mean and std of weights across the layers
                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

                # reset the weights_track list
                adapt_method.model.weights_track = []

        
        miou_mean, miou_std = np.array(miou_seeds).mean(), np.array(miou_seeds).std()
        dice_mean, dice_std = np.array(dice_seeds).mean(), np.array(dice_seeds).std()
        acc_mean, acc_std = np.array(acc_seeds).mean(), np.array(acc_seeds).std()

        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(c_results_path, 'w') as f:        
            f.write(headers + "\n")
            f.write(c_results_print)    

        loss_mean, loss_inc, loss_dec = compute_loss_stats(loss_seed_report)

        all_results[corruption] = c_results_print
        domain_summary.append({
            'corruption': corruption,
            'mIoU_mean': miou_mean,
            'mIoU_std': miou_std,
            'mDice_mean': dice_mean,
            'mDice_std': dice_std,
            'mAcc_mean': acc_mean,
            'mAcc_std': acc_std,
            'loss_mean': loss_mean,
            'loss_increase': loss_inc,
            'loss_decrease': loss_dec,
        })

        # Convert the seed report to a numpy array and average over trials (seeds)
        loss_seed_report = np.array(loss_seed_report)
        avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)  # Shape: [10] (averaged over seeds)

        if args.plot_loss and args.adapt:
            # Plot the averaged loss for this corruption
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds)+1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            
            # Save the plot in the specified directory
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

        # if the runtime calculation is enabled, we will have access to adapt_method.adapt_times and adapt_method.eval_times (each one contains a list of times)
        if args.runtime_calculation:
            if args.adapt:
                mean_adapt_time = np.mean(adapt_method.adapt_times[20:])
                std_adapt_time = np.std(adapt_method.adapt_times[20:])
            else:
                mean_adapt_time = 0
                std_adapt_time = 0
            
            mean_eval_time = np.mean(adapt_method.eval_times[20:])
            std_eval_time = np.std(adapt_method.eval_times[20:])

            mean_total_time = mean_adapt_time + mean_eval_time

            run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
            print(run_time_txt)
            
            runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
            with open(runtime_save_dir, 'a+') as f:
                f.write(run_time_txt + "\n")

            adapt_time_all_corr.append(mean_adapt_time)
            eval_time_all_corr.append(mean_eval_time)

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n===== Per-domain Summary =====")
    for domain_metrics in domain_summary:
        loss_str = ""
        if domain_metrics.get('loss_mean') is not None:
            loss_str = (
                f", Loss {domain_metrics['loss_mean']:.4f}"
                f", Loss+ {domain_metrics['loss_increase']:.4f}"
                f", Loss- {domain_metrics['loss_decrease']:.4f}"
            )
        print(
            f"{domain_metrics['corruption']}: "
            f"mIoU {domain_metrics['mIoU_mean']:.2f} +/- {domain_metrics['mIoU_std']:.2f}, "
            f"mDice {domain_metrics['mDice_mean']:.2f} +/- {domain_metrics['mDice_std']:.2f}, "
            f"mAcc {domain_metrics['mAcc_mean']:.2f} +/- {domain_metrics['mAcc_std']:.2f}"
            f"{loss_str}"
        )

    overall_miou_mean = np.mean([domain_metrics['mIoU_mean'] for domain_metrics in domain_summary])
    overall_mdice_mean = np.mean([domain_metrics['mDice_mean'] for domain_metrics in domain_summary])
    overall_macc_mean = np.mean([domain_metrics['mAcc_mean'] for domain_metrics in domain_summary])

    print("===== Overall Mean Summary =====")
    print(
        f"Overall mean across domains: "
        f"mIoU {overall_miou_mean:.2f}, "
        f"mDice {overall_mdice_mean:.2f}, "
        f"mAcc {overall_macc_mean:.2f}"
    )


    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")


def run_domain_gen(args, device, start_time, all_results_path):
    headers = "mIoU, mDice, mAcc"
    all_results = dict()
    domain_summary = []
    adapt_time_all_corr = []
    eval_time_all_corr = []

    holdout_count = min(args.domain_gen_num, len(args.corruptions_list))
    adapt_corruptions = set(args.corruptions_list[:-holdout_count]) if holdout_count > 0 else set(args.corruptions_list)
    eval_corruptions = list(args.corruptions_list[-holdout_count:]) if holdout_count > 0 else []

    domain_infos = []
    for c_idx, corruption in enumerate(args.corruptions_list):
        domain_infos.append(prepare_domain_info(args, device, corruption, c_idx))

    args.classes = domain_infos[0]['classes']

    continual_methods = None
    if args.reset_mode == 'continual':
        continual_methods = [get_method(args, device) for _ in range(args.trials)]

    for t in range(args.trials):
        if args.reset_mode == 'continual':
            adapt_method = continual_methods[t]
        else:
            adapt_method = get_method(args, device)

        for domain_idx, domain_info in enumerate(domain_infos):

            corruption = domain_info['corruption']
            should_adapt_domain = corruption in adapt_corruptions

            if args.reset_mode == 'normal' and should_adapt_domain:
                adapt_method.reset()

            results = []
            loss_batch_report = []
            weights_batch_report = []
            adapt_len_before = len(adapt_method.adapt_times) if args.runtime_calculation and args.adapt else 0
            eval_len_before = len(adapt_method.eval_times) if args.runtime_calculation else 0

            for batch_idx, data in tqdm(enumerate(domain_info['data_loader']), total=len(domain_info['data_loader'])):
                if args.debug and batch_idx == 10:
                    break

                if args.reset_mode == 'episodic' and should_adapt_domain:
                    adapt_method.reset()

                batch_results, loss_iter_report, _, _, weights = process_single_batch(
                    args,
                    device,
                    adapt_method,
                    data,
                    domain_info,
                ) if should_adapt_domain and args.adapt else process_single_batch_no_adapt(
                    args,
                    device,
                    adapt_method,
                    data,
                    domain_info,
                )

                results.extend(batch_results)
                if loss_iter_report is not None:
                    loss_batch_report.append(loss_iter_report)

                if weights:
                    weights_batch_report.extend(weights)

            metrics = process_metrics(results, domain_info['org_classes'])
            domain_info['miou_seeds'].append(metrics['mIoU'])
            domain_info['dice_seeds'].append(metrics['mDice'])
            domain_info['acc_seeds'].append(metrics['mAcc'])
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")

            if loss_batch_report:
                loss_batch_report = np.array(loss_batch_report)
                avg_loss_per_iter = np.mean(loss_batch_report, axis=0)
                domain_info['loss_seed_report'].append(avg_loss_per_iter)

            if weights_batch_report:
                weights_path = os.path.join(args.save_dir, "weights")

                weights = weights_batch_report
                weights = np.hstack(weights)
                os.makedirs(weights_path, exist_ok=True)

                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

            if args.runtime_calculation:
                if args.adapt:
                    adapt_times = adapt_method.adapt_times[adapt_len_before:]
                    mean_adapt_time = np.mean(adapt_times[20:]) if len(adapt_times) > 20 else (np.mean(adapt_times) if len(adapt_times) > 0 else 0)
                    std_adapt_time = np.std(adapt_times[20:]) if len(adapt_times) > 20 else (np.std(adapt_times) if len(adapt_times) > 0 else 0)
                else:
                    mean_adapt_time = 0
                    std_adapt_time = 0

                eval_times = adapt_method.eval_times[eval_len_before:]
                mean_eval_time = np.mean(eval_times[20:]) if len(eval_times) > 20 else (np.mean(eval_times) if len(eval_times) > 0 else 0)
                std_eval_time = np.std(eval_times[20:]) if len(eval_times) > 20 else (np.std(eval_times) if len(eval_times) > 0 else 0)

                mean_total_time = mean_adapt_time + mean_eval_time

                run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
                print(run_time_txt)

                runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
                with open(runtime_save_dir, 'a+') as f:
                    f.write(run_time_txt + "\n")

                adapt_time_all_corr.append(mean_adapt_time)
                eval_time_all_corr.append(mean_eval_time)

    for domain_info in domain_infos:
        corruption = domain_info['corruption']
        miou_mean = np.array(domain_info['miou_seeds']).mean()
        miou_std = np.array(domain_info['miou_seeds']).std()
        dice_mean = np.array(domain_info['dice_seeds']).mean()
        dice_std = np.array(domain_info['dice_seeds']).std()
        acc_mean = np.array(domain_info['acc_seeds']).mean()
        acc_std = np.array(domain_info['acc_seeds']).std()

        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(domain_info['c_results_path'], 'w') as f:
            f.write(headers + "\n")
            f.write(c_results_print)

        loss_mean, loss_inc, loss_dec = compute_loss_stats(domain_info['loss_seed_report'])

        all_results[corruption] = c_results_print
        domain_summary.append({
            'corruption': corruption,
            'mIoU_mean': miou_mean,
            'mIoU_std': miou_std,
            'mDice_mean': dice_mean,
            'mDice_std': dice_std,
            'mAcc_mean': acc_mean,
            'mAcc_std': acc_std,
            'loss_mean': loss_mean,
            'loss_increase': loss_inc,
            'loss_decrease': loss_dec,
        })

        if args.plot_loss and args.adapt and domain_info['loss_seed_report']:
            loss_seed_report = np.array(domain_info['loss_seed_report'])
            avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds) + 1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n===== Per-domain Summary =====")
    for domain_metrics in domain_summary:
        loss_str = ""
        if domain_metrics.get('loss_mean') is not None:
            loss_str = (
                f", Loss {domain_metrics['loss_mean']:.4f}"
                f", Loss+ {domain_metrics['loss_increase']:.4f}"
                f", Loss- {domain_metrics['loss_decrease']:.4f}"
            )
        print(
            f"{domain_metrics['corruption']}: "
            f"mIoU {domain_metrics['mIoU_mean']:.2f} +/- {domain_metrics['mIoU_std']:.2f}, "
            f"mDice {domain_metrics['mDice_mean']:.2f} +/- {domain_metrics['mDice_std']:.2f}, "
            f"mAcc {domain_metrics['mAcc_mean']:.2f} +/- {domain_metrics['mAcc_std']:.2f}"
            f"{loss_str}"
        )

    if eval_corruptions:
        summary_domains = [domain_metrics for domain_metrics in domain_summary if domain_metrics['corruption'] in eval_corruptions]
    else:
        summary_domains = domain_summary

    overall_miou_mean = np.mean([domain_metrics['mIoU_mean'] for domain_metrics in summary_domains])
    overall_mdice_mean = np.mean([domain_metrics['mDice_mean'] for domain_metrics in summary_domains])
    overall_macc_mean = np.mean([domain_metrics['mAcc_mean'] for domain_metrics in summary_domains])

    print("===== Overall Mean Summary =====")
    print(
        f"Overall mean across evaluation domains: "
        f"mIoU {overall_miou_mean:.2f}, "
        f"mDice {overall_mdice_mean:.2f}, "
        f"mAcc {overall_macc_mean:.2f}"
    )

    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")


def process_single_batch_no_adapt(args, device, adapt_method, data, domain_info):
    inputs = data['img_patches']
    labels = data['gt_patches']
    original_gts = data['gt']

    patch_grid_shape = data['meta']['patch_grid_shape']
    image_shapes = data['meta']['img_shape']
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    with torch.no_grad():
        patch_preds = adapt_method.evaluate(inputs)

    eval_size = getattr(adapt_method, 'eval_size', args.patch_size[0])
    eval_scale = eval_size / args.patch_size[0]

    if args.init_resize:
        if eval_scale < 1.0:
            scaled_patch_size = (round(args.patch_size[0] * eval_scale), round(args.patch_size[1] * eval_scale))
            scaled_patch_stride = round(args.patch_stride * eval_scale)
            scaled_img_shapes = [(round(h * eval_scale), round(w * eval_scale)) for h, w in image_shapes]
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, scaled_img_shapes, scaled_patch_size, scaled_patch_stride)
        else:
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
    else:
        reconstructed_preds = patch_preds

    batch_results = []
    for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):
        pd = pd.softmax(dim=0)

        if domain_info['ext_to_real_cls_indx'] is not None:
            pd = pd.unsqueeze(0)
            pd = (pd * domain_info['ext_to_real_cls_indx']).max(1)[0]

        pd = pd.argmax(dim=0)
        pd = pd.to(gt.device)
        gt = gt[0]
        if eval_scale < 1.0:
            target_h, target_w = scaled_img_shapes[idx]
            gt = torch.nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        batch_results.append(
            intersect_and_union(pd, gt, domain_info['num_org_classes'], domain_info['ignore_index'])
        )

    return batch_results, None, [], [], []


def prepare_domain_info(args, device, corruption, c_idx):
    data_loader, org_classes = segmentation_datasets.prepare_data(
        args.dataset,
        args.data_dir,
        args.init_resize,
        args.patch_size,
        args.patch_stride,
        corruption=corruption,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    if args.class_extensions and data_loader.dataset.class_extensions is not None:
        classes = data_loader.dataset.class_extensions
        print(f"\n+++ Using class extensions")
        print(f"+++ The number of classes [no extension]: {len(org_classes)}")
        print(f"+++ The number of classes after extension:  {len(classes)}")
        ext_to_real_cls_indx = torch.Tensor(data_loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
        num_cls, num_queries = max(ext_to_real_cls_indx) + 1, len(ext_to_real_cls_indx)
        ext_to_real_cls_indx = torch.nn.functional.one_hot(ext_to_real_cls_indx)
        ext_to_real_cls_indx = ext_to_real_cls_indx.T.view(num_cls, num_queries, 1, 1)
    else:
        classes = org_classes
        ext_to_real_cls_indx = None
        print(f"\n+++ The number of classes [no extension]: {len(org_classes)}")

    c_results_path = os.path.join(args.save_dir, f"{c_idx:02}_{corruption}", "results.txt")
    os.makedirs(os.path.dirname(c_results_path), exist_ok=True)

    return {
        'corruption': corruption,
        'data_loader': data_loader,
        'org_classes': org_classes,
        'classes': classes,
        'num_org_classes': len(org_classes),
        'ignore_index': data_loader.dataset.ignore_index,
        'ext_to_real_cls_indx': ext_to_real_cls_indx,
        'c_results_path': c_results_path,
        'miou_seeds': [],
        'dice_seeds': [],
        'acc_seeds': [],
        'loss_seed_report': [],
    }


def process_single_batch(args, device, adapt_method, data, domain_info):
    inputs = data['img_patches']
    labels = data['gt_patches']
    original_gts = data['gt']

    patch_grid_shape = data['meta']['patch_grid_shape']
    image_shapes = data['meta']['img_shape']
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    adapt_len_before = len(adapt_method.adapt_times) if args.runtime_calculation and args.adapt else None
    eval_len_before = len(adapt_method.eval_times) if args.runtime_calculation else None

    loss_iter_report = None
    if args.adapt:
        loss_iter_report = adapt_method.adapt(inputs)

    with torch.no_grad():
        patch_preds = adapt_method.evaluate(inputs)

    eval_size = getattr(adapt_method, 'eval_size', args.patch_size[0])
    eval_scale = eval_size / args.patch_size[0]

    if args.init_resize:
        if eval_scale < 1.0:
            scaled_patch_size = (round(args.patch_size[0] * eval_scale), round(args.patch_size[1] * eval_scale))
            scaled_patch_stride = round(args.patch_stride * eval_scale)
            scaled_img_shapes = [(round(h * eval_scale), round(w * eval_scale)) for h, w in image_shapes]
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, scaled_img_shapes, scaled_patch_size, scaled_patch_stride)
        else:
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
    else:
        reconstructed_preds = patch_preds

    batch_results = []
    for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):
        pd = pd.softmax(dim=0)

        if domain_info['ext_to_real_cls_indx'] is not None:
            pd = pd.unsqueeze(0)
            pd = (pd * domain_info['ext_to_real_cls_indx']).max(1)[0]

        pd = pd.argmax(dim=0)
        pd = pd.to(gt.device)
        gt = gt[0]
        if eval_scale < 1.0:
            target_h, target_w = scaled_img_shapes[idx]
            gt = torch.nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        batch_results.append(
            intersect_and_union(pd, gt, domain_info['num_org_classes'], domain_info['ignore_index'])
        )

    adapt_times = []
    eval_times = []
    if args.runtime_calculation and args.adapt:
        adapt_times = adapt_method.adapt_times[adapt_len_before:]
    if args.runtime_calculation:
        eval_times = adapt_method.eval_times[eval_len_before:]

    weights = []
    if adapt_method.model.weights_track:
        weights = list(adapt_method.model.weights_track)
        adapt_method.model.weights_track = []

    return batch_results, loss_iter_report, adapt_times, eval_times, weights


def summarize_results(results):
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )

    return {
        'mIoU': np.round(np.nanmean(ret_metrics['IoU']) * 100, 2),
        'mDice': np.round(np.nanmean(ret_metrics['Dice']) * 100, 2),
        'mAcc': np.round(np.nanmean(ret_metrics['Acc']) * 100, 2),
    }


def compute_loss_stats(loss_seed_report):
    if loss_seed_report is None or len(loss_seed_report) == 0:
        return None, None, None

    loss_arr = np.array(loss_seed_report, dtype=float)
    if loss_arr.size == 0 or np.all(np.isnan(loss_arr)):
        return None, None, None

    avg_loss_per_iter = np.nanmean(loss_arr, axis=0)
    avg_loss = float(np.nanmean(avg_loss_per_iter))

    if len(avg_loss_per_iter) > 1:
        deltas = np.diff(avg_loss_per_iter)
        positive_deltas = deltas[deltas > 0]
        negative_deltas = deltas[deltas < 0]
        avg_increase = float(np.mean(positive_deltas)) if len(positive_deltas) > 0 else 0.0
        avg_decrease = float(np.mean(negative_deltas)) if len(negative_deltas) > 0 else 0.0
    else:
        avg_increase = 0.0
        avg_decrease = 0.0

    return avg_loss, avg_increase, avg_decrease


def run_lifelong(args, device, start_time, all_results_path):
    headers = "mIoU, mDice, mAcc"
    all_results = dict()
    domain_summary = []
    adapt_time_all_corr = []
    eval_time_all_corr = []
    round_summary = [
        {
            'mIoU': [],
            'mDice': [],
            'mAcc': [],
        }
        for _ in range(args.lifelong_rnds)
    ]

    domain_infos = []
    for c_idx, corruption in enumerate(args.corruptions_list):
        domain_infos.append(prepare_domain_info(args, device, corruption, c_idx))

    args.classes = domain_infos[0]['classes']
    domain_map = {domain_info['corruption']: domain_info for domain_info in domain_infos}

    continual_methods = None
    if args.reset_mode == 'continual':
        continual_methods = [get_method(args, device) for _ in range(args.trials)]

    for t in range(args.trials):
        if args.reset_mode == 'continual':
            adapt_method = continual_methods[t]
        else:
            adapt_method = get_method(args, device)

        trial_results = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_loss_batch_report = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_adapt_times = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_eval_times = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_weights = {domain_info['corruption']: [] for domain_info in domain_infos}

        for round_idx in range(args.lifelong_rnds):
            round_results = {domain_info['corruption']: [] for domain_info in domain_infos}
            print(f"\n===== Lifelong Round {round_idx + 1}/{args.lifelong_rnds} | Trial {t} =====")

            if args.lifelong == 'shuffle_domain_pround':
                round_rng = np.random.default_rng(args.seed + round_idx)
                corruption_order = list(round_rng.permutation(args.corruptions_list))
                print(f"Round {round_idx + 1} domain order: {' -> '.join(corruption_order)}")

                for domain_order_idx, corruption in enumerate(corruption_order):

                    domain_info = domain_map[corruption]

                    if args.reset_mode == 'normal':
                        adapt_method.reset()

                    for batch_idx, data in tqdm(enumerate(domain_info['data_loader']), total=len(domain_info['data_loader'])):
                        if args.debug and batch_idx == 10:
                            break

                        if args.reset_mode == 'episodic':
                            adapt_method.reset()

                        batch_results, loss_iter_report, adapt_times, eval_times, weights = process_single_batch(
                            args,
                            device,
                            adapt_method,
                            data,
                            domain_info,
                        )

                        trial_results[corruption].extend(batch_results)
                        round_results[corruption].extend(batch_results)
                        if loss_iter_report is not None:
                            trial_loss_batch_report[corruption].append(loss_iter_report)
                        trial_adapt_times[corruption].extend(adapt_times)
                        trial_eval_times[corruption].extend(eval_times)
                        trial_weights[corruption].extend(weights)

            elif args.lifelong == 'shuffle_domain_pbatch':
                iterators = {domain_info['corruption']: iter(domain_info['data_loader']) for domain_info in domain_infos}
                active_corruptions = [domain_info['corruption'] for domain_info in domain_infos]
                cycle_idx = 0
                debug_counts = {domain_info['corruption']: 0 for domain_info in domain_infos}

                while active_corruptions:
                    cycle_rng = np.random.default_rng(args.seed + round_idx * 100000 + cycle_idx)
                    cycle_order = list(cycle_rng.permutation(active_corruptions))
                    print(f"Round {round_idx + 1} cycle {cycle_idx + 1} order: {' -> '.join(cycle_order)}")
                    next_active = []

                    for corruption in cycle_order:
                        if args.debug and debug_counts[corruption] == 10:
                            continue

                        try:
                            data = next(iterators[corruption])
                        except StopIteration:
                            continue

                        debug_counts[corruption] += 1
                        domain_info = domain_map[corruption]

                        if args.reset_mode in ('episodic', 'normal'):
                            adapt_method.reset()

                        batch_results, loss_iter_report, adapt_times, eval_times, weights = process_single_batch(
                            args,
                            device,
                            adapt_method,
                            data,
                            domain_info,
                        )

                        trial_results[corruption].extend(batch_results)
                        round_results[corruption].extend(batch_results)
                        if loss_iter_report is not None:
                            trial_loss_batch_report[corruption].append(loss_iter_report)
                        trial_adapt_times[corruption].extend(adapt_times)
                        trial_eval_times[corruption].extend(eval_times)
                        trial_weights[corruption].extend(weights)
                        next_active.append(corruption)

                    active_corruptions = next_active
                    cycle_idx += 1

            round_domain_metrics = []
            for domain_info in domain_infos:
                corruption = domain_info['corruption']
                metrics = summarize_results(round_results[corruption])
                round_domain_metrics.append(metrics)

            round_miou = np.mean([metrics['mIoU'] for metrics in round_domain_metrics])
            round_mdice = np.mean([metrics['mDice'] for metrics in round_domain_metrics])
            round_macc = np.mean([metrics['mAcc'] for metrics in round_domain_metrics])
            round_summary[round_idx]['mIoU'].append(round_miou)
            round_summary[round_idx]['mDice'].append(round_mdice)
            round_summary[round_idx]['mAcc'].append(round_macc)

            print(
                f"Round {round_idx + 1} final metrics: "
                f"mIoU {round_miou:.2f}, "
                f"mDice {round_mdice:.2f}, "
                f"mAcc {round_macc:.2f}"
            )

        for domain_info in domain_infos:
            corruption = domain_info['corruption']
            metrics = process_metrics(trial_results[corruption], domain_info['org_classes'])
            domain_info['miou_seeds'].append(metrics['mIoU'])
            domain_info['dice_seeds'].append(metrics['mDice'])
            domain_info['acc_seeds'].append(metrics['mAcc'])
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")

            if trial_loss_batch_report[corruption]:
                loss_batch_report = np.array(trial_loss_batch_report[corruption])
                avg_loss_per_iter = np.mean(loss_batch_report, axis=0)
                domain_info['loss_seed_report'].append(avg_loss_per_iter)

            if trial_weights[corruption]:
                weights_path = os.path.join(args.save_dir, "weights")
                weights = np.hstack(trial_weights[corruption])
                os.makedirs(weights_path, exist_ok=True)
                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

            if args.runtime_calculation:
                if args.adapt:
                    adapt_times = trial_adapt_times[corruption][20:] if len(trial_adapt_times[corruption]) > 20 else trial_adapt_times[corruption]
                    mean_adapt_time = np.mean(adapt_times) if adapt_times else 0
                    std_adapt_time = np.std(adapt_times) if adapt_times else 0
                else:
                    mean_adapt_time = 0
                    std_adapt_time = 0

                eval_times = trial_eval_times[corruption][20:] if len(trial_eval_times[corruption]) > 20 else trial_eval_times[corruption]
                mean_eval_time = np.mean(eval_times) if eval_times else 0
                std_eval_time = np.std(eval_times) if eval_times else 0
                mean_total_time = mean_adapt_time + mean_eval_time

                run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
                print(run_time_txt)

                runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
                with open(runtime_save_dir, 'a+') as f:
                    f.write(run_time_txt + "\n")

                adapt_time_all_corr.append(mean_adapt_time)
                eval_time_all_corr.append(mean_eval_time)

    for domain_info in domain_infos:
        corruption = domain_info['corruption']
        miou_mean = np.array(domain_info['miou_seeds']).mean()
        miou_std = np.array(domain_info['miou_seeds']).std()
        dice_mean = np.array(domain_info['dice_seeds']).mean()
        dice_std = np.array(domain_info['dice_seeds']).std()
        acc_mean = np.array(domain_info['acc_seeds']).mean()
        acc_std = np.array(domain_info['acc_seeds']).std()

        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(domain_info['c_results_path'], 'w') as f:
            f.write(headers + "\n")
            f.write(c_results_print)

        loss_mean, loss_inc, loss_dec = compute_loss_stats(domain_info['loss_seed_report'])

        all_results[corruption] = c_results_print
        domain_summary.append({
            'corruption': corruption,
            'mIoU_mean': miou_mean,
            'mIoU_std': miou_std,
            'mDice_mean': dice_mean,
            'mDice_std': dice_std,
            'mAcc_mean': acc_mean,
            'mAcc_std': acc_std,
            'loss_mean': loss_mean,
            'loss_increase': loss_inc,
            'loss_decrease': loss_dec,
        })

        if args.plot_loss and args.adapt and domain_info['loss_seed_report']:
            loss_seed_report = np.array(domain_info['loss_seed_report'])
            avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds) + 1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n===== Per-domain Summary =====")
    for domain_metrics in domain_summary:
        loss_str = ""
        if domain_metrics.get('loss_mean') is not None:
            loss_str = (
                f", Loss {domain_metrics['loss_mean']:.4f}"
                f", Loss+ {domain_metrics['loss_increase']:.4f}"
                f", Loss- {domain_metrics['loss_decrease']:.4f}"
            )
        print(
            f"{domain_metrics['corruption']}: "
            f"mIoU {domain_metrics['mIoU_mean']:.2f} +/- {domain_metrics['mIoU_std']:.2f}, "
            f"mDice {domain_metrics['mDice_mean']:.2f} +/- {domain_metrics['mDice_std']:.2f}, "
            f"mAcc {domain_metrics['mAcc_mean']:.2f} +/- {domain_metrics['mAcc_std']:.2f}"
            f"{loss_str}"
        )

    print("===== Per-round Summary =====")
    for round_idx, metrics in enumerate(round_summary):
        round_miou_mean = np.mean(metrics['mIoU'])
        round_miou_std = np.std(metrics['mIoU'])
        round_mdice_mean = np.mean(metrics['mDice'])
        round_mdice_std = np.std(metrics['mDice'])
        round_macc_mean = np.mean(metrics['mAcc'])
        round_macc_std = np.std(metrics['mAcc'])
        print(
            f"Round {round_idx + 1}: "
            f"mIoU {round_miou_mean:.2f} +/- {round_miou_std:.2f}, "
            f"mDice {round_mdice_mean:.2f} +/- {round_mdice_std:.2f}, "
            f"mAcc {round_macc_mean:.2f} +/- {round_macc_std:.2f}"
        )

    overall_miou_mean = np.mean([domain_metrics['mIoU_mean'] for domain_metrics in domain_summary])
    overall_mdice_mean = np.mean([domain_metrics['mDice_mean'] for domain_metrics in domain_summary])
    overall_macc_mean = np.mean([domain_metrics['mAcc_mean'] for domain_metrics in domain_summary])

    print("===== Overall Mean Summary =====")
    print(
        f"Overall mean across domains: "
        f"mIoU {overall_miou_mean:.2f}, "
        f"mDice {overall_mdice_mean:.2f}, "
        f"mAcc {overall_macc_mean:.2f}"
    )

    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")




if __name__ == "__main__":
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Run the main function with the parsed arguments
    main(args)
