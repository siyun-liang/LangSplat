import numpy as np
import os
import random
import torch
import torchvision

from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from autoencoder.model import Autoencoder
from gaussian_renderer import GaussianModel, render
from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
from types import SimpleNamespace
from utils.general_utils import safe_state


scene_gt_frames = {
    "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
    "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
    "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
    "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
}
    
scene_texts = {
    "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
            'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
            'pour-over vessel', 'knife', 'yellow desk'],
    "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
            'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles'],
    "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
    "teatime": ['sheep', 'yellow pouf', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
            'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
            'bag of cookies']
}


def activate_stream(gs_lang_feat, clip_model, gs_xyz, k=10, thresh=0.4):
    valid_map_3d = clip_model.get_max_across_3d(gs_lang_feat)
    n_prompt, n = valid_map_3d.shape
    
    # smooth the relevancy map, similar to in 2D
    gs_xyz_np = gs_xyz.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(gs_xyz_np)
    _, indices = nbrs.kneighbors(gs_xyz_np)
    indices = torch.from_numpy(indices).to(valid_map_3d.device)
    relv_map_smoothed = torch.zeros_like(valid_map_3d)
    gs_mask_pred = torch.zeros_like(valid_map_3d)
    for i in range(n_prompt):
        relv_1d = valid_map_3d[i]  
        neighbors_vals = relv_1d[indices]  
        neighbors_avg = neighbors_vals.mean(dim=1)  
        relv_map_smoothed[i] = 0.5 * (relv_1d + neighbors_avg)
    
        output = relv_map_smoothed[i]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)
        
        gs_mask_pred[i] = output > thresh
    
    return gs_mask_pred > 0.5  # convert float to bool


def render_set(output_dir, views, gaussians, pipeline, background, scene_name, text_indices=[], gs_masks_pred=[]):    
    rgb_path = os.path.join(output_dir, "renders")
    alpha_path = os.path.join(output_dir, "renders_silhouette")
    
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(alpha_path, exist_ok=True)
    
    target_text = scene_texts[scene_name]
    
    opt = {}
    for i, text_idx in enumerate(text_indices):
        opt = SimpleNamespace(include_feature=False, mask=gs_masks_pred[i])
        
        print(f"rendering the {text_idx+1}-th query of {len(target_text)} texts: {target_text[text_idx]}")
        
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            output = render(view, gaussians, pipeline, background, opt)

            rendering = output["render"]
            rendering_alpha = output["render_alpha"]
            
            torchvision.utils.save_image(rendering, os.path.join(rgb_path, view.image_name + f"_{target_text[text_idx]}.png"))
            torchvision.utils.save_image(rendering_alpha, os.path.join(alpha_path, view.image_name + f"_{target_text[text_idx]}.png"))
    

def evaluate(gaussians, model=None, clip_model=None, thresh=0.4, num_knn=10, device="cuda"):
    assert model is not None
    assert clip_model is not None
    
    # load language feature field on 3DGS and restore to 512
    gs_lang_feat = gaussians.get_language_feature
    if gs_lang_feat.shape[1] == 3:  # low dim
        with torch.no_grad():       
            gs_lang_feat = model.decode(gs_lang_feat)
    
    valid_map_3d = clip_model.get_max_across_3d(gs_lang_feat)
    n_prompt, n = valid_map_3d.shape
    
    # smooth the relevancy map, similar to in 2D
    gs_xyz_np = gaussians._xyz.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=num_knn).fit(gs_xyz_np)
    _, indices = nbrs.kneighbors(gs_xyz_np)
    indices = torch.from_numpy(indices).to(valid_map_3d.device)
    relv_map_smoothed = torch.zeros_like(valid_map_3d)
    gs_masks_pred = torch.zeros_like(valid_map_3d)
    scores = torch.zeros(n_prompt)
    
    for i in range(n_prompt):
        relv_1d = valid_map_3d[i]  
        neighbors_vals = relv_1d[indices]  
        neighbors_avg = neighbors_vals.mean(dim=1)  
        relv_map_smoothed[i] = 0.5 * (relv_1d + neighbors_avg)
    
        output = relv_map_smoothed[i]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)
        
        gs_masks_pred[i] = output > thresh
        scores[i] = relv_map_smoothed.max()
    
    del gs_lang_feat
    return scores, gs_masks_pred > 0.5
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                ae_ckpt_path: str, encoder_hidden_dims: list, decoder_hidden_dims: list, scene_name: str, dataset_name: str,
                base_dir: str, output_dir: str, device="cuda"):
    with torch.no_grad():
        # load AutoEncoder
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        bg_color = [1,1,1]  # white background
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # get text features
        clip_model = OpenCLIPNetwork(device)
        target_text = scene_texts[scene_name]
        clip_model.set_positives(target_text)

        scores = []
        gs_mask_preds = []
        for i in range(1,4):
            dataset.model_path = f"{base_dir}_{i}"
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, dataset_name=dataset_name, scene_name=scene_name)
            checkpoint = os.path.join(dataset.model_path, 'chkpnt30000.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            score_lvl, gs_mask_pred_lvl = evaluate(gaussians, model, clip_model)
            scores.append(score_lvl)
            gs_mask_preds.append(gs_mask_pred_lvl)
        
        
        chosen_levels = torch.argmax(torch.stack(scores), dim=0)
        level_wise_texts = {f"{i}": [] for i in range(1, 4)}
        level_wise_gs_mask_preds = {f"{i}": [] for i in range(1, 4)}
        
        for text_idx, best_level in enumerate(chosen_levels):
            level_wise_texts[f"{best_level+1}"].append(text_idx)
            level_wise_gs_mask_preds[f"{best_level+1}"].append(gs_mask_preds[best_level][text_idx])
        
        for i in range(1, 4):
            if len(level_wise_texts[f"{i}"]) > 0:
                dataset.model_path = f"{base_dir}_{i}"
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, dataset_name=dataset_name, scene_name=scene_name)
                checkpoint = os.path.join(dataset.model_path, 'chkpnt30000.pth')
                (model_params, first_iter) = torch.load(checkpoint)
                gaussians.restore(model_params, args, mode='test')
        
                if not skip_train:
                    render_set(output_dir, scene.getTrainCameras(), gaussians, pipeline, background, scene_name, level_wise_texts[f"{i}"], level_wise_gs_mask_preds[f"{i}"])
                if not skip_test:
                    render_set(output_dir,scene.getTestCameras(), gaussians, pipeline, background, scene_name, level_wise_texts[f"{i}"], level_wise_gs_mask_preds[f"{i}"])

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="prompt any label in 3DGS space")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--dataset_name", type=str, default = "lerf_ovs")
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh

    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, args.scene_name, "best_ckpt.pth")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, ae_ckpt_path, args.encoder_dims, args.decoder_dims, args.scene_name, args.dataset_name, args.base_dir, args.output_dir)
