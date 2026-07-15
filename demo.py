# for deterministic behavior in cublas
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from diffusers import AutoencoderKL, SD3Transformer2DModel
from pipelines.res_edit_pipeline import ResEditPipeline
from utils import encode_images, encode_intrinsics

# Global variables to store models
global_models = {
    'pipeline': None,
    'vae': None,
    'device': None,
    'weight_dtype': None
}

def load_text_encoders(class_one, class_two, class_three, pretrained_model_name_or_path, revision=None, variant=None):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision, variant=variant
    )
    text_encoder_three = class_three.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_3", revision=revision, variant=variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel

def create_zero_image(height=512, width=512):
    zero_image = np.zeros((height, width, 3), dtype=np.float32)
    return zero_image

def load_and_process_image(image_path, height=512, width=512, image_type="image", resize=True):
    if image_path is None:
        return create_zero_image(height, width)
    
    if isinstance(image_path, str) and not os.path.exists(image_path):
        return create_zero_image(height, width)
    
    # Handle PIL Image objects from Gradio
    if isinstance(image_path, Image.Image):
        image = image_path.convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')

    if resize:
        image = image.resize((width, height), Image.LANCZOS)

    img = np.array(image).astype(np.float32)
    
    if image_type in ['depth', 'roughness', 'metallic'] and img.ndim == 3:
        img = img[:, :, 0]
    
    infinite = ~np.isfinite(img)
    if np.all(infinite):
        img = 0.5 * np.ones_like(img)
        infinite = np.zeros_like(infinite, dtype=bool)
    
    if np.any(infinite):
        img[infinite] = img[~infinite].min()
    
    if image_type in ['image', 'albedo']:
        img = img.astype(np.float32) / 255.0
        img = img ** 2.2
        img = np.clip(img, 0.0, 1.0) ** (1 / 2.2)
    elif image_type == 'normal':
        img = img.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0
        img = (img + 1.0) / 2.0
    elif image_type == 'depth':
        img = img.astype(np.float32) / 255.0
        if img.max() > img.min():
            img = img / (img.max() + 1e-4)
    elif image_type == 'irradiance':
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32) / 255.0
    
    img = np.clip(img, 0.0, 1.0)
    
    if np.any(infinite):
        if img.ndim == 3:
            img[infinite] = 1.0
        else:
            img[infinite] = 0.0
    
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = np.repeat(img[:, :, np.newaxis] if img.ndim == 2 else img, 3, axis=2)
    
    return img

def initialize_models():
    """Initialize all models at startup"""
    global global_models
    
    # Model path
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"
    # Transformer model path for ResEdit
    transformer_model_path = "johnberg/resedit"
    revision = None
    variant = None
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    weight_dtype = torch.float32
    
    print(f"Initializing models on device: {device}")
    
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer_2", revision=revision
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer_3", revision=revision
    )

    # Import text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder_3")

    # Load models
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, pretrained_model_name_or_path, revision, variant
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant)
    sd3_transformer = SD3Transformer2DModel.from_pretrained(transformer_model_path, subfolder="transformer", revision=revision, variant=variant)

    # Set models to evaluation mode
    sd3_transformer.eval()
    sd3_transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # Move models to device
    vae.to(device, dtype=torch.float32)
    sd3_transformer.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    text_encoder_three.to(device, dtype=weight_dtype)

    # Create pipeline
    pipeline = ResEditPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=sd3_transformer,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        torch_dtype=weight_dtype,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    
    # Store in global dict
    global_models.update({
        'pipeline': pipeline,
        'vae': vae,
        'device': device,
        'weight_dtype': weight_dtype
    })
    
    print("All models loaded successfully!")
    return "Models initialized successfully!"

def process_resedit(
    input_image,
    normal_original, albedo_original, roughness_original, irradiance_original,
    normal_edited, albedo_edited, roughness_edited, irradiance_edited,
    num_inference_steps, guidance_scale, seed, image_guidance_scale,
    residual_opt_steps, residual_lr, diffusion_loss_weight,
    step_size, max_fixed_point_iters, convergence_threshold, optimize_residual,
    adv_weight, adv_target, adv_lr, step_ratio, pool_target, inversion_type,
):
    """Main processing function for resedit"""
    
    if input_image is None:
        return "Please upload an input RGB image", None, None, None
    
    pipeline = global_models['pipeline']
    vae = global_models['vae']
    device = global_models['device']
    weight_dtype = global_models['weight_dtype']
    
    if pipeline is None:
        return "Models not initialized. Please wait for initialization to complete.", None, None, None
    
    try:
        # Process input image size
        # dummy load to get size without resizing
        temp_img = load_and_process_image(input_image, image_type="image", resize=False)
        height = temp_img.shape[0] // 16 * 16
        width = temp_img.shape[1] // 16 * 16
        
        # Load and process all images
        input_processed = load_and_process_image(input_image, height, width, "image")
        
        # Original intrinsics
        normal_orig = load_and_process_image(normal_original, height, width, "normal")
        albedo_orig = load_and_process_image(albedo_original, height, width, "albedo")
        roughness_orig = load_and_process_image(roughness_original, height, width, "roughness")
        irradiance_orig = load_and_process_image(irradiance_original, height, width, "irradiance")
        
        # Edited intrinsics
        normal_edit = load_and_process_image(normal_edited, height, width, "normal") if normal_edited else normal_orig
        albedo_edit = load_and_process_image(albedo_edited, height, width, "albedo") if albedo_edited else albedo_orig
        roughness_edit = load_and_process_image(roughness_edited, height, width, "roughness") if roughness_edited else roughness_orig
        irradiance_edit = load_and_process_image(irradiance_edited, height, width, "irradiance") if irradiance_edited else irradiance_orig
        
        # Create image stacks
        multichannels = [input_processed, normal_orig, albedo_orig, roughness_orig, irradiance_orig]
        multichannels_edited = [input_processed, normal_edit, albedo_edit, roughness_edit, irradiance_edit]
        
        # Convert to tensors
        image_stack = np.stack(multichannels)
        image_stack = torch.from_numpy(image_stack).permute(0, 3, 1, 2).float() * 2.0 - 1.0
        image_stack = image_stack.unsqueeze(0).to(device, dtype=weight_dtype)
        
        image_stack_edited = np.stack(multichannels_edited)
        image_stack_edited = torch.from_numpy(image_stack_edited).permute(0, 3, 1, 2).float() * 2.0 - 1.0
        image_stack_edited = image_stack_edited.unsqueeze(0).to(device, dtype=weight_dtype)
        
        # Extract pixel values and control images
        pixel_values = image_stack[:, 0:1, :, :, :].squeeze(1)
        control_images = image_stack[:, 1:, :, :, :]
        control_images_edited = image_stack_edited[:, 1:, :, :, :]
        
        # Encode control images
        control_latents = encode_intrinsics(control_images, vae.to(device), weight_dtype)
        control_latents_edited = encode_intrinsics(control_images_edited, vae.to(device), weight_dtype)

        # Encode target image for inversion
        with torch.inference_mode():
            target_latents = encode_images(pixel_values, vae, weight_dtype)
        
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        autocast_ctx = torch.autocast(device.type, weight_dtype)
        
        with autocast_ctx:
            # Inversion with residual optimization
            inverted_latents, residual_embeds, pooled_projections = pipeline.forward_diffusion(
                prompt=[""],
                control_image=control_latents,
                latents=target_latents,
                num_inference_steps=30,
                guidance_scale=1.0,
                height=height, width=width,
                inversion_type=inversion_type,
                step_size=step_size,
                max_iters=max_fixed_point_iters,
                th=convergence_threshold,
                residual_optimization=optimize_residual,
                residual_opt_steps=residual_opt_steps,
                residual_lr=residual_lr,
                seed=seed,
                diffusion_weight=diffusion_loss_weight,
                adv_weight=adv_weight,
                adv_target=adv_target,
                adv_lr=adv_lr,
                step_ratio=step_ratio,
                pool_target=pool_target,
            )
        
        # Generate reconstruction and edit images
        print("Generating output images...")
        generator.manual_seed(seed)
        with torch.inference_mode():
            with autocast_ctx:
                # Reconstruction
                print("Generating reconstruction...")
                reconstructed_images = pipeline(
                    prompt_embeds=residual_embeds,
                    pooled_prompt_embeds=pooled_projections,
                    negative_prompt_embeds=residual_embeds,
                    negative_pooled_prompt_embeds=pooled_projections,
                    control_image=control_latents,
                    control_image_edited=control_latents,
                    image_guidance_scale=image_guidance_scale,
                    latents=inverted_latents,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=1.0,
                    generator=generator,
                    height=height, width=width,
                ).images
                
                # Edited
                print("Generating edited image...")
                generator.manual_seed(seed)  # Reset generator for random noise
                edited_images = pipeline(
                    prompt_embeds=residual_embeds,
                    pooled_prompt_embeds=pooled_projections,
                    negative_prompt_embeds=residual_embeds,
                    negative_pooled_prompt_embeds=pooled_projections,
                    control_image=control_latents,
                    control_image_edited=control_latents_edited,
                    image_guidance_scale=image_guidance_scale,
                    latents=inverted_latents,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height, width=width,
                ).images
        
        print("Preparing return values...")
        # Ensure all images are PIL Images for Gradio compatibility
        reconstructed_pil = reconstructed_images[0]
        edited_pil = edited_images[0]

        # Store images for saving
        results_dict = {
            'input_image': input_image,
            'reconstruction': reconstructed_pil,
            'edit': edited_pil,
        }
        
        print("Returning results...")
        return ("Processing completed successfully!", 
                reconstructed_pil, 
                edited_pil,
                results_dict,
                )
        
    except Exception as e:
        import traceback
        error_msg = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Print to console for debugging
        return error_msg, None, None, None

def save_results(results_dict, output_dir="./gradio_results"):
    """Save all generated images"""
    if results_dict is None:
        return "No results to save"
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
                
        for name, image in results_dict.items():
            if image is not None:
                if isinstance(image, Image.Image):
                    image.save(output_path / f"{name}.png")
                elif hasattr(image, 'save'):  # PIL-like object
                    image.save(output_path / f"{name}.png")
        
        return f"Results saved to {output_path}"
    except Exception as e:
        return f"Error saving results: {str(e)}"

# Initialize models on startup
print("Initializing models...")
init_status = initialize_models()
print(init_status)

# Create Gradio interface
with gr.Blocks(title="ResEdit Demo") as demo:
    gr.Markdown("# ResEdit: Residual embeddings for precise generative image editing")
    gr.Markdown("Upload an RGB image and provide/edit intrinsic properties (normal, albedo, roughness, irradiance)")
    
    # Store results for saving
    results_state = gr.State(None)
    
    with gr.Row():
        # Left side - Inputs
        with gr.Column(scale=1):
            gr.Markdown("## Inputs")

            gr.Markdown("### Upload RGB Image")
            input_image = gr.Image(label="Input RGB Image", type="pil")
            
            with gr.Row():
                # First column - Original images
                with gr.Column():
                    gr.Markdown("### Original Intrinsics")
                    normal_original = gr.Image(label="Normal (Original)", type="pil")
                    albedo_original = gr.Image(label="Albedo (Original)", type="pil")
                    roughness_original = gr.Image(label="Roughness (Original)", type="pil")
                    irradiance_original = gr.Image(label="Irradiance (Original)", type="pil")
                
                # Second column - Edited images
                with gr.Column():
                    gr.Markdown("### Edited Intrinsics")
                    normal_edited = gr.Image(label="Normal (Edited)", type="pil")
                    albedo_edited = gr.Image(label="Albedo (Edited)", type="pil")
                    roughness_edited = gr.Image(label="Roughness (Edited)", type="pil")
                    irradiance_edited = gr.Image(label="Irradiance (Edited)", type="pil")
        
        # Right side - Parameters and outputs
        with gr.Column(scale=1):
            gr.Markdown("## Parameters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Diffusion Parameters")
                    num_inference_steps = gr.Slider(1, 100, value=30, step=1, label="Inference Steps")
                    guidance_scale = gr.Slider(1.0, 15.0, value=1.0, step=0.5, label="Guidance Scale")
                    image_guidance_scale = gr.Slider(1.0, 15.0, value=1.0, step=0.1, label="Image Guidance Scale")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                
                with gr.Column():
                    gr.Markdown("### Optimization Parameters")
                    optimize_residual = gr.Checkbox(value=True, label="Optimize Residual")
                    residual_opt_steps = gr.Slider(50, 2000, value=400, step=10, label="Residual Optimization Steps")
                    residual_lr = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Residual Learning Rate")
                    diffusion_loss_weight = gr.Slider(0.0, 20.0, value=1.0, step=1.0, label="Diffusion Loss Weight")

                    # Adversarial parameters
                    adv_weight = gr.Slider(0.0, 2.0, value=0.015, step=0.01, label="Adversarial Weight λ_adv")
                    adv_target = gr.Dropdown(
                        choices=["normal", "albedo", "roughness", "irradiance"],
                        value="albedo",
                        label="Adversarial Target Intrinsic"
                    )
                    adv_lr = gr.Slider(1e-4, 1e-2, value=5e-3, step=1e-4, label="Adversary LR")
                    step_ratio = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Warmup Ratio for Adversary")
                    pool_target = gr.Checkbox(value=False, label="Use Pooling for Adversary Target")
            
            with gr.Row():
                gr.Markdown("### Inversion Parameters")
                step_size = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Step Size")
                max_fixed_point_iters = gr.Slider(10, 100, value=40, step=5, label="Max Fixed Point Iters")
                convergence_threshold = gr.Slider(1e-4, 1e-2, value=2e-3, step=1e-4, label="Convergence Threshold")
                inversion_type = gr.Dropdown(
                    choices=["exact", "DDIM"], value="exact",
                    label="Inversion type (exact or DDIM)"
                )
            
            with gr.Row():
                process_btn = gr.Button("Process", variant="primary", size="lg")
                save_btn = gr.Button("Save Results", variant="secondary")
            
            status_text = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("## Outputs")
            
            with gr.Row():
                with gr.Column():
                    reconstruction = gr.Image(label="Reconstruction", type="pil", format="png")
                with gr.Column():
                    edit_result = gr.Image(label="Edit", type="pil", format="png")
    
    # Event handlers
    process_btn.click(
        fn=process_resedit,
        inputs=[
            input_image,
            normal_original, albedo_original, roughness_original, irradiance_original,
            normal_edited, albedo_edited, roughness_edited, irradiance_edited,
            num_inference_steps, guidance_scale, seed, image_guidance_scale,
            residual_opt_steps, residual_lr, diffusion_loss_weight,
            step_size, max_fixed_point_iters, convergence_threshold, optimize_residual,
            adv_weight, adv_target, adv_lr, step_ratio, pool_target, inversion_type,
        ],
        outputs=[
            status_text, reconstruction, edit_result, results_state
        ]
    )
    
    save_btn.click(
        fn=save_results,
        inputs=[results_state],
        outputs=[status_text]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
