import torch
from PIL import Image
import os
import re
from PIL import Image


from models.controlnext_vid_svd import ControlNeXtSDVModel
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline

from diffusers.utils import load_image, export_to_video, export_to_gif

import torch
from PIL import Image
import os
import re
from PIL import Image


from models.controlnext_vid_svd import ControlNeXtSDVModel
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline

from diffusers.utils import load_image, export_to_video, export_to_gif

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))
        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images

if __name__ == "__main__":

    # controlnext = ControlNeXtSDVModel.from_pretrained(
    #     "./outputs/event_motion_svd",
    #     subfolder="controlnext",
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=False,
    # )
    # unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
    #     "./outputs/event_motion_svd",
    #     subfolder="unet",
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=False,
    # )
    # unet = UNetSpatioTemporalConditionControlNeXtModel()
    # unet.load_state_dict(torch.load("pretrained/svd_pose/unet.bin"))


    # Initialize and convert the controlnext model to half-precision
    controlnext = ControlNeXtSDVModel().cuda().half()

    # Load and convert the controlnext state dictionary
    state_dict_contrlnext = torch.load("pretrained/svd_pose/controlnet.bin")
    for name, param in state_dict_contrlnext.items():
        param = param.cuda().half()  # Move to GPU and convert to float16
        controlnext.state_dict()[name].copy_(param)
        del param  # Free memory
    del state_dict_contrlnext  # Free memory

    # Initialize and convert the unet model to half-precision
    unet = UNetSpatioTemporalConditionControlNeXtModel().cuda().half()

    # Load and convert the unet state dictionary
    state_dict_unet = torch.load("pretrained/svd_pose/unet.bin", map_location='cpu')
    for name, param in state_dict_unet.items():
        param = param.cuda().half()  # Move to GPU and convert to float16
        unet.state_dict()[name].copy_(param)
        del param  # Free memory
    del state_dict_unet  # Free memory



    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        unet=unet,
        controlnext=controlnext,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16, variant="fp16", local_files_only=True,
        cache_dir="/root/autodl-tmp/models_cache"
    )

    validation_image_dir = "./example/reference/01.jpeg"
    validation_control_folder = "/root/autodl-tmp/TikTok_event/dwpose_train_set/video_001"
    num_frames = 7
    device = "cuda"
    wdtype=torch.float16

    validation_image = Image.open(validation_image_dir).convert('RGB')
    validation_control_images = load_images_from_folder(validation_control_folder)
    validation_control_images = validation_control_images[:num_frames]

    validation_image = validation_image.resize((576, 1024))

    generator = torch.manual_seed(2024)

    pipe = pipe.to(device)

    with torch.inference_mode():
        frames = pipe(
            validation_image,
            validation_control_images,
            control_scale=0.9,
            num_frames=num_frames,
            width=576,
            height=1024,
            decode_chunk_size=4,
            generator=generator,
            motion_bucket_id=127,
            fps=7, 
            num_inference_steps=30
            ).frames[0]
    export_to_video(frames, "generated.mp4", fps=7)