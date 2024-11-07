import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm  # Ensure that tqdm is installed
import cv2

class TikTokEvent(Dataset):
    def __init__(
            self,
            root_dir,  # TikTok_event root directory
            subset='train',  # Default to loading the training set; options are 'train', 'val', 'test'
            width=604,
            height=1080,
            sample_n_frames=14,
            interval_frame=1,
            ref_aug=True,
            ref_aug_ratio=0.9
        ):
        self.root_dir = root_dir
        self.subset = subset
        self.sample_n_frames  = sample_n_frames
        self.width            = width
        self.height           = height
        self.interval_frame   = interval_frame
        self.ref_aug          = ref_aug
        self.ref_aug_ratio    = ref_aug_ratio

        sample_size = (height, width)

        # Define pixel transformations
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        # Define event transformations (previously guide_transforms)
        self.event_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
        ])

        # Get data paths based on the subset
        self.data_paths = self._get_data_paths()

    def _is_image_file(self, filename):
        """Check if a file is an image."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

    def _get_data_paths(self):
        """Traverse the root directory to find all video frame files, event frames, and dwpose frames."""
        data_paths = []
        subset_mapping = {
            'train': 'train_set',
            'val': 'val_set',
            'test': 'test_set'
        }

        set_name = subset_mapping.get(self.subset)
        if not set_name:
            raise ValueError(f"Invalid subset name: {self.subset}. Choose from 'train', 'val', 'test'.")

        # Define directories for each subset
        image_source = os.path.join(self.root_dir, set_name)
        event_source = os.path.join(self.root_dir, f"event_{set_name}")
        dwpose_source = os.path.join(self.root_dir, f"dwpose_{set_name}")  # Retained for dwpose

        # Traverse each video folder in image_source
        for video_dir in sorted(os.listdir(image_source)):
            video_image_folder = os.path.join(image_source, video_dir)
            video_event_folder = os.path.join(event_source, video_dir)
            video_dwpose_folder = os.path.join(dwpose_source, video_dir)

            # Check that all related folders exist
            if not (os.path.isdir(video_image_folder) and
                    os.path.isdir(video_event_folder) and
                    os.path.isdir(video_dwpose_folder)):
                print(f"Warning: Video folder '{video_dir}' is missing in some subsets, skipping.")
                continue

            # Get all image files
            image_files = sorted([f for f in os.listdir(video_image_folder) if self._is_image_file(f)])
            event_files = sorted([f for f in os.listdir(video_event_folder) if self._is_image_file(f)])
            dwpose_files = sorted([f for f in os.listdir(video_dwpose_folder) if self._is_image_file(f)])

            # Ensure all types of files have the same count
            min_len = min(len(image_files), len(event_files), len(dwpose_files))
            if min_len == 0:
                print(f"Warning: At least one type of image file is empty in video folder '{video_dir}', skipping.")
                continue

            # Keep only files up to the minimum length
            image_files = image_files[:min_len]
            event_files = event_files[:min_len]
            dwpose_files = dwpose_files[:min_len]

            data_paths.append({
                'video_frames_dir': video_image_folder,
                'event_frames_dir': video_event_folder,
                'dwpose_frames_dir': video_dwpose_folder,
                'video_frames': image_files,
                'event_frames': event_files,
                'dwpose_frames': dwpose_files
            })

        return data_paths

    def load_frames(self, frame_paths, base_dir, transforms_pipeline=None):
        """Load and transform frames."""
        frames = [Image.open(os.path.join(base_dir, frame_path)) for frame_path in frame_paths]
        frames = [pil_image_to_numpy(frame) for frame in frames]
        frames = np.array(frames)
        if transforms_pipeline:
            frames = transforms_pipeline(torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.)
        else:
            frames = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.
        return frames

    def get_batch(self, idx):
        while True:
            try:
                data_info = self.data_paths[idx]
                video_frames_dir = data_info['video_frames_dir']
                event_frames_dir = data_info['event_frames_dir']
                dwpose_frames_dir = data_info['dwpose_frames_dir']

                video_frames = data_info['video_frames']
                event_frames = data_info['event_frames']
                dwpose_frames = data_info['dwpose_frames']

                length = len(video_frames)
                assert length >= self.sample_n_frames * self.interval_frame, f"Too few frames in '{video_frames_dir}'. Required: {self.sample_n_frames * self.interval_frame}, Found: {length}"

                # Calculate maximum starting frame index
                max_start_frame = (length - self.sample_n_frames * self.interval_frame) // self.interval_frame
                bg_frame_id = random.randint(0, max_start_frame) * self.interval_frame

                # Select frame indices
                frame_ids = list(range(bg_frame_id, bg_frame_id + self.sample_n_frames * self.interval_frame, self.interval_frame))

                # Select frame paths
                selected_video_frames = [video_frames[i] for i in frame_ids]
                selected_event_frames = [event_frames[i] for i in frame_ids]
                selected_dwpose_frames = [dwpose_frames[i] for i in frame_ids]

                # Load frame data
                pixel_values = self.load_frames(selected_video_frames, video_frames_dir, self.pixel_transforms)
                event_values = self.load_frames(selected_event_frames, event_frames_dir, self.event_transforms)
                dwpose_values = self.load_frames(selected_dwpose_frames, dwpose_frames_dir, self.event_transforms)

                # Randomly select reference_image
                reference_id = random.randint(0, length - 1)
                reference_image_path = os.path.join(video_frames_dir, video_frames[reference_id])
                reference_image = Image.open(reference_image_path)
                reference_image = pil_image_to_numpy(reference_image)
                reference_image = torch.from_numpy(reference_image.transpose(2, 0, 1)).float() / 255.

                # Random cropping
                vid_width = pixel_values.shape[-1]
                vid_height = pixel_values.shape[-2]
                if vid_height / vid_width > self.height / self.width:
                    crop_width = vid_width
                    crop_height = int(vid_width * self.height / self.width)
                    h0 = random.randint(0, vid_height - crop_height)
                    w0 = 0
                else:
                    crop_width = int(vid_height * self.width / self.height)
                    crop_height = vid_height
                    h0 = 0
                    w0 = random.randint(0, vid_width - crop_width)

                # Crop video frames
                pixel_values = pixel_values[:, :, h0:h0 + crop_height, w0:w0 + crop_width]
                event_values = event_values[:, :, h0:h0 + crop_height, w0:w0 + crop_width]
                dwpose_values = dwpose_values[:, :, h0:h0 + crop_height, w0:w0 + crop_width]
                reference_image = reference_image[:, h0:h0 + crop_height, w0:w0 + crop_width]

                return pixel_values, event_values, dwpose_values, reference_image
            except Exception as e:
                print(f"****** Failed to load: {data_info['video_frames_dir']} ******")
                print(f"Error: {e}")
                idx = random.randint(0, len(self.data_paths) - 1)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        pixel_values, event_values, dwpose_values, reference_image = self.get_batch(idx)

        sample = dict(
            pixel_values=pixel_values,
            event_values=event_values,
            dwpose_values=dwpose_values,
            reference_image=reference_image
        )
        return sample

# Utility functions
def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255.

if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/TikTok_event"  # Replace with the root directory of TikTok_event
    save_dir = "./saved_images"  # Directory to save images
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Load the training set
    dataset_train = TikTokEvent(
        root_dir=root_dir,
        subset='train',  # Load the training set
        width=604,
        height=1080,
        sample_n_frames=14,
        interval_frame=1,
        ref_aug=False
    )

    # Test loading the training set
    idx = random.randint(0, len(dataset_train) - 1)
    sample = dataset_train[idx]

    # Check the loaded results
    pixel_values = sample["pixel_values"]
    event_values = sample["event_values"]
    dwpose_values = sample["dwpose_values"]
    reference_image = sample["reference_image"]

    # Save each loaded image
    for i in range(pixel_values.shape[0]):
        pixel_image = (pixel_values[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        event_image = (event_values[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        dwpose_image = (dwpose_values[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # Save images
        Image.fromarray(pixel_image).save(os.path.join(save_dir, f"pixel_frame_{idx}_{i}.png"))
        Image.fromarray(event_image).save(os.path.join(save_dir, f"event_frame_{idx}_{i}.png"))
        Image.fromarray(dwpose_image).save(os.path.join(save_dir, f"dwpose_frame_{idx}_{i}.png"))

    # Save the reference image
    reference_image_np = (reference_image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(reference_image_np).save(os.path.join(save_dir, f"reference_image_{idx}.png"))

    print(f"Saved images for index {idx} in '{save_dir}'.")

    # Check if the data matches the expected shapes
    assert pixel_values.shape == (14, 3, 1080, 604), "Pixel values shape mismatch!"
    assert event_values.shape == (14, 3, 1080, 604), "Event values shape mismatch!"
    assert dwpose_values.shape == (14, 3, 1080, 604), "DwPose values shape mismatch!"
    assert reference_image.shape == (3, 1080, 604), "Reference image shape mismatch!"

    print("Data loading test passed successfully!")
