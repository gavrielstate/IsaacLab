# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self, input_channels: int = 7):
        super().__init__()
        self.input_channels = input_channels


        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(1, 16),  # GroupNorm with 1 group is equivalent to LayerNorm over channels
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(1, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(1, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )


        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        # Apply normalization to RGB channels if they exist
        if self.input_channels >= 7:  # Has RGB + depth + segmentation
            x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
            if self.input_channels > 7:  # Has segmentation
                x[:, 4:7, :, :] = self.data_transforms(x[:, 4:7, :, :])
        elif self.input_channels >= 4:  # Has RGB + depth
            x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        elif self.input_channels >= 1:  # Has depth only
            pass  # No normalization for depth only
        
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    
    The feature extractor now supports optional image parameters:
    - At least one of rgb_img, depth_img, or segmentation_img must be provided
    - The CNN will automatically adapt its input channels based on the available image types
    - RGB images are normalized using ImageNet statistics
    - Depth images are normalized to [0, 1] range
    - Segmentation images are normalized and mean-subtracted
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str):
        """Initialize the feature extractor model.

        Args:
            cfg (FeatureExtractorCfg): Configuration for the feature extractor model.
            device (str): Device to run the model on.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model - will be initialized with correct input channels
        self.feature_extractor = None  # Will be set in step() method
        self.input_channels = None

        self.step_count = 0
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Store checkpoint path for later loading if needed
        self.checkpoint_path = None
        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                self.checkpoint_path = os.path.join(self.log_dir, latest_file)
                print(f"[INFO]: Found feature extractor checkpoint at {self.checkpoint_path}")

        if self.cfg.train:
            self.optimizer = None  # Will be set when model is created
            self.l2_loss = nn.MSELoss()

    def _preprocess_images(
        self, rgb_img: torch.Tensor | None = None, depth_img: torch.Tensor | None = None, 
        segmentation_img: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor | None): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor | None): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor | None): Segmentation image tensor. Shape: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]: Preprocessed RGB, depth, and segmentation
        """
        if rgb_img is not None:
            rgb_img = rgb_img / 255.0
        
        if depth_img is not None:
            # process depth image
            depth_img[depth_img == float("inf")] = 0
            depth_img /= 5.0
            depth_img /= torch.max(depth_img)
        
        if segmentation_img is not None:
            # process segmentation image
            segmentation_img = segmentation_img / 255.0
            mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
            segmentation_img -= mean_tensor
            
        return rgb_img, depth_img, segmentation_img

    def _save_images(self, rgb_img: torch.Tensor | None = None, depth_img: torch.Tensor | None = None, 
                    segmentation_img: torch.Tensor | None = None):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor | None): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor | None): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor | None): Segmentation image tensor. Shape: (N, H, W, 3).
        """
        if rgb_img is not None:
            save_images_to_file(rgb_img, "shadow_hand_rgb.png")
        if depth_img is not None:
            save_images_to_file(depth_img, "shadow_hand_depth.png")
        if segmentation_img is not None:
            save_images_to_file(segmentation_img, "shadow_hand_segmentation.png")

    def _initialize_model_if_needed(self, rgb_img: torch.Tensor | None = None, 
                                  depth_img: torch.Tensor | None = None, 
                                  segmentation_img: torch.Tensor | None = None):
        """Initialize the model with the correct number of input channels based on available images."""
        # Calculate input channels
        input_channels = 0
        if rgb_img is not None:
            input_channels += 3
        if depth_img is not None:
            input_channels += 1
        if segmentation_img is not None:
            input_channels += 3
            
        if input_channels == 0:
            raise ValueError("At least one image type (rgb_img, depth_img, or segmentation_img) must be provided")
            
        # Only reinitialize if input channels changed
        if self.input_channels != input_channels:
            self.input_channels = input_channels
            self.feature_extractor = FeatureExtractorNetwork(input_channels)
            self.feature_extractor.to(self.device)
            
            # Try to load checkpoint if available and compatible
            if self.cfg.load_checkpoint and self.checkpoint_path is not None:
                try:
                    checkpoint_state = torch.load(self.checkpoint_path, weights_only=True)
                    # Check if checkpoint is compatible with current model architecture
                    if checkpoint_state.get('cnn.0.weight', None) is not None:
                        checkpoint_input_channels = checkpoint_state['cnn.0.weight'].shape[1]
                        if checkpoint_input_channels == input_channels:
                            print(f"[INFO]: Loading feature extractor checkpoint from {self.checkpoint_path}")
                            self.feature_extractor.load_state_dict(checkpoint_state)
                        else:
                            print(f"[WARNING]: Checkpoint has {checkpoint_input_channels} input channels but current model has {input_channels}. Skipping checkpoint loading.")
                    else:
                        print(f"[WARNING]: Checkpoint format not recognized. Skipping checkpoint loading.")
                except Exception as e:
                    print(f"[WARNING]: Failed to load checkpoint: {e}")
            
            if self.cfg.train:
                self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
                self.feature_extractor.train()
            else:
                self.feature_extractor.eval()

    def step(
        self, rgb_img: torch.Tensor | None = None, depth_img: torch.Tensor | None = None, 
        segmentation_img: torch.Tensor | None = None, gt_pose: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor | None): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor | None): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor | None): Segmentation image tensor. Shape: (N, H, W, 3).
            gt_pose (torch.Tensor | None): Ground truth pose tensor (position and corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor | None, torch.Tensor]: Pose loss and predicted pose.
        """
        # Initialize model with correct input channels
        self._initialize_model_if_needed(rgb_img, depth_img, segmentation_img)

        # Preprocess images
        rgb_img, depth_img, segmentation_img = self._preprocess_images(rgb_img, depth_img, segmentation_img)

        if self.cfg.write_image_to_file:
            self._save_images(rgb_img, depth_img, segmentation_img)

        # Concatenate available images
        img_parts = []
        if rgb_img is not None:
            img_parts.append(rgb_img)
        if depth_img is not None:
            img_parts.append(depth_img)
        if segmentation_img is not None:
            img_parts.append(segmentation_img)
            
        img_input = torch.cat(img_parts, dim=-1)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
