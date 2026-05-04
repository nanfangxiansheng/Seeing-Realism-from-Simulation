# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from einops import repeat
from omegaconf import OmegaConf

from cosmos_transfer2._src.imaginaire.utils.config_helper import override
from cosmos_transfer2._src.predict2.configs.common.defaults.ema import PowerEMAConfig
from cosmos_transfer2._src.predict2.configs.common.defaults.tokenizer import DummyJointImageVideoConfig
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import (
    VideoPredictionControlConditioner,
    VideoPredictionControlConditionerImageContext,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.net import TRANSFER2_CONTROL2WORLD_NET_2B
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace import (
    ControlVideo2WorldConfig,
    ControlVideo2WorldModel,
)

"""
pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_test.py --all
"""


@pytest.fixture
def control_video2world_config():
    control_video2world_model_config = ControlVideo2WorldConfig(
        tokenizer=DummyJointImageVideoConfig,
        conditioner=VideoPredictionControlConditioner,
        net=TRANSFER2_CONTROL2WORLD_NET_2B,
        ema=PowerEMAConfig,
        state_t=3,
    )
    control_video2world_model_config = override(control_video2world_model_config)
    return control_video2world_model_config


@pytest.fixture
def control_video2world_with_image_context_config():
    control_video2world_model_config = ControlVideo2WorldConfig(
        tokenizer=DummyJointImageVideoConfig,
        conditioner=VideoPredictionControlConditionerImageContext,  # Use image context conditioner
        net=TRANSFER2_CONTROL2WORLD_NET_2B,
        ema=PowerEMAConfig,
        state_t=3,
    )
    control_video2world_model_config = override(control_video2world_model_config)

    # Add image context dimension to the config
    img_context_dim = 1152

    # Allow modifying the struct to add new fields
    OmegaConf.set_struct(control_video2world_model_config.net, False)
    control_video2world_model_config.net.extra_image_context_dim = img_context_dim
    OmegaConf.set_struct(control_video2world_model_config.net, True)

    return control_video2world_model_config


@pytest.mark.flaky(max_runs=3)
def test_control_video2world_model_init(control_video2world_config):
    vid2vid_model = ControlVideo2WorldModel(control_video2world_config).cuda()
    vid2vid_model.on_train_start()


@pytest.mark.flaky(max_runs=3)
def test_control_video2world_model_with_image_context_init(control_video2world_with_image_context_config):
    vid2vid_model = ControlVideo2WorldModel(control_video2world_with_image_context_config).cuda()
    vid2vid_model.on_train_start()


@pytest.fixture
def video_batch():
    batch_size = 1
    num_frame = 17
    resolution_h = 480
    resolution_w = 640
    # video batch
    data_batch = {
        "dataset_name": "video_data",
        "video": (torch.randn(batch_size, 3, num_frame, resolution_h, resolution_w) * 255).to(dtype=torch.uint8),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024, dtype=torch.float32),
        "fps": torch.randint(16, 32, (batch_size,)).float(),
        "padding_mask": repeat(
            torch.zeros(size=(1, resolution_h, resolution_w)),
            "... -> b ...",
            b=batch_size,
        ),
        "control_input_edge": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "control_input_vis": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "control_input_depth": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "control_input_seg": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
    }
    return data_batch


@pytest.fixture
def video_batch_with_image_context():
    batch_size = 1
    num_frame = 17
    resolution_h = 480
    resolution_w = 640
    # video batch
    data_batch = {
        "dataset_name": "video_data",
        "video": (torch.randn(batch_size, 3, num_frame, resolution_h, resolution_w) * 255).to(dtype=torch.uint8),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024, dtype=torch.float32),
        "fps": torch.randint(16, 32, (batch_size,)).float(),
        "padding_mask": repeat(
            torch.zeros(size=(1, resolution_h, resolution_w)),
            "... -> b ...",
            b=batch_size,
        ),
        "control_input_edge": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "control_input_vis": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "control_input_depth": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "control_input_seg": torch.randn(batch_size, 1, num_frame, resolution_h, resolution_w, dtype=torch.float32),
        "image_context": torch.randn(batch_size, 3, 1, resolution_h, resolution_w, dtype=torch.float32),
    }
    return data_batch


"""
Usage:
    pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_test.py -k test_control_video2world_model_training_step
"""


@pytest.mark.flaky(max_runs=3)
@pytest.mark.L0
def test_control_video2world_model_training_step(control_video2world_config, video_batch):
    model = ControlVideo2WorldModel(control_video2world_config).cuda()
    model.on_train_start()

    # video batch
    for k, v in video_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        video_batch[k] = _v
    video_output_batch, video_loss = model.training_step(video_batch, 2)
    video_loss.backward()


"""
Usage:
    pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_test.py -k test_control_video2world_model_training_step_with_image_context
"""


@pytest.mark.flaky(max_runs=3)
def test_control_video2world_model_training_step_with_image_context(
    control_video2world_with_image_context_config, video_batch_with_image_context
):
    model = ControlVideo2WorldModel(control_video2world_with_image_context_config).cuda()
    model.on_train_start()

    # video batch
    for k, v in video_batch_with_image_context.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        video_batch_with_image_context[k] = _v
    video_output_batch, video_loss = model.training_step(video_batch_with_image_context, 2)
    video_loss.backward()


"""
Usage:
    pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_test.py -k test_control_video2world_model_sampling
"""


@pytest.mark.L0
@pytest.mark.flaky(max_runs=3)
def test_control_video2world_model_sampling(control_video2world_config, video_batch):
    model = ControlVideo2WorldModel(control_video2world_config).cuda()
    model.on_train_start()

    # video batch sampling
    for k, v in video_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        video_batch[k] = _v
    sample = model.generate_samples_from_batch(video_batch)
    sample = model.decode(sample)
    expected_shape = video_batch[model.input_data_key].shape
    assert sample.shape == expected_shape, f"Expected shape: {expected_shape}, got: {sample.shape}"


"""
Usage:
    pytest -s cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_test.py -k test_control_video2world_model_sampling_with_image_context
"""


@pytest.mark.flaky(max_runs=3)
def test_control_video2world_model_sampling_with_image_context(
    control_video2world_with_image_context_config, video_batch_with_image_context
):
    model = ControlVideo2WorldModel(control_video2world_with_image_context_config).cuda()
    model.on_train_start()

    # video batch sampling
    for k, v in video_batch_with_image_context.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        video_batch_with_image_context[k] = _v
    sample = model.generate_samples_from_batch(video_batch_with_image_context)
    sample = model.decode(sample)
    expected_shape = video_batch_with_image_context[model.input_data_key].shape
    assert sample.shape == expected_shape, f"Expected shape: {expected_shape}, got: {sample.shape}"
