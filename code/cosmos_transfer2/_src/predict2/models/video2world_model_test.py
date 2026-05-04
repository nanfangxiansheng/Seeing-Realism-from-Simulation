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
from flaky import flaky

from cosmos_transfer2._src.imaginaire.utils.config_helper import override
from cosmos_transfer2._src.predict2.configs.common.defaults.ema import PowerEMAConfig
from cosmos_transfer2._src.predict2.configs.common.defaults.tokenizer import DummyJointImageVideoConfig
from cosmos_transfer2._src.predict2.configs.video2world.defaults.conditioner import VideoPredictionConditioner
from cosmos_transfer2._src.predict2.configs.video2world.defaults.net import mini_net
from cosmos_transfer2._src.predict2.models.video2world_model import Video2WorldConfig, Video2WorldModel

"""
pytest -s projects/cosmos/diffusion/v2/models/video2world_model_test.py --all
"""


@pytest.fixture
def video2world_config():
    video2world_model_config = Video2WorldConfig(
        tokenizer=DummyJointImageVideoConfig,
        conditioner=VideoPredictionConditioner,
        net=mini_net,
        ema=PowerEMAConfig,
    )
    video2world_model_config = override(video2world_model_config)
    return video2world_model_config


@flaky(max_runs=3)
def test_video2world_model_init(video2world_config):
    video2world_model = Video2WorldModel(video2world_config).cuda()
    video2world_model.on_train_start()


@pytest.fixture
def image_batch():
    batch_size = 1
    num_frame = 17
    image_batch_size = batch_size * num_frame // 2
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size * num_frame // 2, 3, 1024, 1024, dtype=torch.float32),
        "t5_text_embeddings": torch.randn(image_batch_size, 512, 1024, dtype=torch.float32),
        "fps": torch.randint(16, 32, (image_batch_size,)).float(),
        "padding_mask": repeat(
            torch.zeros(size=(1, 1024, 1024)),
            "... -> b ...",
            b=image_batch_size,
        ),
    }
    return data_batch


@pytest.fixture
def video_batch():
    batch_size = 1
    num_frame = 17
    # video batch
    data_batch = {
        "dataset_name": "video_data",
        "video": (torch.randn(batch_size, 3, num_frame, 1024, 1024) * 255).to(dtype=torch.uint8),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024, dtype=torch.float32),
        "fps": torch.randint(16, 32, (batch_size,)).float(),
        "padding_mask": repeat(
            torch.zeros(size=(1, 1024, 1024)),
            "... -> b ...",
            b=batch_size,
        ),
    }
    return data_batch


"""
Usage:
    pytest -s projects/cosmos/diffusion/v2/models/video2world_model_test.py -k test_video2world_model_training_step
"""


@flaky(max_runs=3)
def test_video2world_model_training_step(video2world_config, video_batch, image_batch):
    model = Video2WorldModel(video2world_config).cuda()
    model.on_train_start()

    # image batch
    for k, v in image_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        image_batch[k] = _v
    image_output_batch, image_loss = model.training_step(image_batch, 1)
    image_loss.backward()

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
    pytest -s projects/cosmos/diffusion/v2/models/video2world_model_test.py -k test_video2world_model_sampling
"""


@flaky(max_runs=3)
def test_video2world_model_sampling(video2world_config, video_batch, image_batch):
    model = Video2WorldModel(video2world_config).cuda()
    model.on_train_start()

    # image batch sampling
    for k, v in image_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        image_batch[k] = _v
    sample = model.generate_samples_from_batch(image_batch)
    sample = model.decode(sample)
    expected_shape = image_batch[model.input_image_key].shape
    assert sample.shape == expected_shape, f"Expected shape: {expected_shape}, got: {sample.shape}"

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
