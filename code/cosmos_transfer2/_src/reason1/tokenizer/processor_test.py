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

"""
Usage:
pytest projects/cosmos/reasoning/v1/tokenizer/processor_test.py --L1
"""

import pytest
import torch
from PIL import Image

from cosmos_transfer2._src.reason1.datasets.debug_data_qwen import create_debug_input


@pytest.mark.skip(reason="need qwen libraries")
def test_assistant_tokens_mask_qwen_processor():
    """
    Usage:
    pytest  -s projects/cosmos/reasoning/v1/tokenizer/processor_test.py::test_assistant_tokens_mask_qwen_processor --L1
    """
    from cosmos_transfer2._src.reason1.tokenizer.processor import Processor

    # Initialize tokenizer
    tokenizer = Processor(name="Qwen/Qwen2.5-VL-3B-Instruct")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What's 11+11?"},
        {"role": "assistant", "content": "the value is 22"},
        {"role": "user", "content": "What's 34-11?"},
        {"role": "assistant", "content": "it is 23"},
    ]
    expected_mask = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )["input_ids"]
    print("input_ids", input_ids, input_ids.shape)
    # Generate mask
    mask = tokenizer.add_assistant_tokens_mask(input_ids)

    # Verify mask properties
    assert isinstance(mask, torch.Tensor), "Mask should be a tensor"
    assert mask.shape == input_ids.shape, "Mask shape should match input_ids shape"
    assert mask.dtype == torch.bool, "Mask should be boolean type"

    # Convert to list for comparison
    mask_list = mask.squeeze().tolist()
    print(f"Generated Mask: {mask_list}")

    # Decode the input_ids
    decoded_text_all = tokenizer.decode([i for i in input_ids])
    print(f"Decoded all IDs: {decoded_text_all}")

    decoded_text_masked = tokenizer.decode([input_ids[i] for i in range(len(input_ids)) if mask[i]])
    print(f"Decoded masked IDs: {decoded_text_masked}")
    decoded_text_masked_expected = """the value is 22<|im_end|>it is 23<|im_end|>"""
    assert decoded_text_masked == decoded_text_masked_expected
    decoded_text_unmasked = tokenizer.decode([input_ids[i] for i in range(len(input_ids)) if not mask[i]])
    print(f"Decoded unmasked IDs: {decoded_text_unmasked}")

    # # Verify mask contents
    assert len(mask_list) == len(expected_mask), "Mask length mismatch"
    assert sum(mask_list) == sum(expected_mask), "Number of assistant tokens mismatch"
    assert mask_list == expected_mask, "Mask pattern does not match expected"


@pytest.mark.skip(reason="need qwen libraries")
def test_assistant_tokens_mask_qwen_processor_with_image():
    """
    Usage:
    pytest  -s projects/cosmos/reasoning/v1/tokenizer/processor_test.py::test_assistant_tokens_mask_qwen_processor_with_image --L1
    """
    from cosmos_transfer2._src.reason1.tokenizer.processor import Processor

    # Initialize tokenizer
    tokenizer = Processor(name="Qwen/Qwen2.5-VL-3B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    output = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    input_ids = output["input_ids"]
    print("input_ids", input_ids, input_ids.shape)
    # Generate mask
    mask = tokenizer.add_assistant_tokens_mask(input_ids)
    print("mask", mask, mask.shape)


@pytest.mark.skip(reason="need qwen libraries")
def test_assistant_tokens_mask_qwen_processor_with_list_of_images():
    """
    Usage:
    pytest  -s projects/cosmos/reasoning/v1/tokenizer/processor_test.py::test_assistant_tokens_mask_qwen_processor_with_list_of_images --L1
    """
    from cosmos_transfer2._src.reason1.tokenizer.processor import Processor

    # Initialize tokenizer
    tokenizer = Processor(name="Qwen/Qwen2.5-VL-3B-Instruct")
    # create a PIL image
    image = Image.new("RGB", (100, 100), color="red")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    output = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    input_ids = output["input_ids"]
    print("input_ids", input_ids, input_ids.shape)
    # Generate mask
    mask = tokenizer.add_assistant_tokens_mask(input_ids)
    print("mask", mask, mask.shape)

    # Image token ids
    image_token_id = tokenizer.image_token_id
    print("image_token_id", image_token_id)
    num_image_token = (input_ids == image_token_id).sum()
    assert num_image_token == 32, "num_image_token for 2 images should be 32"
    print("num_image_token for 2 images", num_image_token)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    output = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    input_ids = output["input_ids"]
    num_image_token = (input_ids == image_token_id).sum()
    print("num_image_token for 1 images", num_image_token)
    assert num_image_token == 16, "num_image_token for 1 images should be 16"


@pytest.mark.skip(reason="need qwen libraries")
def test_create_dummy_input():
    """
    Usage:
    pytest  -s projects/cosmos/reasoning/v1/tokenizer/processor_test.py::test_create_dummy_input --L1
    """
    from cosmos_transfer2._src.reason1.tokenizer.processor import Processor

    tokenizer = Processor(name="Qwen/Qwen2.5-VL-3B-Instruct")
    input_dict = create_debug_input(tokenizer, seq_len=8000, padding=True, num_images=1)
    print(input_dict.keys())
    print(input_dict["tokens"].shape)
    print(input_dict["attention_mask"])
