#!/usr/bin/env python
# coding=utf-8
"""
特点：
	•	不训练、不反传、不 optimizer、不 EMA
	•	只 forward + loss
	•	最大程度复用你现有代码与配置
	•	RDT 原生 loss，不自己造轮子
"""


import os
import math
import yaml
import torch
from tqdm import tqdm
from accelerate import Accelerator
from pathlib import Path

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset import VLAConsumerDataset, DataCollatorForVLAConsumerDataset


def main(args):
    # -------------------------
    # Load configs
    # -------------------------
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # -------------------------
    # Encoders
    # -------------------------
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=device,
        )
        tokenizer = text_embedder.tokenizer
        text_encoder = text_embedder.model.to(device, dtype=weight_dtype).eval()

    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path,
        args=None,
    )
    vision_encoder.vision_tower.to(device, dtype=weight_dtype).eval()
    image_processor = vision_encoder.image_processor

    # -------------------------
    # RDT model
    # -------------------------
    img_cond_len = (
        config["common"]["img_history_size"]
        * config["common"]["num_cameras"]
        * vision_encoder.num_patches
    )

    rdt = RDTRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=img_cond_len,
        img_pos_embed_config=[
            ("image", (
                config["common"]["img_history_size"],
                config["common"]["num_cameras"],
                -vision_encoder.num_patches,
            )),
        ],
        lang_pos_embed_config=[
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        dtype=weight_dtype,
    ).to(device)

    if args.checkpoint_path is not None:
        if os.path.isdir(args.checkpoint_path):
            # HuggingFace-style pretrained model
            rdt = RDTRunner.from_pretrained(args.checkpoint_path).to(device, dtype=weight_dtype)
        else:
            # training checkpoint (.pt)
            ckpt = torch.load(args.checkpoint_path, map_location="cpu")
            rdt.load_state_dict(ckpt["module"] if "module" in ckpt else ckpt)
    rdt.eval()

    # -------------------------
    # Dataset / Dataloader
    # -------------------------
    dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForVLAConsumerDataset(tokenizer),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataloader = accelerator.prepare(dataloader)

    # -------------------------
    # Loss evaluation
    # -------------------------
    loss_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating loss"):
            images = batch["images"].to(device, dtype=weight_dtype)
            states = batch["states"].to(device, dtype=weight_dtype)[:, -1:, :]
            actions = batch["actions"].to(device, dtype=weight_dtype)
            state_elem_mask = batch["state_elem_mask"].to(device, dtype=weight_dtype)
            ctrl_freqs = batch["ctrl_freqs"]

            B, _, C, H, W = images.shape
            img_embeds = vision_encoder(
                images.reshape(-1, C, H, W)
            ).reshape(B, -1, vision_encoder.hidden_size)

            lang_attn_mask = batch["lang_attn_mask"]
            if args.precomp_lang_embed:
                text_embeds = batch["lang_embeds"].to(device, dtype=weight_dtype)
            else:
                text_embeds = text_encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=lang_attn_mask,
                )["last_hidden_state"]

            loss = rdt(
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_embeds,
                state_tokens=states,
                action_gt=actions,
                action_mask=state_elem_mask.unsqueeze(1),
                ctrl_freqs=ctrl_freqs,
            )

            loss_list.append(loss.item())

    mean_loss = sum(loss_list) / len(loss_list)
    print(f"\nMean loss: {mean_loss:.6f}")
    print(f"Num batches: {len(loss_list)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--precomp_lang_embed", action="store_true")
    parser.add_argument("--load_from_hdf5", action="store_true")

    args = parser.parse_args()
    main(args)