#!/usr/bin/env python
# coding=utf-8
"""
Loss evaluation over multiple tasks.
- Load model & encoders ONCE
- Switch datasets per task
- Forward + loss only
"""
import json
import os
import yaml
import torch
from tqdm import tqdm
from accelerate import Accelerator

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset import VLAConsumerDataset, DataCollatorForVLAConsumerDataset


@torch.no_grad()
def eval_one_dataset(
    dataloader,
    rdt,
    vision_encoder,
    text_encoder,
    device,
    weight_dtype,
    precomp_lang_embed,
):
    loss_list = []

    for batch in tqdm(dataloader, desc="Evaluating loss", leave=False):
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
        if precomp_lang_embed:
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
    return mean_loss, len(loss_list)


def main(args):
    # -------------------------
    # Load base config (ONCE)
    # -------------------------
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            results = json.load(f)
    else:
        results = {}
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # -------------------------
    # Encoders (ONCE)
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
    # RDT model (ONCE)
    # -------------------------
    img_cond_len = (
        config["common"]["img_history_size"]
        * config["common"]["num_cameras"]
        * vision_encoder.num_patches
    )

    if os.path.isdir(args.checkpoint_path):
        rdt = RDTRunner.from_pretrained(args.checkpoint_path)
    else:
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
        )
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        rdt.load_state_dict(ckpt["module"] if "module" in ckpt else ckpt)

    rdt.to(device, dtype=weight_dtype).eval()

    # -------------------------
    # Loop over tasks
    # -------------------------
    for model_config_path in args.model_config_paths:
        print(f"\n==============================")
        print(f"🔍 Evaluating task: {os.path.basename(model_config_path)}")
        print(f"==============================")

        dataset = VLAConsumerDataset(
            model_config_path=model_config_path,
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

        mean_loss, num_batches = eval_one_dataset(
            dataloader,
            rdt,
            vision_encoder,
            text_encoder,
            device,
            weight_dtype,
            args.precomp_lang_embed,
        )

        task_name = os.path.splitext(os.path.basename(model_config_path))[0]

        results[task_name] = {
            "mean_loss": mean_loss,
            "num_batches": num_batches,
            "num_samples": len(dataset),
        }

        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Mean loss: {mean_loss:.6f}")
        print(f"Num batches: {num_batches}")
        print(f"[Saved] {task_name} -> {args.output_json}")

        # 清理，防止 HDF5 / worker 堆积
        del dataloader, dataset
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--model_config_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of model config yaml paths (one per task)",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="finetune")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--precomp_lang_embed", action="store_true")
    parser.add_argument("--load_from_hdf5", action="store_true")
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output json file for loss results",
    )
    args = parser.parse_args()
    main(args)