Note: this is the code collection of paper [ICML 2026 accepted as regular]:

**[Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](https://arxiv.org/abs/2605.02757)**

The framework is as follows:

<img width="1621" height="721" alt="图片" src="https://github.com/user-attachments/assets/64fa70e8-a506-4614-8522-5a84b8db7a7c" />

The poster on xiaohongshu platform is as follows:

- 71 【ICML2026：低成本破解VLA的Sim-to-Real难题 - 爱学人工智能 | 小红书 - 你的生活兴趣社区】 😆 L2Bw0TnkqAC2BjS 😆 https://www.xiaohongshu.com/discovery/item/69f9af07000000003501c545?source=webshare&xhsshare=pc_web&xsec_token=CBh2WwN_bAasqgtLRVlQPMWA6Toz_pOdPnm4cBrXdXi6k=&xsec_source=pc_share

# Project Setup and Usage for coreselection

## Environment

```bash
# Make sure python version == 3.10
conda activate Corset_Sampling
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
pip install packaging==24.0
pip install ninja
ninja --version; echo $?
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt
```

This guide outlines the three-stage pipeline: generating video embeddings, computing task difficulty, and performing coreset pruning.

## 1. Cosmos Video Embedding

To generate video embeddings using the Cosmos model, follow these steps:

### 1.1 Download Weights

Download the `nvidia/Cosmos-Embed1-448p` model weights and place them into the `cosmos_embedding/model_weight/` directory.

### 1.2 Run Inference

Refer to the arguments in `cosmos_embedding/gen_embedding.py` to run the inference script.

**Directory Structure:**

```text
cosmos_embedding/
├── model_weight/       # Place nvidia/Cosmos-Embed1-448p files here
└── gen_embedding.py
```

---

## 2. Difficulty Computation (RDT)

This step computes the difficulty score (loss) for the dataset using the RDT model.

### 2.1 Preparation

1.  **Download Weights:**
    Download the following models and place them in `difficuty_compute/RDT/weights/`:
    *   `google/t5-v1_1-xxl`
    *   `google/siglip-so400m-patch14-384`
    *   `rdt-1b` (Robotics Diffusion Transformer)

2.  **Configuration:**
    Ensure that the `model_config` and `training_data` directories located in `difficuty_compute/RDT/` are set up strictly following the requirements of the **RoboTwin 2.0** repository.

**Directory Structure:**

```text
difficuty_compute/
└── RDT/
    ├── weights/
    │   ├── t5-v1_1-xxl/
    │   ├── siglip-so400m-patch14-384/
    │   └── rdt-1b/
    ├── model_config/   # Must follow RoboTwin 2.0 specs
    └── training_data/  # Must follow RoboTwin 2.0 specs
```

### 2.2 Execution

Navigate to the RDT directory and run the evaluation script:

```bash
cd difficuty_compute/RDT
bash run_loss_eval_all_tasks.sh
```

**Output:** This will generate the file `difficuty_compute/RDT/rdt_1b_finetune_loss.json`.

---

## 3. Coreset Pruning (D2Pruning)

The final step merges the embeddings and difficulty scores to sample the coreset.

### 3.1 Data Merging

Before running the sampling script, ensure you merge the outputs from Step 1 and Step 2.

*   **Source 1:** `cosmos_embedding/video_embeddings.json`
*   **Source 2:** `difficuty_compute/RDT/rdt_1b_finetune_loss.json`
*   **Target:** `d2pruning/core/data/merged_dataset_input.json`

### 3.2 Run Sampling

Navigate to the `d2pruning` directory and execute the sampling script:

```bash
cd d2pruning

python ./core/data/coreset_sampling.py \
  --input_path "./core/data/merged_dataset_input.json" \
  --output_path "./core/data/coreset.json" \
  --subset_size 2000 \
  --stratas 50 \
  --budget_mode confidence
```

**Arguments:**

*   `--subset_size`: The number of samples to select (e.g., 2000).
*   `--stratas`: Number of difficulty intervals for stratified sampling.
*   `--budget_mode`: Allocation strategy (`confidence` assigns more budget to harder samples).

# Project Setup and Usage for efficient transfer

## Environmental setup 

- First: follow the setup.md in docs from Nvidia Cosmos.

## Preprocess
We provide examples code in affixilary_real and affixilary_robotwin, note the directory needs to be set before processing:

- First: use sample.py to get and list videos

- Second: use reset_video.py to reset videos' pixels

## Run 
use the file  depth_video_loop.sh in shell_script to run the full process. It is found that using an A800 80GB can perform the following correctly.

```bash
bash ./shell_script/depth_video_loop.sh --begin 74 --end 90 --gpu_a 7  --master_port 12305 --out_dir out_sampled
```

## Examples

We provide two generated examples in directory examples_generated.

Additionally, we also provide records from real world experiments in the folder:  real_experiment_records
