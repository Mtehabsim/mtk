# MTK: Defending LLMs Against Jailbreak Attacks via Manifold Trajectory Kinetics

Official implementation of our USENIX Security 2026 paper:
**Defending Jailbreak Attacks on Large Language Models via Manifold Trajectory Kinetics (MTK)**.

<img width="2302" height="952" alt="1~)(@9872}TIZ~37KZME4GS" src="https://github.com/user-attachments/assets/67b6c47a-2d11-4257-b667-b59007044edd" />


## 📋 Environment Requirements

### Basic Environment

- Python 3.8+ (3.9/3.10 recommended for PyTorch compatibility)

- CUDA 11.7+ (recommended for GPU acceleration; CPU is supported but slower)

- GPU Memory: ≥16GB for LLaVA-1.6-Vicuna-7B, ≥12GB for Qwen-VL-Chat

### Dependency Installation

```bash
# Clone repository
git clone https://github.com/Rookie143/mtk.git
cd mtk/vlm

# Create and activate the VLM mtk environment
conda create -n vlm_mtk python=3.10 -y
conda activate vlm_mtk

# Install vlm core dependencies
pip install -r requirements.txt

# Exit the VLM virtual environment 
conda deactivate

# Create and activate the LLM mtk environment 
conda create -n llm_mtk python=3.10 -y 
conda activate llm_mtk

#Install llm core dependencies
cd ../llm/llm_mtk
pip install -r requirements.txt
```
> Note: Activate the corresponding virtual environment before running the VLM or LLM scripts:
>
> ```bash
> # For VLM experiments
> conda activate vlm_mtk
>
> # For LLM experiments
> conda activate llm_mtk
> ```

## 🚀 VLM_Quick Start

### 1. Dataset Preparation

Place datasets in the specified paths (modify paths in `load_datasets.py`):

|Dataset|Path Example|Purpose|
|---|---|---|
|[VQA](https://visualqa.org/download.html)|./datasets/vqa/test2015|Benign samples (training)|
|[MM-Vet v2](https://github.com/yuweihao/MM-Vet)|./datasets/mm-vet-v2|Benign samples (testing)|
|[SD-AdvBench]|./datasets/sd_advbench|Malicious samples (training)|
|[MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench)|./datasets/MM-SafetyBench|Malicious samples (testing)|
|[FigStep](https://github.com/CryptoAILab/FigStep/tree/main/data/images/SafeBench)|./datasets/FigStep|Malicious samples (testing)|
|[JailBreakV_28K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)|./datasets/JailBreakV_28K|Malicious samples (testing)|
|[USB-Overrefusal](https://huggingface.co/datasets/cgjacklin/USB/tree/main)|./datasets/MM-SafetyBench|Benign samples (testing)|

>You can use download_vlm_dataset.sh to quickly download these datasets.

The paths of the datasets should be organized as follows:


```text
datasets/
└── datasets/
    ├── vqa/
    │   ├── OpenEnded_mscoco_test2015_questions.json
    │   └── test2015/
    │
    ├── usb/
    │   ├── overfuse_data.csv
    │   └── img/
    │        └──overrefusal_eval
    │        └──vulnerability_eval
    │ 
    │
    ├── mm-vet-v2/
    │   ├── mm-vet-v2.json
    │   └── non_palette_images/
    │   └── images/    
    │
    ├── sd_advbench/
    │   ├── prompt_img_map.csv
    │   └── outputs_new/
    │
    ├── MM-SafetyBench/
    │   ├── image/
    │   └── data/
    │
    ├── FigStep/
    │   └── data/
    │       └── images/
    │       └── question/
    │
    └── JailBreakV_28K/
        ├── JailBreakV_28K.csv
        └── mini_JailBreakV_28K.csv
        └── RedTeam_2K.csv
        └── figstep/
        └── llm_transfer_attack/
        └── query_related/
```

### 2. Model Weights Preparation

Activate the VLM environment and enter the VLM directory:

```bash
conda activate vlm_mtk
cd mtk/vlm
```

Install the Hugging Face command-line tool:

```bash
pip install -U huggingface_hub
```

Download the multimodal model weights from Hugging Face and save them to the paths used by the evaluation scripts:

```bash

# Download LLaVA-1.6-Vicuna-7B
hf download llava-hf/llava-v1.6-vicuna-7b-hf \
    --local-dir ./models/llava-v1.6-vicuna-7b-hf

# Download Qwen-VL-Chat
hf download Qwen/Qwen-VL-Chat \
    --local-dir ./models/qwen_vl_chat
```

The downloaded models should be stored at the following locations:

| Model               | Hugging Face Repository                                                                     | Local Path                         |
| ------------------- | ------------------------------------------------------------------------------------------- |------------------------------------|
| LLaVA-1.6-Vicuna-7B | [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | `./models/llava-v1.6-vicuna-7b-hf` |
| Qwen-VL-Chat        | [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)                               | `./models/qwen_vl_chat`            |

If the model paths in the test scripts differ from the paths above, modify the corresponding `from_pretrained` arguments before running the evaluation.


### 3. Run Detection

```bash
# Run jailbreak detection evaluation for LLaVA
python test_AUROC_llava.py 

# Run jailbreak detection evaluation for Qwen-VL
python test_AUROC_qwen.py
```

## 🚀 LLM_Quick Start

### 1. Training Datasets Preparation

Place **training datasets** in ./datasets/train_data:

|Dataset|Path Example|Purpose|
|---|---|---|
|[Alpaca ](https://huggingface.co/datasets/tatsu-lab/alpaca)|./datasets/train_data|Benign samples|
|[Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)|./datasets/train_data|Benign samples|
|[Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)|./datasets/train_data|Benign samples|
|[Or-Bench_80k](https://huggingface.co/datasets/bench-llm/or-bench)|./datasets/train_data|Pseudo-Malicious samples|
|[MaliciousInstruct](https://huggingface.co/datasets/walledai/MaliciousInstruct)|./datasets/train_data|Malicious samples|
|[Advbench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)|./datasets/train_data|Malicious samples|
|[PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)|./datasets/train_data|Malicious samples|



> ⚠️ Note: The required training datasets have already been processed and included in the `./datasets/train_data` directory. No additional dataset conversion or preparation is required.

----------

### 2. Test Datasets Preparation

Place **test datasets** under:

`./datasets/{model_name}_test`

where `{model_name}` must be one of:
-   `llama2`
-   `llama3`
-   `mistral`
-   `vicuna`
    

#### File Naming Convention

-   **Jailbreak / malicious attack samples**:
    
    `{attack_name}_1.json`
    
-   **Benign samples**:
    
    `{benign_name}_0.json`
    

Here, the suffix `_1` indicates malicious/jailbreak data, and `_0` indicates benign data.  
This naming convention is required for correct label parsing during evaluation.

### 3. Model Weights Preparation

Activate the LLM environment and enter the LLM directory:

```bash
conda activate llm_mtk
cd mtk/llm/llm_mtk
```

Install the Hugging Face command-line tool:

```bash
pip install -U huggingface_hub
```

Download the required model weights:

```bash
# Download Llama2-7b-chat-hf
hf download meta-llama/Llama-2-7b-chat-hf \
    --local-dir ./model/Llama2

# Download Llama3-8b-Instruct
hf download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir ./model/llama3

# Download Mistral-7b-instruct-v0.2
hf download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ./model/mistral_7b

# Download Vicuna-7b-v1.5
hf download lmsys/vicuna-7b-v1.5 \
    --local-dir ./model/vicuna-7b-v1_5
```

After downloading, the model weights should be organized as follows:

```text
mtk/
└── llm/
    └── llm_mtk/
        └── model/
            ├── Llama2/
            ├── llama3/
            ├── mistral_7b/
            └── vicuna-7b-v1_5/
```

Each model is stored in its own subdirectory following the Hugging Face standard structure. If the model paths configured in the evaluation scripts differ from the paths above, modify the corresponding local paths before running the scripts.



### 4. Run Detection

```bash
# Run jailbreak detection evaluation for llama2
python mtk_llama2.py

# Run jailbreak detection evaluation for llama3
python mtk_llama3.py

# Run jailbreak detection evaluation for mitstral
python mtk_mistral.py

# Run jailbreak detection evaluation for vicuna
python mtk_vicuna.py
```

