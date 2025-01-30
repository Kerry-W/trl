# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:

python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0

With LoRA:
python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-6 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --use_peft
"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    BasePairwiseJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from torch.utils.data import Dataset
import json
import os
import re

class DpoDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "prompt": [self.data[idx]["messages"][0]],
            "completion": [self.data[idx]["messages"][1]],
        }

class GSM8kJudge(BasePairwiseJudge):
    def judge(self, prompts: list[str], completions: list[list[str]], answers: list[str], shuffle_order: bool = False) -> list[int]:
        ranks = []
        for completion, answer in zip(completions, answers):
            answer = answer[0]["content"].split("####")[-1].strip()
            pred0 = completion[0].split("####")[-1].strip()
            pred1 = completion[1].split("####")[-1].strip()
            if pred0 == answer and pred1 == answer:
                ranks.append(1 if len(completion[0]) > len(completion[1]) else 0)
            elif pred0 == answer:
                ranks.append(0)
            elif pred1 == answer:
                ranks.append(1)
            else:
                ranks.append(-1)
        return ranks

class MATHJudge(BasePairwiseJudge):
    def judge(self, prompts: list[str], completions: list[list[str]], answers: list[str], shuffle_order: bool = False) -> list[int]:
        ranks = []
        pattern = r'boxed\{(.*)\}'
        for completion, answer in zip(completions, answers):
            answer = re.search(pattern, answer[0]["content"]).group(1)
            match = re.search(pattern, completion[0])
            if match:
                pred0 = match.group(1)
            else:
                pred0 = ""
            match = re.search(pattern, completion[1])
            if match:
                pred1 = match.group(1)
            else:
                pred1 = ""
            if pred0 == answer and pred1 == answer:
                ranks.append(1 if len(completion[0]) > len(completion[1]) else 0)
            elif pred0 == answer:
                ranks.append(0)
            elif pred1 == answer:
                ranks.append(1)
            else:
                ranks.append(-1)
        return ranks

JUDGES = {"GSM8kJudge": GSM8kJudge, "MATHJudge": MATHJudge}

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    if training_args.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            num_labels=1,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
            truncation=True,
            truncation_side="left",  # since we judge the completion, truncating left is more appropriate
        )
    else:
        reward_model = None
        reward_tokenizer = None

    if training_args.judge is not None:
        judge_cls = JUDGES[training_args.judge]
        judge = judge_cls()
    else:
        judge = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        judge=judge,
        args=training_args,
        train_dataset=DpoDataset(os.path.join(script_args.dataset_name, "train_test.jsonl")),
        eval_dataset=DpoDataset(os.path.join(script_args.dataset_name, "test.jsonl")),
        processing_class=tokenizer,
        reward_processing_class=reward_tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)

"""
ScriptArguments(
    dataset_name='../data/gsm8k',
    dataset_config=None,
    dataset_train_split='train',
    dataset_test_split='test',
    gradient_checkpointing_use_reentrant=False,
    ignore_bias_buffers=False
)

OnlineDPOConfig(
    _n_gpu=1,
    accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    average_tokens_across_devices=False,
    batch_eval_metrics=False,
    beta=0.1,
    bf16=False,
    bf16_full_eval=False,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=None,
    dataset_num_proc=None,
    ddp_backend=None,
    ddp_broadcast_buffers=None,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed=None,
    disable_dropout=True,
    disable_tqdm=False,
    dispatch_batches=None,
    do_eval=False,
    do_predict=False,
    do_train=False,
    ds3_gather_for_generation=True,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_do_concat_batches=True,
    eval_on_start=False,
    eval_steps=None,
    eval_strategy=no,
    eval_use_gather_object=False,
    evaluation_strategy=None,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs=None,
    greater_is_better=None,
    group_by_length=False,
    half_precision_backend=auto,
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=None,
    hub_strategy=every_save,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_for_metrics=[],
    include_inputs_for_metrics=False,
    include_num_input_tokens_seen=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    judge=GSM8kJudge,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=5e-07,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=0,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=../result/gsm8k_dpo/runs/Jan27_15-45-38_n124-253-169,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=500,
    logging_strategy=steps,
    loss_type=sigmoid,
    lr_scheduler_kwargs={},
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_length=512,
    max_new_tokens=64,
    max_steps=5000,
    metric_for_best_model=None,
    missing_eos_penalty=None,
    mp_parameters=,
    neftune_noise_alpha=None,
    no_cuda=False,
    num_train_epochs=3.0,
    optim=adamw_torch,
    optim_args=None,
    optim_target_modules=None,
    output_dir=../result/gsm8k_dpo,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=[],
    restore_callback_states_from_checkpoint=False,
    resume_from_checkpoint=None,
    reward_model_path=None,
    run_name=../result/gsm8k_dpo,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy=steps,
    save_total_limit=None,
    seed=42,
    skip_memory_metrics=True,
    split_batches=None,
    temperature=0.9,
    tf32=None,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torch_empty_cache_steps=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_liger_kernel=False,
    use_mps_device=False,
    use_vllm=False,
    warmup_ratio=0.1,
    warmup_steps=0,
    weight_decay=0.0,
)

ModelConfig(model_name_or_path='../result/gsm8k_sft',
    model_revision='main',
    torch_dtype=None,
    trust_remote_code=False,
    attn_implementation=None,
    use_peft=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None,
    lora_modules_to_save=None,
    lora_task_type='CAUSAL_LM',
    use_rslora=False,
    load_in_8bit=False,
    load_in_4bit=False,
    bnb_4bit_quant_type='nf4',
    use_bnb_nested_quant=False
)
"""