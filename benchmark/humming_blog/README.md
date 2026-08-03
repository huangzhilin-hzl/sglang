# Humming SGLang Blog Reproduction

This guide reproduces Figures 2–7 and the accuracy checks reported in [the Humming SGLang blog pull request](https://github.com/lm-sys/lm-sys.github.io/pull/380). It covers Kimi-K2.6, DeepSeek-V4-Flash, and GLM-5.2 W4AFP8. Raw benchmark outputs and plotting data are generated outside the source tree and are intentionally not committed.

## Common Environment and Protocol

| Component | Version or specification |
| --- | --- |
| GPU | 8 × NVIDIA H20-3e |
| NVIDIA Driver | `570.133.20` |
| CUDA | `13.0` |
| SGLang | Commit [`d6ef68881e263812d4901f632786015005c4d050`](https://github.com/sgl-project/sglang/commit/d6ef68881e263812d4901f632786015005c4d050) |
| Humming | `humming-kernels[cu13]==0.1.11` |
| EvalScope | Commit [`acd09b44384d53174768bb1063f675420f76fae9`](https://github.com/modelscope/evalscope/commit/acd09b44384d53174768bb1063f675420f76fae9) |

### Build and Verify the Agentic Replay Data

The replay data is derived from [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) at revision `35455389ab51bf5e2306bfd436ef72d0f98bf882` and padded with [`nvidia/OpenScienceReasoning-2`](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) at revision `174b02c9cdf231f220765b2a1d5ece4550921894`. The pinned builder is public at [`Jiminator/sglang@2bac7e1`](https://github.com/Jiminator/sglang/blob/2bac7e166a7b5bf518b778817ec464cec0f75e3e/benchmark/glm_nvfp4_blog/build_openhands_padded_dataset.py).

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/Jiminator/sglang.git /path/to/glm-nvfp4-repro
git -C /path/to/glm-nvfp4-repro sparse-checkout init --cone
git -C /path/to/glm-nvfp4-repro sparse-checkout set benchmark/glm_nvfp4_blog
git -C /path/to/glm-nvfp4-repro checkout 2bac7e166a7b5bf518b778817ec464cec0f75e3e

python3 /path/to/glm-nvfp4-repro/benchmark/glm_nvfp4_blog/build_openhands_padded_dataset.py \
  --model /path/to/models/Kimi-K2.6 \
  --pad-source openscience \
  --first-turn-length 74160 \
  --subsequent-turn-length 753 \
  --num-turns 13 \
  --number 128 \
  --output-path /path/to/data/openhands-kimi26.json

python3 /path/to/glm-nvfp4-repro/benchmark/glm_nvfp4_blog/build_openhands_padded_dataset.py \
  --model /path/to/models/GLM-5.2-W4AFP8 \
  --pad-source openscience \
  --first-turn-length 74160 \
  --subsequent-turn-length 753 \
  --num-turns 13 \
  --number 128 \
  --output-path /path/to/data/openhands-glm52-w4afp8.json

shasum -a 256 \
  /path/to/data/openhands-kimi26.json \
  /path/to/data/openhands-glm52-w4afp8.json
```

Expected dataset hashes are:

| Dataset | Source and use | SHA256 |
| --- | --- | --- |
| `openhands-kimi26.json` | Builder output using the pinned Kimi tokenizer; Figure 5 | `0a5842ae03d3a216ecf7d355f40fbe5c1bbaf8bd8904b87951a65d4057e3ca51` |
| `openhands-glm52-w4afp8.json` | Builder output using the pinned GLM tokenizer; reused for Figures 6 and 7 | `8cc10321e9a8628218dfa1c31eab6d05ef0adacab46f282f4d34899848787215` |

Stop if either checksum differs. A mismatch means the input corpus, tokenizer, builder, or generation parameters are not identical to the published workload.

The agentic workload replays 13-turn OpenHands coding conversations. A conversation starts with approximately 75K–80K input tokens, later turns add about 753 tokens, and every turn generates 220 tokens. Each profile is evaluated at concurrency 1, 2, 4, and 8 for three independent rounds. Start a fresh server before every profile and round, then run one warmup request before collecting formal results.

## Kimi-K2.6

### Server Launch

Set the checkpoint path and common server arguments:

```bash
# Hugging Face model ID: moonshotai/Kimi-K2.6
# Hugging Face EAGLE3 draft model ID: nvidia/Kimi-K2.6-Eagle3
export MODEL_PATH=/path/to/models/Kimi-K2.6
export DRAFT_MODEL_PATH=/path/to/models/Kimi-K2.6-Eagle3
export PORT=8188
export SGLANG_ENABLE_SPEC_V2=1

KIMI_SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --served-model-name moonshotai/Kimi-K2.6
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tp-size 8
  --pp-size 1
  --enable-cache-report
  --enable-metrics
  --log-level info
  --max-running-requests 48
  --cuda-graph-max-bs-decode 16
  --disable-prefill-cuda-graph
  --mem-fraction-static 0.8
  --context-length 90000
  --chunked-prefill-size 8192
  --attention-backend flashmla
  --kv-cache-dtype fp8_e4m3
  --reasoning-parser kimi_k2
  --tool-call-parser kimi_k2
  --mm-attention-backend fa3
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --speculative-draft-attention-backend fa3
)
```

Launch exactly one profile at a time.

#### Marlin WINT4A16

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
python3 -m sglang.launch_server "${KIMI_SERVER_ARGS[@]}"
```

#### Humming WINT4A16

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
python3 -m sglang.launch_server \
  "${KIMI_SERVER_ARGS[@]}" \
  --quantization humming \
  --moe-runner-backend humming
```

#### Humming WINT4AFP8

```bash
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'
python3 -m sglang.launch_server \
  "${KIMI_SERVER_ARGS[@]}" \
  --quantization humming \
  --moe-runner-backend humming
```

### Figures 2 and 3: Text and Image Latency

The latency figures use the same EAGLE3 3/1/4 configuration and a separate launch block for their latency-specific context and memory settings. Set `MODE=text` for Figure 2 or `MODE=image` for Figure 3, select one `PROFILE`, and run the launch block in a dedicated terminal. Valid profiles are `marlin-wint4a16`, `humming-wint4a16`, and `humming-wint4afp8`.

```bash
# Hugging Face model ID: moonshotai/Kimi-K2.6
# Hugging Face EAGLE3 draft model ID: nvidia/Kimi-K2.6-Eagle3
export MODEL_PATH=/path/to/models/Kimi-K2.6
export DRAFT_MODEL_PATH=/path/to/models/Kimi-K2.6-Eagle3
export PORT=8188
export MODE=text
export PROFILE=humming-wint4afp8
export SGLANG_ENABLE_SPEC_V2=1

KIMI_LATENCY_ARGS=(
  --model-path "${MODEL_PATH}"
  --served-model-name moonshotai/Kimi-K2.6
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tp-size 8
  --mem-fraction-static 0.85
  --max-running-requests 64
  --chunked-prefill-size 8192
  --context-length 266240
  --attention-backend flashmla
  --kv-cache-dtype fp8_e4m3
  --mm-attention-backend fa3
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --speculative-draft-attention-backend fa3
  --enable-cache-report
  --enable-metrics
)

if [[ "${MODE}" == image ]]; then
  KIMI_LATENCY_ARGS+=(
    --mem-fraction-static 0.8
    --max-running-requests 48
    --disable-radix-cache
  )
fi

PROFILE_ARGS=()
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
unset SGLANG_HUMMING_MOE_GEMM_TYPE
case "${PROFILE}" in
  marlin-wint4a16)
    ;;
  humming-wint4a16)
    PROFILE_ARGS+=(--quantization humming --moe-runner-backend humming)
    ;;
  humming-wint4afp8)
    export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'
    PROFILE_ARGS+=(--quantization humming --moe-runner-backend humming)
    ;;
  *)
    echo "unsupported PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac
if [[ "${MODE}" == image && "${PROFILE}" != marlin-wint4a16 ]]; then
  export SGLANG_HUMMING_MOE_GEMM_TYPE=indexed
fi

python3 -m sglang.launch_server \
  "${KIMI_LATENCY_ARGS[@]}" \
  "${PROFILE_ARGS[@]}"
```

After the server is ready, run the Figure 2 text sweep from another terminal. TTFT uses five requests at concurrency 1. TPOT uses output length 1,024 and sets batch size equal to concurrency. Unlike the exploratory single-run TPOT panel in the original report, this reproduction protocol runs every point three times and always aggregates all three rounds.

```bash
source /path/to/venvs/humming-blog/bin/activate
export MODEL_PATH=/path/to/models/Kimi-K2.6
export PORT=8188
export PROFILE=humming-wint4afp8
export LATENCY_ROOT=/path/to/performance-results/kimi26-text

for ROUND in 1 2 3; do
  mkdir -p "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}"
  for ISL in 4096 8192 16384 32768 65536 131072 262144; do
    python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --model moonshotai/Kimi-K2.6 \
      --tokenizer "${MODEL_PATH}" \
      --dataset-name random \
      --random-input-len "${ISL}" \
      --random-output-len 1 \
      --num-prompts 5 \
      --request-rate inf \
      --max-concurrency 1 \
      --seed 1024 \
      --flush-cache \
      --warmup-requests 2 \
      --disable-tqdm \
      --output-file "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}/ttft-isl${ISL}-bs1.jsonl"
  done

  for ISL in 1024 32768 131072; do
    for BS in 1 8; do
      python3 -m sglang.benchmark.serving \
        --backend sglang-oai-chat \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --model moonshotai/Kimi-K2.6 \
        --tokenizer "${MODEL_PATH}" \
        --dataset-name random \
        --random-input-len "${ISL}" \
        --random-output-len 1024 \
        --num-prompts "${BS}" \
        --request-rate inf \
        --max-concurrency "${BS}" \
        --seed 1024 \
        --flush-cache \
        --warmup-requests 2 \
        --disable-tqdm \
        --output-file "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}/tpot-isl${ISL}-bs${BS}.jsonl"
    done
  done
done
```

Restart the server and repeat the complete sweep for each profile. Do not mix profiles in one server lifetime.

For Figure 3, launch the server with `MODE=image`, then run this sweep for every profile:

```bash
source /path/to/venvs/humming-blog/bin/activate
export MODEL_PATH=/path/to/models/Kimi-K2.6
export PORT=8188
export PROFILE=humming-wint4afp8
export LATENCY_ROOT=/path/to/performance-results/kimi26-image

for ROUND in 1 2 3; do
  mkdir -p "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}"
  for IMAGE_COUNT in 4 8 12 16 20; do
    python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --model moonshotai/Kimi-K2.6 \
      --tokenizer "${MODEL_PATH}" \
      --dataset-name image \
      --random-input-len 1024 \
      --random-output-len 128 \
      --image-count "${IMAGE_COUNT}" \
      --image-resolution 1080p \
      --image-format jpeg \
      --image-content random \
      --num-prompts 64 \
      --request-rate inf \
      --max-concurrency 1 \
      --seed 1024 \
      --flush-cache \
      --warmup-requests 2 \
      --disable-tqdm \
      --output-file "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}/ttft-isl${IMAGE_COUNT}-bs1.jsonl"
  done
done
```

In the image result filenames, `isl` stores the image count so that the common latency aggregation command below can consume the files without a second schema.

### Accuracy Evaluation

Run the same 200-example GSM8K subset for every profile with 8-shot prompting, temperature 0, and a maximum of 4,096 generated tokens:

```bash
evalscope eval \
  --model moonshotai/Kimi-K2.6 \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --limit 200 \
  --dataset-args '{"gsm8k":{"few_shot_num":8,"few_shot_random":false}}' \
  --generation-config '{"temperature":0,"max_tokens":4096}' \
  --work-dir /path/to/accuracy-results
```

Empty HTTP-200 responses count as incorrect. Restart the server with each profile before repeating the evaluation.

### Agentic Pareto Evaluation

For a given profile and round, set the concurrency variables according to this table:

| Concurrency | `PARALLEL` | `NUMBER` | `DATASET_OFFSET` |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 4 | 0 |
| 2 | 2 | 8 | 4 |
| 4 | 4 | 8 | 12 |
| 8 | 8 | 16 | 20 |

Valid `PROFILE` values are `marlin-wint4a16`, `humming-wint4a16`, and `humming-wint4afp8`.

After every server restart, run one warmup conversation before the formal concurrency sweep:

```bash
export DATASET_PATH=/path/to/data/openhands-kimi26.json
export RESULTS_DIR=/path/to/performance-results/kimi26
export PROFILE=humming-wint4afp8
export ROUND=1

echo "0a5842ae03d3a216ecf7d355f40fbe5c1bbaf8bd8904b87951a65d4057e3ca51  ${DATASET_PATH}" \
  | shasum -a 256 --check

evalscope perf \
  --model moonshotai/Kimi-K2.6 \
  --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api openai \
  --dataset swe_smith \
  --dataset-path "${DATASET_PATH}" \
  --dataset-offset 0 \
  --max-tokens 220 \
  --multi-turn \
  --number 1 \
  --parallel 1 \
  --extra-args '{"ignore_eos":true,"temperature":1.0,"top_p":0.95}' \
  --name warmup \
  --outputs-dir "${RESULTS_DIR}/${PROFILE}/round-${ROUND}/warmup" \
  --no-timestamp
```

Then set the formal concurrency variables and collect one point:

```bash
export DATASET_PATH=/path/to/data/openhands-kimi26.json
export RESULTS_DIR=/path/to/performance-results/kimi26
export PROFILE=humming-wint4afp8
export ROUND=1
export PARALLEL=4
export NUMBER=8
export DATASET_OFFSET=12

evalscope perf \
  --model moonshotai/Kimi-K2.6 \
  --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api openai \
  --dataset swe_smith \
  --dataset-path "${DATASET_PATH}" \
  --dataset-offset "${DATASET_OFFSET}" \
  --max-tokens 220 \
  --multi-turn \
  --number "${NUMBER}" \
  --parallel "${PARALLEL}" \
  --extra-args '{"ignore_eos":true,"temperature":1.0,"top_p":0.95}' \
  --name formal \
  --outputs-dir "${RESULTS_DIR}/${PROFILE}/round-${ROUND}/c-${PARALLEL}" \
  --no-timestamp
```

Run the warmup and all four formal concurrency settings for rounds 1, 2, and 3, restarting the server before each round.

## DeepSeek-V4-Flash

The commands below use the public SGLang and Humming versions pinned in the common environment table. The benchmark checkpoint's routed experts use MXFP4 weights with UE8M0 scales; dense and shared experts use FP8.

### Server Launch

```bash
# Hugging Face base model ID: deepseek-ai/DeepSeek-V4-Flash
export DSV4_MXFP4_MODEL_PATH=/path/to/models/DeepSeek-V4-Flash-MXFP4
export DSV4_MODEL_PATH="${DSV4_MXFP4_MODEL_PATH}"
export DSV4_MODEL_ID=DeepSeek-V4-Flash
export PORT=8001

DSV4_SERVER_ARGS=(
  --model-path "${DSV4_MODEL_PATH}"
  --served-model-name "${DSV4_MODEL_ID}"
  --trust-remote-code
  --host 0.0.0.0
  --port "${PORT}"
  --tp-size 8
  --chunked-prefill-size 8192
  --mem-fraction-static 0.8
  --max-running-requests 64
  --cuda-graph-max-bs-decode 64
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --enable-nsa-prefill-context-parallel
  --nsa-prefill-cp-mode round-robin-split
  --tool-call-parser deepseekv4
  --reasoning-parser deepseek-v4
  --enable-cache-report
  --enable-metrics
  --log-level info
  --watchdog-timeout 3600
)
```

#### Marlin MXFP4A16

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
python3 -m sglang.launch_server \
  "${DSV4_SERVER_ARGS[@]}" \
  --moe-runner-backend marlin \
  --speculative-moe-runner-backend marlin
```

#### Humming MXFP4A16

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
python3 -m sglang.launch_server \
  "${DSV4_SERVER_ARGS[@]}" \
  --moe-runner-backend humming \
  --speculative-moe-runner-backend humming
```

#### Humming MXFP4AFP8

```bash
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'
python3 -m sglang.launch_server \
  "${DSV4_SERVER_ARGS[@]}" \
  --moe-runner-backend humming \
  --speculative-moe-runner-backend humming
```

### Figure 4: TTFT and TPOT Latency

Figure 4 compares two distinct checkpoints: the public official FP8 checkpoint and the Humming MXFP4AFP8 checkpoint described above. Use profile name `official-fp8` for the public checkpoint and `humming-mxfp4afp8` for the Humming checkpoint. Keep every other server argument identical.

Launch the FP8 baseline by setting `DSV4_MODEL_PATH=/path/to/models/DeepSeek-V4-Flash-FP8`, rebuilding `DSV4_SERVER_ARGS`, and running:

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
python3 -m sglang.launch_server "${DSV4_SERVER_ARGS[@]}"
```

Launch the Humming candidate with `DSV4_MODEL_PATH=${DSV4_MXFP4_MODEL_PATH}` and the `Humming MXFP4AFP8` command above. Restart the server for each profile and round. The following sweep uses exact random-token lengths; all plotted points are arithmetic means across three rounds.

```bash
source /path/to/venvs/humming-blog/bin/activate
export DSV4_TOKENIZER_PATH=/path/to/models/DeepSeek-V4-Flash-FP8
export PORT=8001
export PROFILE=humming-mxfp4afp8
export LATENCY_ROOT=/path/to/performance-results/dsv4-flash-latency

for ROUND in 1 2 3; do
  mkdir -p "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}"
  for ISL in 16384 32768 65536 131072; do
    python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --model "${DSV4_MODEL_ID}" \
      --tokenizer "${DSV4_TOKENIZER_PATH}" \
      --dataset-name random \
      --random-input-len "${ISL}" \
      --random-output-len 256 \
      --num-prompts 1 \
      --request-rate inf \
      --max-concurrency 1 \
      --seed 1 \
      --flush-cache \
      --warmup-requests 2 \
      --disable-tqdm \
      --output-file "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}/ttft-isl${ISL}-bs1.jsonl"
  done

  for ISL in 1024 32768 65536 131072; do
    for BS in 1 4 8; do
      python3 -m sglang.benchmark.serving \
        --backend sglang-oai-chat \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --model "${DSV4_MODEL_ID}" \
        --tokenizer "${DSV4_TOKENIZER_PATH}" \
        --dataset-name random \
        --random-input-len "${ISL}" \
        --random-output-len 1024 \
        --num-prompts "${BS}" \
        --request-rate inf \
        --max-concurrency "${BS}" \
        --seed 1 \
        --flush-cache \
        --warmup-requests 2 \
        --disable-tqdm \
        --output-file "${LATENCY_ROOT}/${PROFILE}/round-${ROUND}/tpot-isl${ISL}-bs${BS}.jsonl"
    done
  done
done
```

The original internal latency harness used token-length-matched ShareGPT-style prompts. The command above removes the unavailable seed-text dependency by using SGLang's random dataset. It reproduces the published shape matrix and aggregation method, but exact numerical identity requires the original prompt buckets in addition to the MXFP4 checkpoint.

### Agentic Pareto Evaluation

Figures 6 and 7 use the same generated replay artifact. The supplied Pareto reports fix its SHA256 to `8cc10321e9a8628218dfa1c31eab6d05ef0adacab46f282f4d34899848787215`. Verify the local replay file before running:

```bash
export DSV4_DATASET_PATH=/path/to/data/openhands-glm52-w4afp8.json
echo "8cc10321e9a8628218dfa1c31eab6d05ef0adacab46f282f4d34899848787215  ${DSV4_DATASET_PATH}" \
  | shasum -a 256 --check
```

Valid `PROFILE` values are `marlin-mxfp4a16`, `humming-mxfp4a16`, and `humming-mxfp4afp8`. Run one warmup conversation after every server restart:

```bash
export RESULTS_DIR=/path/to/performance-results/dsv4-flash
export PROFILE=humming-mxfp4afp8
export ROUND=1

evalscope perf \
  --model "${DSV4_MODEL_ID}" \
  --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api openai \
  --dataset swe_smith \
  --dataset-path "${DSV4_DATASET_PATH}" \
  --max-tokens 220 \
  --multi-turn \
  --seed 1024 \
  --number 1 \
  --parallel 1 \
  --extra-args '{"ignore_eos":true,"temperature":1.0,"top_p":0.95}' \
  --name "${PROFILE}_warmup" \
  --outputs-dir "${RESULTS_DIR}/${PROFILE}/round-${ROUND}/warmup" \
  --no-timestamp
```

One formal EvalScope invocation produces all four concurrency points:

```bash
evalscope perf \
  --model "${DSV4_MODEL_ID}" \
  --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api openai \
  --dataset swe_smith \
  --dataset-path "${DSV4_DATASET_PATH}" \
  --max-tokens 220 \
  --multi-turn \
  --seed 1024 \
  --dataset-offset 0 \
  --number 4 8 8 16 \
  --parallel 1 2 4 8 \
  --extra-args '{"ignore_eos":true,"temperature":1.0,"top_p":0.95}' \
  --name "${PROFILE}_round_${ROUND}" \
  --outputs-dir "${RESULTS_DIR}/${PROFILE}/round-${ROUND}" \
  --no-timestamp
```

Repeat the warmup and formal invocation for rounds 1, 2, and 3, restarting the server before each round.
## GLM-5.2 W4AFP8

The GLM-5.2 Pareto result archive contains 24 formal summaries: SGLang CUTLASS WINT4AFP8 and Humming WINT4AFP8, each at concurrency 1, 2, 4, and 8 for three rounds. The checkpoint uses group-size-128 INT4 weights and FP8 activations.

### Server Launch

```bash
# Hugging Face model ID: PhalaCloud/GLM-5.2-W4AFP8
export GLM_MODEL_PATH=/path/to/models/GLM-5.2-W4AFP8
export GLM_MODEL_ID=PhalaCloud/GLM-5.2-W4AFP8
export PORT=8188

GLM_SERVER_ARGS=(
  --model-path "${GLM_MODEL_PATH}"
  --served-model-name "${GLM_MODEL_ID}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --disable-shared-experts-fusion
  --tp-size 8
  --pp-size 1
  --kv-cache-dtype fp8_e4m3
  --context-length 90000
  --mem-fraction-static 0.85
  --max-running-requests 16
  --max-prefill-tokens 8192
  --chunked-prefill-size 8192
  --cuda-graph-max-bs-decode 16
  --reasoning-parser glm45
  --tool-call-parser glm47
  --speculative-algorithm EAGLE
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
  --enable-cache-report
  --enable-metrics
  --log-level info
  --watchdog-timeout 3600
)
```

#### SGLang CUTLASS WINT4AFP8

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
unset SGLANG_HUMMING_ONLINE_QUANT_CONFIG
python3 -m sglang.launch_server \
  "${GLM_SERVER_ARGS[@]}" \
  --quantization w4afp8
```

#### Humming WINT4AFP8

```bash
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
unset SGLANG_HUMMING_ONLINE_QUANT_CONFIG
python3 -m sglang.launch_server \
  "${GLM_SERVER_ARGS[@]}" \
  --quantization humming \
  --moe-runner-backend humming \
  --speculative-moe-runner-backend humming
```

### Agentic Pareto Evaluation

Valid `PROFILE` values are `baseline` and `humming`. Run one warmup conversation after every server restart:

```bash
export GLM_DATASET_PATH=/path/to/data/openhands-glm52-w4afp8.json
export RESULTS_DIR=/path/to/performance-results/glm52
export PROFILE=humming
export ROUND=1

echo "8cc10321e9a8628218dfa1c31eab6d05ef0adacab46f282f4d34899848787215  ${GLM_DATASET_PATH}" \
  | shasum -a 256 --check

evalscope perf \
  --model "${GLM_MODEL_ID}" \
  --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api openai \
  --dataset swe_smith \
  --dataset-path "${GLM_DATASET_PATH}" \
  --max-tokens 220 \
  --multi-turn \
  --number 1 \
  --parallel 1 \
  --extra-args '{"ignore_eos":true,"temperature":1.0,"top_p":0.95}' \
  --name "${PROFILE}_warmup" \
  --outputs-dir "${RESULTS_DIR}/${PROFILE}/round-${ROUND}/warmup" \
  --no-timestamp
```

One formal invocation produces all four concurrency points and the same `parallel_1_number_4`, `parallel_2_number_8`, `parallel_4_number_8`, and `parallel_8_number_16` layout found in the result archive:

```bash
evalscope perf \
  --model "${GLM_MODEL_ID}" \
  --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api openai \
  --dataset swe_smith \
  --dataset-path "${GLM_DATASET_PATH}" \
  --max-tokens 220 \
  --multi-turn \
  --number 4 8 8 16 \
  --parallel 1 2 4 8 \
  --extra-args '{"ignore_eos":true,"temperature":1.0,"top_p":0.95}' \
  --name "${PROFILE}_round_${ROUND}" \
  --outputs-dir "${RESULTS_DIR}/${PROFILE}/round-${ROUND}" \
  --no-timestamp
```

Repeat the warmup and formal invocation for rounds 1, 2, and 3, restarting the server before each round.
