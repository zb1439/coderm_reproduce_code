# CodeRM Reproduce - Progress Report

_Last updated: 2026-04-23 10:42_

## 概要

用 3 个小模型在 4 个 benchmark 上生成 100 candidate × task，然后打标得到 pass/fail ground truth。

**Models:**
- `gemma-3-4b` → `google/gemma-3-4b-it` (OpenRouter)
- `ministral-3b` → `mistralai/ministral-3b-2512` (OpenRouter)
- `qwen3-8b` → `qwen/qwen3-8b` (OpenRouter, `/no_think`)

**Benchmarks:**
- HumanEval+ (164 tasks) - evalplus 格式
- MBPP+ (378 tasks) - evalplus 格式
- LCB v4 (168 tasks, coderm 论文子集) - LCB graded 格式
- LCB v6new (342 新题, v6-v4 增量) - LCB graded 格式

## 最终结果表（Pass Rates）

| Benchmark | Model | Sol 数 | Label 数 | Sol pass rate | Pass@100 |
|---|---|---|---|---|---|
| HumanEval+ | gemma-3-4b | 16400/16400 | 16400/16400 ✅ | 65.5% | 78.7% |
| HumanEval+ | ministral-3b | 16400/16400 | 16400/16400 ✅ | 74.4% | 94.5% |
| HumanEval+ | qwen3-8b | 16400/16400 | 16400/16400 ✅ | 77.4% | 84.1% |
| MBPP+ | gemma-3-4b | 37800/37800 | 37800/37800 ✅ | 68.8% | 78.0% |
| MBPP+ | ministral-3b | 37800/37800 | 37800/37800 ✅ | 51.8% | 81.2% |
| MBPP+ | qwen3-8b | 37800/37800 | 37800/37800 ✅ | 69.9% | 74.9% |
| LCB v4 (168) | gemma-3-4b | 16800/16800 | 16800/16800 ✅ | 20.5% | 27.4% |
| LCB v4 (168) | ministral-3b | 16800/16800 | 16800/16800 ✅ | 28.3% | 60.1% |
| LCB v4 (168) | qwen3-8b | 16800/16800 | 16800/16800 ✅ | 41.9% | 55.4% |
| LCB v6 new (342) | gemma-3-4b | 34200/34200 | 34200/34200 ✅ | 15.3% | 21.9% |
| LCB v6 new (342) | ministral-3b | 34200/34200 | 34100/34200 ⚠️ 341/342 | 20.5% | 44.3% |
| LCB v6 new (342) | qwen3-8b | 34200/34200 | 34200/34200 ✅ | 22.7% | 33.0% |

## 文件路径

| Benchmark | Model | Solution | Label |
|---|---|---|---|
| HumanEval+ | gemma-3-4b | `data/result/humaneval_plus/sol_gemma-3-4b_100_func.jsonl` | `output/humaneval_plus/evalplus_gemma-3-4b_eval_results.json` |
| HumanEval+ | ministral-3b | `data/result/humaneval_plus/sol_ministral-3b_100_func.jsonl` | `output/humaneval_plus/evalplus_ministral-3b_eval_results.json` |
| HumanEval+ | qwen3-8b | `data/result/humaneval_plus/sol_qwen3-8b_100_func.jsonl` | `output/humaneval_plus/evalplus_qwen3-8b_eval_results.json` |
| MBPP+ | gemma-3-4b | `data/result/mbpp_plus/sol_gemma-3-4b_100_func.jsonl` | `output/mbpp_plus/evalplus_gemma-3-4b_eval_results.json` |
| MBPP+ | ministral-3b | `data/result/mbpp_plus/sol_ministral-3b_100_func.jsonl` | `output/mbpp_plus/evalplus_ministral-3b_eval_results.json` |
| MBPP+ | qwen3-8b | `data/result/mbpp_plus/sol_qwen3-8b_100_func.jsonl` | `output/mbpp_plus/evalplus_qwen3-8b_eval_results.json` |
| LCB v4 (168) | gemma-3-4b | `data/result/livecodebench/sol_gemma-3-4b_100_func_converted.jsonl` | `output/livecodebench/graded_gemma-3-4b.json` |
| LCB v4 (168) | ministral-3b | `data/result/livecodebench/sol_ministral-3b_100_func_converted.jsonl` | `output/livecodebench/graded_ministral-3b.json` |
| LCB v4 (168) | qwen3-8b | `data/result/livecodebench/sol_qwen3-8b_100_func_converted.jsonl` | `output/livecodebench/graded_qwen3-8b.json` |
| LCB v6 new (342) | gemma-3-4b | `data/result/livecodebench_v6new/sol_gemma-3-4b_100_func_converted.jsonl` | `output/livecodebench_v6new/graded_gemma-3-4b.json` |
| LCB v6 new (342) | ministral-3b | `data/result/livecodebench_v6new/sol_ministral-3b_100_func_converted.jsonl` | `output/livecodebench_v6new/graded_ministral-3b.json` |
| LCB v6 new (342) | qwen3-8b | `data/result/livecodebench_v6new/sol_qwen3-8b_100_func_converted.jsonl` | `output/livecodebench_v6new/graded_qwen3-8b.json` |

## 数据完整性

- ✅ 11/12 组合完全 100% 打标
- ⚠️ LCB v6new Ministral-3B: 341/342 (1 个 task `abc377_e` 因为某个 solution 导致 ProcessPool 崩溃，无法 grade)
- Raw solutions 全部 100% 完整
- Func solutions 全部 = raw（`--require_extractable` 保证）

## 关键经验（更新 2026-04）

1. **macOS 下 LCB 评估要小心内存**：
   - `run_lcb_eval.py` 需分 chunk 处理，不然会 OOM 崩系统
   - 某些 solution 在 State=U 状态疯狂分配内存，SIGKILL 都杀不掉
   - 配合 `/tmp/runaway_killer.sh` 监控，RSS>2GB 就 kill

2. **inference_api.py 新增开关**：
   - `--require_extractable`: 丢弃无法提取的 response 重新生成
   - `--strict_code_only`: user prompt 追加"只输出代码无注释"，把 Ministral/Gemma 在难题上的罗嗦注释抑制住
   - 两个一起用 v6new 的 extraction loss 从 15% → 0%

3. **Qwen3-8B 必须 `/no_think`**：否则所有 token 用在 reasoning，输不出代码

4. **分 chunk 打标支持 resume**：`--chunk_size N` 每 N 个 task 存一次盘，进程挂了可接着跑

## 关键脚本

- `inference/inference_api.py` — 生成（支持 `--require_extractable`, `--strict_code_only`, `--disable_thinking`）
- `evaluation/run_lcb_eval.py` — LCB 打标，分 chunk 流式写盘
- `evaluation/incremental_evalplus.py` — HumanEval+/MBPP+ 增量打标
- `evaluation/incremental_lcb_eval.py` — LCB 增量打标
- `preprocess/extract_solution.py` — raw → func
- `preprocess/convert_class_to_func.py` — LCB class Solution → standalone func
- `preprocess/build_lcb_v6_new_prompts.py` — 生成 LCB v6new prompt

## 使用

```bash
# Clone + checkout branch
git clone https://github.com/zb1439/coderm_reproduce_code.git
cd coderm_reproduce_code
git checkout codex/solution-api

# API key (OpenRouter)
export OPENROUTER_API_KEY=<your key>

# Generate + label
bash scripts/run_eval_remote.sh
```

见 `EVAL_PLAN.md` 获取后续 best-of-N via reward-model UT eval 实验的详细方案。
