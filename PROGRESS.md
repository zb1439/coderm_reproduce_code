# CodeRM Reproduce - Progress Report

_Generated: 2026-04-16 23:44:33_

## 项目概要

复现 CodeRM 论文，用 3 个小模型在 4 个 benchmark 上生成 100 candidate × task，然后打标得到 pass/fail ground truth。

**Models:**
- `gemma-3-4b` → `google/gemma-3-4b-it` (OpenRouter)
- `ministral-3b` → `mistralai/ministral-3b-2512` (OpenRouter)
- `qwen3-8b` → `qwen/qwen3-8b` (OpenRouter, `/no_think` 禁用思考模式)

## 当前状态 (Solution 生成 + 打标)

| Benchmark | Model | Solution | Label | Pass Rates |
|---|---|---|---|---|
| HumanEval+ | gemma-3-4b | ✅ 16400/16400 | ✅ | sol 65.5% | p@100 78.7% |
| HumanEval+ | ministral-3b | ✅ 16400/16400 | ✅ | sol 74.4% | p@100 94.5% |
| HumanEval+ | qwen3-8b | ✅ 16147/16400 | ✅ | sol 77.6% | p@100 84.1% |
| MBPP+ | gemma-3-4b | ✅ 37800/37800 | ✅ | sol 68.8% | p@100 78.0% |
| MBPP+ | ministral-3b | ✅ 37791/37800 | ✅ | sol 51.9% | p@100 81.0% |
| MBPP+ | qwen3-8b | ✅ 37615/37800 | ✅ | sol 69.9% | p@100 74.9% |
| LCB v4 (168) | gemma-3-4b | ✅ 16742/16800 | ❌ | - |
| LCB v4 (168) | ministral-3b | ✅ 16599/16800 | ❌ | - |
| LCB v4 (168) | qwen3-8b | ✅ 16541/16800 | ✅ | sol 41.9% | p@100 55.4% |
| LCB v6 new (342) | gemma-3-4b | 🔄 30893/34200 (90%) | ❌ | - |
| LCB v6 new (342) | ministral-3b | 🔄 14393/34200 (42%) | ❌ | - |
| LCB v6 new (342) | qwen3-8b | 🔄 29830/34200 (87%) | ❌ | - |

## 文件路径

| Benchmark | Model | Sol Path | Label Path |
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

## 剩余任务

**打标未完成：5 个**

- LCB v4 (168) / gemma-3-4b
- LCB v4 (168) / ministral-3b
- LCB v6 new (342) / gemma-3-4b
- LCB v6 new (342) / ministral-3b
- LCB v6 new (342) / qwen3-8b

**Solution 生成未完成：3 个**

- LCB v6 new (342) / gemma-3-4b: 🔄 30893/34200 (90%)
- LCB v6 new (342) / ministral-3b: 🔄 14393/34200 (42%)
- LCB v6 new (342) / qwen3-8b: 🔄 29830/34200 (87%)

## 正在运行

`bash scripts/run_remaining_serial.sh` 串行脚本

执行顺序：
1. ✅ 并发启动 v6 new 3 模型推理（API，不吃 CPU）
2. 🔄 LCB v4 Gemma 打标（正在跑，2 workers）
3. ⏳ LCB v4 Ministral 打标
4. ⏳ 等 v6 new 推理完成
5. ⏳ 提取 v6 new func
6. ⏳ v6 new 打标：Qwen → Gemma → Ministral

**注意**: 之前用 4 workers 内存爆了，已降为 `--num_workers 2 --timeout 20`

## 关键脚本

- `inference/inference_api.py` - 解生成（OpenRouter API）
- `evaluation/run_lcb_eval.py` - LiveCodeBench 打标（用 lcb_runner，需 PYTHONPATH=/tmp/lcb_repo）
- `preprocess/extract_solution.py` - 从 raw response 提取代码
- `preprocess/convert_class_to_func.py` - LCB class Solution → 独立函数 (for coderm UT eval)
- `preprocess/build_lcb_v6_new_prompts.py` - 为 LCB v6 new 342 题生成 prompt 文件
- `scripts/run_remaining_serial.sh` - **串行执行剩余任务的脚本**
- `scripts/run_eval_remote.sh` - 给服务器跑的评估脚本

## 已知问题 & 经验

1. **Qwen 3.5 不是 Qwen 公开版本**，memory 中记录的 Qwen3.5 是 DeepInfra 挂的 reasoning 模型，需要 `chat_template_kwargs: {enable_thinking: False}` 才能真正关闭思考。即使关了也话痨。不如 Qwen3-8B 稳定。

2. **Qwen3-8B 需要 `/no_think` 关闭思考模式**，否则 reasoning 占满 token。OpenRouter 上用 `extra_body.enable_thinking=False` 或 user message 加 `/no_think`。

3. **Ministral-3B LCB 表现差是真的**（5% sol pass vs Qwen3-8B 39%），参数量小且没 thinking，难题逻辑推理弱。MBPP+ 上还有 camelCase → snake_case 函数名错误。

4. **evalplus 在 macOS 上 setrlimit 会炸**，要 try/except 绕过（在 `.venv-local/lib/.../evalplus/eval/utils.py`）。

5. **lcb_runner 在 macOS 上也有 setrlimit 问题**，在 `/tmp/lcb_repo/lcb_runner/evaluation/testing_util.py` 同样 patch。

6. **lcb_runner 用相对路径加载 few-shot**，必须从 `/tmp/lcb_repo` 目录运行或 patch `code_generation.py`。

7. **LCB 需要的 benchmark**：coderm 论文用的 168 task 是 release_v4 子集（不跨版本）；v6 new 342 是 v6 - v4 的增量（2024-09-22 ~ 2025-04-06）。

8. **LCB 评估慢**：16800 solutions × 2 workers ≈ 1-3 小时（视 timeout 触发率）。之前 4 workers 会 OOM。

9. **OpenRouter API key** 通过环境变量 `OPENROUTER_API_KEY` 传入（运行脚本前 export）

## Git 状态

- 分支: `codex/solution-api`
- PR: #3 (on github.com/zb1439/coderm_reproduce_code)
- 最近 commit: Qwen3-8B 解 + LCB v6 new 部分解 已 push

