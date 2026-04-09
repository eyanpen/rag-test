#!/usr/bin/env python3
"""LLM-as-Judge: 用 LLM 对 predictions.json 中每条结果打分 (0-10)."""
import argparse
import json
import logging
import re
import time

import openai

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

JUDGE_PROMPT = """\
你是一个严格的答案评估专家。请对比 ground_truth（标准答案）和 generated_answer（生成答案），给出 0-10 的整数分数。

评分标准：
- 10分：语义完全一致，关键信息无遗漏
- 7-9分：核心含义正确，有少量细节差异
- 4-6分：部分正确，有明显遗漏或偏差
- 1-3分：仅有少量相关内容
- 0分：完全不相关或答非所问

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

请严格按以下 JSON 格式回复，不要输出其他内容：
{{"score": <0-10整数>, "reason": "<简要说明>"}}"""


def judge_one(client: openai.OpenAI, model: str, item: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        ground_truth=item.get("ground_truth", ""),
        generated_answer=item.get("generated_answer", ""),
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            # 提取 JSON
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                result = json.loads(m.group())
                return {"judge_score": int(result["score"]), "judge_reason": result["reason"]}
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed for {item.get('id','?')}: {e}")
            time.sleep(2)
    return {"judge_score": -1, "judge_reason": "Judge failed after retries"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="no-key")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if "error" in item:
            item["judge_score"] = -1
            item["judge_reason"] = "Skipped: query error"
            continue
        log.info(f"Judging {i+1}/{len(data)}: {item.get('id','?')}")
        result = judge_one(client, args.model, item)
        item.update(result)

    with open(args.input, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    scored = [d for d in data if d.get("judge_score", -1) >= 0]
    if scored:
        avg = sum(d["judge_score"] for d in scored) / len(scored)
        log.info(f"Judge done: {len(scored)} scored, avg={avg:.1f}/10")


if __name__ == "__main__":
    main()
