"""
通用验证指标计算与结果保�?
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_LABELS = ["exceptional", "strong", "fair", "limited"]


def calculate_metrics(
    results: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    计算准确率和 per-label Precision / Recall / F1
    支持两种模式�?
      1. logp 模式：基�?top1_prediction / top2_prediction
      2. 文本匹配模式：基�?is_match / prediction

    Args:
        results: 验证结果列表，每条至少包�?ground_truth
        labels: 标签列表（默�?["exceptional","strong","fair","limited"]�?

    Returns:
        指标字典
    """
    if labels is None:
        labels = DEFAULT_LABELS

    total = len(results)
    if total == 0:
        return {"total_samples": 0}

    # avg-accuracy mode (text-mode avg_n): prefer per-item avg_accuracy if present
    use_avg_accuracy = any(r.get("avg_accuracy") is not None for r in results)

    # 自动判断模式
    use_logp = any(r.get("top1_prediction") is not None for r in results)
    use_text = any(r.get("is_match") is not None for r in results)

    correct_top1 = 0
    correct_top1_or_top2 = 0
    correct_text = 0
    sum_avg = 0.0
    cnt_avg = 0

    for r in results:
        gt = (r.get("ground_truth") or "").lower()
        if use_avg_accuracy:
            aa = r.get("avg_accuracy")
            if isinstance(aa, (int, float)):
                sum_avg += float(aa)
                cnt_avg += 1
        elif use_logp:
            p1 = (r.get("top1_prediction") or "").lower()
            p2 = (r.get("top2_prediction") or "").lower()
            if p1 and p1 in gt:
                correct_top1 += 1
            if (p1 and p1 in gt) or (p2 and p2 in gt):
                correct_top1_or_top2 += 1
        elif use_text and r.get("is_match"):
            correct_text += 1

    # Per-label P / R / F1
    # avg_accuracy 模式：用 vote_predictions 中的多数票（或所有预测）统计 per-label
    label_stats = {label: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for label in labels}

    for r in results:
        gt = (r.get("ground_truth") or "").lower()
        if use_avg_accuracy:
            # �?vote_predictions 中每次预测的平均贡献，等效于对每条样本做软统�?
            preds = [p for p in (r.get("vote_predictions") or []) if p]
            if not preds:
                for label in labels:
                    if label == gt:
                        label_stats[label]["fn"] += 1
                continue
            n = len(preds)
            for label in labels:
                hit_as_pred = sum(1 for p in preds if p == label) / n
                is_gt = (label == gt)
                # tp: 预测为该 label �?gt 也是�?label（按比例�?
                if is_gt:
                    label_stats[label]["tp"] += hit_as_pred
                    label_stats[label]["fn"] += (1 - hit_as_pred)
                else:
                    label_stats[label]["fp"] += hit_as_pred
        elif use_logp:
            pred = (r.get("top1_prediction") or "").lower()
            if not pred:
                continue
            for label in labels:
                if label == pred and label == gt:
                    label_stats[label]["tp"] += 1
                elif label == pred and label != gt:
                    label_stats[label]["fp"] += 1
                elif label != pred and label == gt:
                    label_stats[label]["fn"] += 1
        elif use_text:
            pred = (r.get("prediction") or "").lower()
            if not pred:
                continue
            for label in labels:
                if label == pred and label == gt:
                    label_stats[label]["tp"] += 1
                elif label == pred and label != gt:
                    label_stats[label]["fp"] += 1
                elif label != pred and label == gt:
                    label_stats[label]["fn"] += 1

    per_label = {}
    for label in labels:
        tp = label_stats[label]["tp"]
        fp = label_stats[label]["fp"]
        fn = label_stats[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    # Balanced accuracy = macro recall（每�?label �?recall 取算术平均，抵消类别不平衡）
    recalls = [
        per_label[label]["recall"]
        for label in labels
        if per_label[label]["support"] > 0
    ]
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0

    # Macro F1 = ƽ��ÿ�� label �� F1����֧��Ϊ 0 �ı�ǩ������
    f1s = [
        per_label[label]['f1']
        for label in labels
        if per_label[label]['support'] > 0
    ]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    metrics: Dict[str, Any] = {
        "total_samples": total,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_label_metrics": per_label,
    }
    if use_avg_accuracy:
        metrics["accuracy_avg"] = (sum_avg / cnt_avg) if cnt_avg > 0 else 0.0
    elif use_logp:
        metrics["accuracy_top1"] = correct_top1 / total
        metrics["accuracy_top1_or_top2"] = correct_top1_or_top2 / total
    elif use_text:
        metrics["accuracy_text_match"] = correct_text / total

    return metrics


def save_results(
    metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """
    保存指标和详细结果到 output_dir

    Args:
        metrics: calculate_metrics 返回的字�?
        results: 验证结果列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    overall_path = os.path.join(output_dir, "validation_results.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    logger.info(f"整体结果已保存到: {overall_path}")

    jsonl_path = os.path.join(output_dir, "validation_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"每条结果逐行保存�? {jsonl_path}")

