"""
Metadata 处理 pipeline：Excel 双层表 → 合并 → 中文字段名翻译为英文 → 输出 JSON。

合并自：
- scripts/process_metadata_excel.py（Excel → 带 journal-data 的 JSON）
- scripts/translate_final.py（按位置的字段名映射）
- scripts/fix_missing_fields.py（作者/机构/摘要等补充映射）
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 按 Excel 导出后常见顺序的英文字段名（仅当列名未在 FIELD_NAME_MAPPING 中时作后备）
FIELD_NAMES_BY_POSITION = [
    "title",
    "start_page",
    "end_page",
    "page_info",
    "issn",
    "eissn",
    "journal_name",
    "publish_info",
    "publish_date",
    "year",
    "month",
    "baseid",
    "id",
    "journal-data",
]

# 所有可能出现的中文/原始列名 → 英文字段名（主表 + journal-data 嵌套均用）
# 以列名为准，避免不同 Excel（如 OB 元数据表）列顺序与 FIELD_NAMES_BY_POSITION 不一致导致错位
FIELD_NAME_MAPPING = {
    # Sheet1 元数据常见列（OB: 题名、作者、关键词、机构、摘要、栏目信息、doi、起始页、结束页、页码信息、issn、eissn、期刊名、url、出版日期、出版年、卷、期、baseid、rawid、id）
    "题名": "title",
    "标题": "title",
    "作者": "authors",
    "关键词": "keywords",
    "机构": "affiliation",
    "摘要": "abstract",
    "栏目信息": "section_info",
    "篇目信息": "article_info",
    "doi": "doi",
    "起始页": "start_page",
    "结束页": "end_page",
    "页码信息": "page_info",
    "页数信息": "page_info",
    "issn": "issn",
    "eissn": "eissn",
    "期刊名": "journal_name",
    "url": "url",
    "出版日期": "publish_date",
    "出版信息": "publish_info",
    "出版年": "publication_year",
    "年": "year",
    "月": "month",
    "卷": "volume",
    "期": "issue",
    "baseid": "baseid",
    "rawid": "rawid",
    "id": "id",
    # Sheet0 期刊表 + 合并后的 journal-data
    "Journal": "Journal",  # 保留，嵌套内不翻译
    "下载时间区间": "download_period",
    "备注": "note",
    "Unnamed: 2": "journal_url",
}

# 兼容旧变量名；嵌套 dict 翻译时用（不含主表专有键）
ADDITIONAL_FIELD_MAPPING = {
    "作者": "authors",
    "机构": "affiliation",
    "摘要": "abstract",
    "出版年": "publication_year",
    "卷": "volume",
    "期": "issue",
    "下载时间区间": "download_period",
    "备注": "note",
    "Unnamed: 2": "journal_url",
}


def process_metadata_excel(excel_path: str | Path) -> List[Dict[str, Any]]:
    """
    读取双层 Excel，将 Sheet0（期刊信息）按 id 合并进 Sheet1（主表）的 journal-data 字段。

    Args:
        excel_path: metadata.xlsx 路径

    Returns:
        记录列表，每条含 journal-data（可能为空 dict）
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要安装 pandas 与 openpyxl: pip install pandas openpyxl") from None

    excel_path = Path(excel_path)
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    if len(sheet_names) < 2:
        raise ValueError(f"Excel 需至少 2 个工作表，当前: {sheet_names}")

    df1 = pd.read_excel(excel_path, sheet_name=0)
    df2 = pd.read_excel(excel_path, sheet_name=1)

    def find_id_column(df: "pd.DataFrame") -> str:
        if "id" in df.columns:
            return "id"
        for col in df.columns:
            if "id" in col.lower():
                return col
        return df.columns[0]

    id_col1 = find_id_column(df1)
    id_col2 = find_id_column(df2)

    journal_dict: Dict[Any, Dict] = {}
    for _, row in df1.iterrows():
        jid = row[id_col1]
        jdata = row.drop(id_col1).to_dict()
        jdata = {k: v for k, v in jdata.items() if pd.notna(v)}
        journal_dict[jid] = jdata

    result: List[Dict[str, Any]] = []
    for _, row in df2.iterrows():
        record = row.to_dict()
        record = {k: v for k, v in record.items() if pd.notna(v)}
        rid = record.get(id_col2)
        record["journal-data"] = journal_dict.get(rid, {})
        result.append(record)

    logger.info("process_metadata_excel: 记录数=%d, 期刊匹配=%d", len(result), sum(1 for r in result if r.get("journal-data")))
    return result


def _build_field_mapping(first_keys: List[str]) -> Dict[str, str]:
    """列名 → 英文字段名：优先按列名查 FIELD_NAME_MAPPING，否则按位置用 FIELD_NAMES_BY_POSITION。"""
    mapping: Dict[str, str] = {}
    for i, key in enumerate(first_keys):
        if key in FIELD_NAME_MAPPING:
            mapping[key] = FIELD_NAME_MAPPING[key]
        elif i < len(FIELD_NAMES_BY_POSITION):
            mapping[key] = FIELD_NAMES_BY_POSITION[i]
        else:
            mapping[key] = ADDITIONAL_FIELD_MAPPING.get(key, key)
    return mapping


def _translate_nested_dict(obj: Any, key_mapping: Dict[str, str]) -> Any:
    """递归翻译嵌套 dict 的 key（仅对 dict 子节点应用 key_mapping）。"""
    if isinstance(obj, dict):
        return {key_mapping.get(k, k): _translate_nested_dict(v, key_mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_translate_nested_dict(x, key_mapping) for x in obj]
    return obj


def translate_field_names(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将每条记录的中文 key 按首条顺序与 ADDITIONAL_FIELD_MAPPING 翻译为英文；
    并递归翻译嵌套 dict（如 journal-data）中的 key。
    """
    if not records:
        return records
    first_keys = list(records[0].keys())
    field_mapping = _build_field_mapping(first_keys)
    # 嵌套 dict 仅用 ADDITIONAL_FIELD_MAPPING（无位置顺序）
    nested_mapping = {**ADDITIONAL_FIELD_MAPPING}

    for i, record in enumerate(records):
        new_record = {}
        for old_key, value in record.items():
            new_key = field_mapping.get(old_key, old_key)
            # 递归翻译嵌套 dict 中的中文 key
            if isinstance(value, dict):
                value = _translate_nested_dict(value, nested_mapping)
            new_record[new_key] = value
        record.clear()
        record.update(new_record)
        if (i + 1) % 20000 == 0:
            logger.info("translate_field_names: 已处理 %d/%d", i + 1, len(records))
    return records


def run_pipeline(
    excel_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    完整 pipeline：metadata.xlsx → 合并 → 字段名翻译 → 写出 metadata_translated.json。

    Args:
        excel_path: 输入的 metadata.xlsx 路径
        output_path: 输出的 JSON 路径；默认在 excel 同目录下，文件名为 metadata_translated.json

    Returns:
        输出文件的 Path
    """
    excel_path = Path(excel_path)
    if output_path is None:
        output_path = excel_path.parent / "metadata_translated.json"
    else:
        output_path = Path(output_path)

    records = process_metadata_excel(excel_path)
    translate_field_names(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("已保存 %d 条记录到 %s", len(records), output_path)
    return output_path
