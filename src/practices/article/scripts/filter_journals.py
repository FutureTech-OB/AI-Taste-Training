"""
随机筛选期刊脚本
根据目标文章数量筛选文章，优先保留最新的文章
"""
import asyncio
import sys
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from bson import ObjectId
import toml

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.practices.article.models import Article


def load_database_config() -> Tuple[str, str]:
    """从配置文件加载数据库配置"""
    config_path = project_root / 'assets' / 'database.toml'
    default_db_name = 'RItest'
    default_connection_string = f"mongodb://root:password@166.111.96.30:27027/{default_db_name}?authSource=admin"
    
    if not config_path.exists():
        return default_connection_string, default_db_name
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        default_config = config.get('default', {})
        db_name = default_config.get('db_name', default_db_name)
        conn_str = default_config.get('connection_string', '')
        
        if not conn_str:
            conn_str = default_connection_string.replace(default_db_name, db_name)
        elif '<DBNAME>' in conn_str:
            conn_str = conn_str.replace('<DBNAME>', db_name)

        return conn_str, db_name
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return default_connection_string, default_db_name


def calculate_dynamic_factor(
    items: List[Dict],
    target_count: int,
    total_items: int,
    factor_range: Tuple[float, float] = (0.3, 1.0),
    max_iterations: int = 50,
    tolerance: int = 10
) -> float:
    """
    动态计算分配因子，使得分配的总数接近目标值（只能多不能少，误差tolerance内可接受）

    Args:
        items: 项目列表，每项包含 'count' 字段
        target_count: 目标数量
        total_items: 总项目数
        factor_range: 因子范围 (min, max)
        max_iterations: 最大迭代次数
        tolerance: 可接受的误差范围

    Returns:
        最优因子
    """
    if target_count >= total_items:
        return 1.0

    def calculate_allocation(factor):
        allocated = 0
        for item in items:
            count = item['count']
            if total_items > 0:
                base_ratio = count / total_items
                discounted_ratio = base_ratio ** factor
                target = max(1, int(target_count * discounted_ratio))
                target = min(target, count)
                allocated += target
        return allocated

    left, right = factor_range
    best_factor = (left + right) / 2

    for _ in range(max_iterations):
        mid = (left + right) / 2
        allocated = calculate_allocation(mid)

        if allocated < target_count:
            left = mid
        elif allocated - target_count > tolerance:
            right = mid
            else:
            best_factor = mid
            break
        best_factor = mid

    return best_factor


async def allocate_articles_by_year(
    articles: List[Any],
    target_count: int,
    current_year: int = 2025,
    return_year_info: bool = False
) -> Tuple[List[str], Optional[Dict[int, Dict[str, int]]]]:
    """
    按年份分配文章：2025年全选，其他年份按动态权重分配后随机采样

    Args:
        articles: 文章列表
        target_count: 目标文章数量
        current_year: 当前年份
        return_year_info: 是否返回每年的quota和原始数量信息

    Returns:
        (选中的文章ID列表, 每年的quota和原始数量信息)
    """
    if not articles:
        return [], None if return_year_info else None

    # 按年份分组
    articles_by_year = {}
    for article in articles:
        year = article.published_year if article.published_year else current_year
        if year not in articles_by_year:
            articles_by_year[year] = []
        articles_by_year[year].append(article)

    # 先计算每年的 quota 分配
    all_years = sorted(articles_by_year.keys(), reverse=True)
    if not all_years:
        return [], None if return_year_info else None

    # 步骤1：2025年全选（作为基准）
    year_2025_articles = articles_by_year.get(current_year, [])
    year_2025_count = len(year_2025_articles)
    
    # 如果2025年文章数 >= quota，直接从2025年随机选quota篇
    if year_2025_count >= target_count:
        sampled_2025 = random.sample(year_2025_articles, target_count)
        year_info = None
        if return_year_info:
            year_info = {
                current_year: {
                    'original': year_2025_count,
                    'quota': target_count,
                    'selected': target_count
                }
            }
        return [str(article.id) for article in sampled_2025], year_info
    
    # 否则，2025年全选，剩余quota分配给其他年份
    selected_ids = []
    year_info = {} if return_year_info else None
    
    if year_2025_count > 0:
        selected_ids.extend([str(article.id) for article in year_2025_articles])
        if return_year_info:
            year_info[current_year] = {
                'original': year_2025_count,
                'quota': year_2025_count,
                'selected': year_2025_count
            }
    
    remaining_quota = target_count - year_2025_count
    if remaining_quota <= 0:
        return selected_ids, year_info if return_year_info else None
    
    # 步骤2：分配剩余quota到其他年份，使用二分搜索找decay_factor
    # 约束：2025 >= 2024 >= 2023 >= ...（递减）
    other_years = sorted([y for y in all_years if y != current_year], reverse=True)
    if not other_years:
        return selected_ids
    
    def calculate_allocation(decay_factor: float) -> Tuple[int, Dict[int, int]]:
        """
        计算给定decay_factor下的分配总数和每年的分配数
        约束：每年 <= 第一年选中数 * decay_factor^迭代次数
        例如：2024 <= 2025 * decay^1, 2023 <= 2025 * decay^2, ...
        """
        allocations = {}
        
        for idx, year in enumerate(other_years):
            # 迭代次数：2024是第1次，2023是第2次，以此类推
            iteration = idx + 1
            # 该年的quota = 第一年选中数 * decay_factor^迭代次数
            year_quota = year_2025_count * (decay_factor ** iteration)
            # 实际选中数 = min(quota, 该年文章总数)
            actual_count = min(int(year_quota), len(articles_by_year[year]))
            allocations[year] = actual_count
        
        total_allocated = sum(allocations.values())
        return total_allocated, allocations
    
    # 先尝试 decay_factor = 1.0（最大可能）
    max_possible, _ = calculate_allocation(1.0)
    
    if max_possible < remaining_quota:
        # 不够，直接用 1.0，能拿多少拿多少
        best_decay_factor = 1.0
        else:
        # 够，用二分搜索找到合适的 decay_factor
        left, right = 0.5, 1.0
        best_decay_factor = 0.9
        tolerance = 1  # 允许的误差范围
        
        for _ in range(50):
            mid = (left + right) / 2
            allocated, _ = calculate_allocation(mid)
            
            if allocated < remaining_quota:
                left = mid  # 需要更大的 decay_factor
            elif allocated - remaining_quota > tolerance:
                right = mid  # 太多了，减小 decay_factor
            else:
                best_decay_factor = mid
                break
            best_decay_factor = mid
    
    # 使用找到的best_decay_factor计算每年的分配
    _, year_allocations = calculate_allocation(best_decay_factor)
    
    # 步骤3：根据每年的分配随机采样
    for year in other_years:
        allocation = year_allocations.get(year, 0)
        original_count = len(articles_by_year[year])
        if allocation > 0:
            year_articles = articles_by_year[year]
            sampled = random.sample(year_articles, min(allocation, len(year_articles)))
            selected_ids.extend([str(article.id) for article in sampled])
            if return_year_info:
                year_info[year] = {
                    'original': original_count,
                    'quota': allocation,
                    'selected': len(sampled)
                }
        elif return_year_info:
            year_info[year] = {
                'original': original_count,
                'quota': 0,
                'selected': 0
            }
    
    return selected_ids, year_info if return_year_info else None


async def filter_articles_by_target(
    subject: str,
    target_articles: Dict[str, int],
    rank_stats: Dict[str, Dict[str, Any]],
    connection_string: str,
    db_name: str,
    show_journal_quota: bool = False,
) -> Dict[str, List[str]]:
    """
    根据目标文章数量筛选文章
    
    策略：
    1. rank 级别：目标文章数 target_articles[rank] 为硬性 quota（允许 +3 误差）
    2. 期刊级别：按期刊发文量比例（带折扣因子）分配 journal quota，并归一化使总和≈rank quota
    3. 年份级别：对每本期刊，2025 年优先保留，其余年份按固定 decay_factor 分配后随机采样
    
    Args:
        subject: 学科名称
        target_articles: 每个 rank 的目标文章数量（rank quota）
        rank_stats: 每个 rank 的统计信息
        connection_string: MongoDB连接字符串
        db_name: 数据库名称
    
    Returns:
        (每个 rank 应该保留的文章ID列表, 每期刊每年的quota和原始数量信息)
    """
    selected_article_ids = {}
    journal_year_info_all = {}  # rank -> journal -> year -> {original, quota, selected}
    
    for rank, target_count in target_articles.items():
        if rank not in rank_stats:
            selected_article_ids[rank] = []
            continue
        
        stats = rank_stats[rank]
        journals = stats["journals"]
        total_articles_in_rank = stats["total_articles"]
        
        # 按发文量降序排序期刊
        journals_sorted = sorted(
            journals, key=lambda x: x["article_count"], reverse=True
        )

        # -------------------------------
        # 期刊级别：按比例 + 折扣因子分配 journal quota
        # -------------------------------
        discount_factor = 0.9  # 固定衰减因子，控制“大期刊统治力”

        # 1) 计算每本期刊的权重
        journal_weights: Dict[str, float] = {}
        for j in journals_sorted:
            article_count = j["article_count"]
            if total_articles_in_rank > 0:
                base_ratio = article_count / total_articles_in_rank
                weight = base_ratio**discount_factor
            else:
                weight = 1.0
            journal_weights[j["journal"]] = weight

        total_weight = sum(journal_weights.values())

        # 2) 计算 raw quota 并初始化为四舍五入结果
        journal_targets: Dict[str, int] = {}
        raw_quotas: Dict[str, float] = {}
        for j in journals_sorted:
            journal = j["journal"]
            article_count = j["article_count"]
            if total_weight > 0:
                raw_quota = target_count * journal_weights[journal] / total_weight
            else:
                raw_quota = target_count / max(len(journals_sorted), 1)

            raw_quotas[journal] = raw_quota

            quota = int(round(raw_quota))
            # 不超过该期刊总文章数
            quota = min(quota, article_count)
            # 不允许负数
            quota = max(quota, 0)
            journal_targets[journal] = quota

        # 3) 调整总 quota，使其在 [target_count, target_count + 3] 范围内（如果可能）
        total_quota = sum(journal_targets.values())

        # 如果总 quota 明显小于目标，并且还有空间，则向高权重期刊补充
        if total_quota < target_count:
            diff = target_count - total_quota
            # 按 raw_quota 从大到小排序，优先给 quota 偏小且有容量的期刊补 1
            journals_by_priority = sorted(
                journals_sorted,
                key=lambda j: raw_quotas.get(j["journal"], 0.0),
                reverse=True,
            )
            for j in journals_by_priority:
                if diff <= 0:
                        break
                journal = j["journal"]
                article_count = j["article_count"]
                current_quota = journal_targets[journal]
                if current_quota < article_count:
                    journal_targets[journal] = current_quota + 1
                    diff -= 1

            total_quota = sum(journal_targets.values())
        
        # 允许最多超出 3 篇，如果超出过多则从“多分配”的期刊回收
        if total_quota > target_count + 3:
            excess = total_quota - (target_count + 3)
            # 按 (quota - raw_quota) 从大到小排序，从“多分配”的期刊回收
            journals_by_excess = sorted(
                journals_sorted,
                key=lambda j: journal_targets[j["journal"]]
                - raw_quotas.get(j["journal"], 0.0),
                reverse=True,
            )
            for j in journals_by_excess:
                if excess <= 0:
                    break
                journal = j["journal"]
                current_quota = journal_targets[journal]
                if current_quota > 0:
                    journal_targets[journal] = current_quota - 1
                    excess -= 1

        # 可选：打印每期刊 quota，用于调试 / 日志
        if show_journal_quota:
            print(f"\n  {rank} rank 每期刊 quota 分配:")
            print(f"    {'期刊':<50} {'原始文章数':<12} {'分配quota':<12}")
            print(f"    {'-'*50} {'-'*12} {'-'*12}")
            for j in journals_sorted:
                journal = j["journal"]
                article_count = j["article_count"]
                quota = journal_targets.get(journal, 0)
                journal_display = (
                    journal[:47] + "..." if len(journal) > 50 else journal
                )
                print(
                    f"    {journal_display:<50} {article_count:<12} {quota:<12}"
                )
            print(
                f"    总计: {sum(journal_targets.values())} 篇 "
                f"(目标: {target_count} 篇, 允许 +3 误差)"
            )

        # 从每本期刊中选择文章，并收集每年的quota和原始数量信息
        selected_ids: List[str] = []
        journal_year_info: Dict[str, Dict[int, Dict[str, int]]] = {}  # journal -> year -> {original, quota, selected}
        
        for journal, target_for_journal in journal_targets.items():
            articles = await Article.find({
                "subject": subject,
                "type": "study",
                "rank": rank,
                "journal": journal,
                "abort": {"$ne": True}
            }).to_list()
            
            journal_selected, year_info = await allocate_articles_by_year(
                articles, target_for_journal, return_year_info=True
                )
            selected_ids.extend(journal_selected)
            if year_info:
                journal_year_info[journal] = year_info
        
        # 检查总数：由于每期刊每年最多超出1篇，理论上最多可能超出 期刊数*年份数
        # 但为了尽量接近目标，如果超出太多，随机减少
        max_allowed = target_count + len(journals_sorted)  # 每期刊最多超出1篇（假设平均1个年份）
        if len(selected_ids) > max_allowed:
            selected_ids = random.sample(selected_ids, max_allowed)

        selected_article_ids[rank] = selected_ids
        journal_year_info_all[rank] = journal_year_info

    return selected_article_ids, journal_year_info_all


async def get_rank_statistics(subject: str) -> Dict[str, Dict[str, Any]]:
    """获取每个 rank 的统计信息"""
    pipeline = [
        {"$match": {
            "subject": subject,
            "type": "study",
            "journal": {"$ne": None}
        }},
        {
            "$group": {
                "_id": {"rank": "$rank", "journal": "$journal"},
                "article_count": {"$sum": 1}
            }
        }
    ]
    
    results = await Article.aggregate(pipeline).to_list()
    
    rank_stats = {}
    for result in results:
        rank = result['_id']['rank']
        journal = result['_id']['journal']
        article_count = result['article_count']

        if rank not in rank_stats:
        rank_stats[rank] = {
                'journals': [],
                'total_articles': 0,
                'journal_count': 0
            }

        rank_stats[rank]['journals'].append({
            'journal': journal,
            'article_count': article_count
        })
        rank_stats[rank]['total_articles'] += article_count
        rank_stats[rank]['journal_count'] += 1

    for rank in rank_stats:
        stats = rank_stats[rank]
        stats['avg_articles_per_journal'] = stats['total_articles'] / stats['journal_count'] if stats['journal_count'] > 0 else 0
    
    return rank_stats


async def filter_journals(
    subject: Optional[str] = None,
    connection_string: Optional[str] = None,
    db_name: Optional[str] = None,
    seed: Optional[int] = None,
    dry_run: bool = True,
    target_articles: Optional[Dict[str, int]] = None,
    subject_targets_lookup: Optional[Dict[str, Dict[str, int]]] = None
):
    """
    筛选期刊和文章
    
    Args:
        subject: 学科名称，如果为None则遍历所有学科
        connection_string: MongoDB连接字符串
        db_name: 数据库名称
        seed: 随机种子
        dry_run: 是否为试运行
        target_articles: 目标文章数量
        subject_targets_lookup: 学科到目标文章数量的查找字典，用于处理多个学科时自动查找
    """
    # 加载配置
    if connection_string is None or db_name is None:
        config_conn_str, config_db_name = load_database_config()
        if connection_string is None:
            connection_string = config_conn_str
        if db_name is None:
            db_name = config_db_name
    
    if '<DBNAME>' in connection_string:
        connection_string = connection_string.replace('<DBNAME>', db_name)
    
    print(f"连接数据库: {connection_string}")
    print(f"数据库名称: {db_name}")
    if seed is not None:
        random.seed(seed)
        print(f"随机种子: {seed}")
    print()
    
    # 初始化数据库连接
    client = AsyncIOMotorClient(connection_string)
    database = client[db_name]
    await init_beanie(database=database, document_models=[Article])

    # 如果没有指定学科，获取所有学科
    subjects_to_process = []
    if subject is None or subject == "":
        print("未指定学科，将遍历所有学科...")
        all_subjects = await Article.distinct("subject", {"type": "study"})
        subjects_to_process = [s for s in all_subjects if s]
        print(f"找到 {len(subjects_to_process)} 个学科: {', '.join(subjects_to_process)}\n")
    else:
        subjects_to_process = [subject]

    # 全局统计：所有学科合计
    global_total_articles = 0
    global_kept_articles = 0

    # 遍历每个学科
    for current_subject in subjects_to_process:
        print(f"\n{'='*80}")
        print(f"处理学科: {current_subject}")
        print(f"{'='*80}\n")
    
    # 获取统计信息
    print("正在获取统计信息...")
        rank_stats = await get_rank_statistics(current_subject)
    
    if 'exceptional' not in rank_stats:
            print(f"错误: 学科 {current_subject} 未找到 exceptional rank，跳过")
            continue
    
    exceptional_count = rank_stats['exceptional']['journal_count']
    print(f"\n基准: exceptional rank 有 {exceptional_count} 个期刊")
    print("\n各 rank 统计信息:")
    print(f"  {'Rank':<15} {'期刊数':<10} {'文章总数':<12} {'平均每期刊':<15}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*15}")
    
    for rank in ['exceptional', 'strong', 'fair', 'limited']:
        if rank in rank_stats:
            stats = rank_stats[rank]
            print(f"  {rank:<15} {stats['journal_count']:<10} {stats['total_articles']:<12} {stats['avg_articles_per_journal']:<15.2f}")
    
        # 确定当前学科的目标文章数量
        current_target_articles = target_articles
        if not current_target_articles and subject_targets_lookup and current_subject in subject_targets_lookup:
            current_target_articles = subject_targets_lookup[current_subject]
            print(f"\n使用预定义的目标文章数量: {current_target_articles}")

        if not current_target_articles:
            print("\n错误: 必须指定目标文章数量")
            continue

        print("\n根据目标文章数量筛选文章...")
        print(f"\n目标文章数量:")
        print(f"  {'Rank':<15} {'目标文章数':<15}")
        print(f"  {'-'*15} {'-'*15}")
        for rank in ['exceptional', 'strong', 'fair', 'limited']:
            if rank in current_target_articles:
                print(f"  {rank:<15} {current_target_articles[rank]:<15}")
        
        # 筛选文章
        selected_article_ids, journal_year_info_all = await filter_articles_by_target(
            current_subject, current_target_articles, rank_stats, connection_string, db_name
        )
        
        # 显示选中的文章统计
        print("\n选中的文章统计:")
        for rank in ['exceptional', 'strong', 'fair', 'limited']:
            if rank not in selected_article_ids:
                continue
            
            article_ids = selected_article_ids[rank]
            if not article_ids:
                continue
            
            # 将字符串ID转换为ObjectId进行查询
            try:
                object_ids = [ObjectId(article_id) for article_id in article_ids]
            except Exception:
                # 如果转换失败，尝试直接使用字符串
                object_ids = article_ids
            
            articles = await Article.find({
                "_id": {"$in": object_ids}
            }).to_list()
            
            # 统计年份分布
            year_counts = {}
            journal_counts = {}
            for article in articles:
                year = article.published_year if article.published_year else "未知"
                year_counts[year] = year_counts.get(year, 0) + 1
                journal = article.journal if article.journal else "未知"
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
            
            # 打印总体保留数量与目标对比
            actual_count = len(article_ids)
            target_for_rank = None
            if current_target_articles and rank in current_target_articles:
                target_for_rank = current_target_articles[rank]

            print(f"\n  {rank} rank: {actual_count} 篇文章")
            if target_for_rank is not None:
                diff = actual_count - target_for_rank
                print(f"    目标: {target_for_rank} 篇, 实际保留: {actual_count} 篇, 差值: {diff:+d}")

            # 查询该rank下所有文章的年份分布（原始数量）
            all_articles_pipeline = [
                {"$match": {
                    "subject": current_subject,
                    "type": "study",
                    "rank": rank,
                    "journal": {"$ne": None},
                    "abort": {"$ne": True}
                }},
                {"$group": {
                    "_id": "$published_year",
                    "count": {"$sum": 1}
                }}
            ]
            all_articles_by_year = await Article.aggregate(all_articles_pipeline).to_list()
            original_year_counts = {}
            for item in all_articles_by_year:
                year = item['_id'] if item['_id'] is not None else 2025  # 默认2025
                original_year_counts[year] = item['count']

            # 计算每年的quota（基于目标数量和权重分配）
            year_quota_calculated = {}
            if target_for_rank is not None and original_year_counts:
                current_year = 2025
                all_years_sorted = sorted([y for y in original_year_counts.keys() if isinstance(y, int)], reverse=True)
                
                if len(all_years_sorted) > 1:
                    # 使用二分查找计算decay factor和quota
                    def calc_quota_allocation(decay_factor):
                        year_weights = {}
                        for year in all_years_sorted:
                            year_diff = current_year - year
                            if year == current_year:
                                year_weights[year] = 1.0
    else:
                                year_weights[year] = decay_factor ** year_diff
                        
                        total_weight = sum(year_weights.values())
                        if total_weight == 0:
                            return 0
                        
                        allocated = 0
                        for year in all_years_sorted:
                            weight = year_weights[year]
                            quota = int(target_for_rank * weight / total_weight)
                            quota = min(quota, original_year_counts[year])
                            allocated += quota
                        return allocated
                    
                    # 二分查找
                    left, right = 0.5, 1.0
                    best_decay = 0.9
                    tolerance = 1
                    total_articles = sum(original_year_counts.values())
                    
                    if total_articles <= target_for_rank:
                        best_decay = 1.0
                    else:
                        for _ in range(50):
                            mid = (left + right) / 2
                            allocated = calc_quota_allocation(mid)
                            if allocated < target_for_rank:
                                left = mid
                            elif allocated - target_for_rank > tolerance:
                                right = mid
                            else:
                                best_decay = mid
                                break
                            best_decay = mid
                    
                    # 计算每年的quota
                    year_weights = {}
                    for year in all_years_sorted:
                        year_diff = current_year - year
                        if year == current_year:
                            year_weights[year] = 1.0
                        else:
                            year_weights[year] = best_decay ** year_diff
                    
                    total_weight = sum(year_weights.values())
                    allocated = 0
                    for year in all_years_sorted:
                        weight = year_weights[year]
                        quota = int(target_for_rank * weight / total_weight)
                        quota = min(quota, original_year_counts[year])
                        year_quota_calculated[year] = quota
                        allocated += quota
                    
                    # 补充剩余quota
                    if allocated < target_for_rank:
                        remaining = target_for_rank - allocated
                        for year in all_years_sorted:
                            if remaining <= 0:
                                break
                            current_quota = year_quota_calculated.get(year, 0)
                            available = original_year_counts[year] - current_quota
                            add_quota = min(remaining, available)
                            year_quota_calculated[year] = current_quota + add_quota
                            remaining -= add_quota
            else:
                    # 只有一年
                    year = all_years_sorted[0] if all_years_sorted else current_year
                    year_quota_calculated[year] = min(target_for_rank, original_year_counts.get(year, 0))

            # 打印年份分布（显示原始数量、quota、实际保留数量）
            if year_counts:
                print(f"    年份分布:")
                sorted_years = sorted(
                    year_counts.items(),
                    key=lambda x: x[0] if isinstance(x[0], int) else 0,
                    reverse=True
                )
                # 显示前5年或所有年份
                for year, kept_count in sorted_years[:5]:
                    original_count = original_year_counts.get(year, 0)
                    quota = year_quota_calculated.get(year, 0)
                    print(f"      {year}: 原始={original_count} 篇, quota={quota} 篇, 实际保留={kept_count} 篇")

                # 检查 2025 >= 2024 >= 2023 是否满足，给出提示
                key_years = [2025, 2024, 2023]
                if all((y in year_counts and isinstance(y, int)) for y in key_years):
                    c2025 = year_counts[2025]
                    c2024 = year_counts[2024]
                    c2023 = year_counts[2023]
                    if not (c2025 >= c2024 >= c2023):
                        print(
                            f"    [提示] 实际年份分布未满足 2025≥2024≥2023: "
                            f"2025={c2025}, 2024={c2024}, 2023={c2023}"
                        )

            print(f"    涉及期刊数: {len(journal_counts)}")
            
            # 显示每期刊每年的quota和原始数量
            if rank in journal_year_info_all:
                print(f"\n    每期刊每年的quota和原始数量:")
                journal_info = journal_year_info_all[rank]
                for journal, year_info in sorted(journal_info.items()):
                    journal_display = journal[:60] + "..." if len(journal) > 63 else journal
                    print(f"      {journal_display}:")
                    for year in sorted(year_info.keys(), reverse=True):
                        info = year_info[year]
                        print(f"        {year}: 原始={info['original']} 篇, quota={info['quota']} 篇, 实际选中={info['selected']} 篇")
    
        # 计算统计影响（当前学科）
    print("\n统计影响:")
        
        # 统计该学科所有 type='study' 的文章（包括没有journal的）
        total_study_in_subject = await Article.find({
            "subject": current_subject,
            "type": "study"
        }).count()
        
        # 统计该学科有journal且在目标rank的文章
        total_articles = 0
        kept_articles = 0
        
        for rank in ['exceptional', 'strong', 'fair', 'limited']:
            if rank not in rank_stats:
                continue
            rank_total = rank_stats[rank]['total_articles']
            total_articles += rank_total
            if rank in selected_article_ids:
                kept_articles += len(selected_article_ids[rank])
        
        print(f"  该学科所有type='study'的文章数: {total_study_in_subject}")
        print(f"  有journal且在目标rank的文章数: {total_articles}")
        print(f"  保留文章数: {kept_articles}")
        print(f"  删除文章数（有journal且在目标rank的）: {total_articles - kept_articles}")
        if total_articles > 0:
            print(f"  保留比例（有journal且在目标rank的）: {kept_articles / total_articles * 100:.2f}%")
        if total_study_in_subject > total_articles:
            no_journal_count = total_study_in_subject - total_articles
            print(f"  没有journal的文章数: {no_journal_count}")

        # 累计到全局统计
        global_total_articles += total_articles
        global_kept_articles += kept_articles
    
    # 执行筛选
    if dry_run:
        print("\n[试运行模式] 不会实际更新数据库")
        print("使用 --no-dry-run 参数来实际执行筛选")
    else:
        print("\n开始更新数据库...")
            kept_count = 0
            aborted_count = 0
            
            for rank in ['exceptional', 'strong', 'fair', 'limited']:
                if rank not in selected_article_ids:
                    result_aborted = await Article.find({
                        "subject": current_subject,
                        "type": "study",
                        "rank": rank,
                        "journal": {"$ne": None}
                    }).update_many({"$set": {"abort": True}})
                    aborted_count += result_aborted.modified_count
                    continue
                
                article_ids = selected_article_ids[rank]
                if not article_ids:
                    continue
                
                # 将字符串ID转换为ObjectId进行查询
                try:
                    object_ids = [ObjectId(article_id) for article_id in article_ids]
                except Exception:
                    # 如果转换失败，尝试直接使用字符串
                    object_ids = article_ids
                
                await Article.find({
                    "_id": {"$in": object_ids}
                }).update_many({"$unset": {"abort": ""}})
                kept_count += len(article_ids)
                
                result_aborted = await Article.find({
                    "subject": current_subject,
                    "type": "study",
                    "rank": rank,
                    "journal": {"$ne": None},
                    "_id": {"$nin": object_ids}
                }).update_many({"$set": {"abort": True}})
                aborted_count += result_aborted.modified_count
            
            print(f"  保留文章数: {kept_count} 篇")
            print(f"  已标记 abort=True: {aborted_count} 篇文章")
            print("\n筛选完成！")

    # 所有学科处理完成后，输出全局总计
    if subjects_to_process:
        # 查询数据库中所有 type='study' 的文章总数
        total_study_articles = await Article.find({"type": "study"}).count()
        
        # 统计没有journal的文章数
        no_journal_count = await Article.find({
                    "type": "study",
            "$or": [
                {"journal": None},
                {"journal": {"$exists": False}}
            ]
                }).count()
                
        # 统计有journal但rank不在目标范围内的文章数
        other_rank_count = await Article.find({
                    "type": "study",
            "journal": {"$ne": None},
            "rank": {"$nin": ["exceptional", "strong", "fair", "limited"]}
        }).count()
        
        # 统计已处理学科中有journal且在目标rank的文章数
        processed_subjects_with_journal = await Article.find({
            "type": "study",
            "subject": {"$in": subjects_to_process},
            "journal": {"$ne": None},
            "rank": {"$in": ["exceptional", "strong", "fair", "limited"]}
        }).count()
        
        # 统计未处理学科的文章数
        all_subjects = await Article.distinct("subject", {"type": "study"})
        unprocessed_subjects = [s for s in all_subjects if s and s not in subjects_to_process]
        unprocessed_count = await Article.find({
            "type": "study",
            "subject": {"$in": unprocessed_subjects}
        }).count() if unprocessed_subjects else 0
        
        print(f"\n{'='*80}")
        print("所有已处理学科的总体统计:")
        print(f"  数据库中所有type='study'的文章数: {total_study_articles}")
        print(f"  ──────────────────────────────────────────────")
        print(f"  已处理学科中有journal且在目标rank的文章数: {global_total_articles}")
        print(f"  保留文章数: {global_kept_articles}")
        print(f"  删除文章数（已处理学科中有journal且在目标rank的）: {global_total_articles - global_kept_articles}")
        if global_total_articles > 0:
            print(f"  保留比例（已处理学科中有journal且在目标rank的）: {global_kept_articles / global_total_articles * 100:.2f}%")
        print(f"  ──────────────────────────────────────────────")
        print(f"  未统计的文章分布:")
        print(f"    没有journal的文章数: {no_journal_count}")
        print(f"    有journal但rank不在目标范围的: {other_rank_count}")
        print(f"    未处理学科的文章数: {unprocessed_count}")
        if unprocessed_subjects:
            print(f"    未处理学科: {', '.join(unprocessed_subjects[:10])}{'...' if len(unprocessed_subjects) > 10 else ''}")
        print(f"    其他（差异）: {total_study_articles - global_total_articles - no_journal_count - other_rank_count - unprocessed_count}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    import json
    
    # 预定义的学科目标文章数量
    SUBJECT_TARGETS = {
        "ECONOMICS": {
            "exceptional": 1500,
            "strong": 2000,
            "fair": 1500,
            "limited": 1300
        },
        "BUSINESS, FINANCE": {
            "exceptional": 1400,
            "strong": 1600,
            "fair": 1300,
            "limited": 1000
        },
        "SOCIOLOGY": {
            "exceptional": 1300,
            "strong": 1600,
            "fair": 1100,
            "limited": 1100
        },
        "PSYCHOLOGY, MULTIDISCIPLINARY": {
            "exceptional": 900,
            "strong": 900,
            "fair": 1000,
            "limited": 1000
        },
        "POLITICAL SCIENCE": {
            "exceptional": 1000,
            "strong": 1200,
            "fair": 800,
            "limited": 700
        },
        "COMMUNICATION": {
            "exceptional": 383,
            "strong": 1000,
            "fair": 800,
            "limited": 817
        },
        "PUBLIC ADMINISTRATION": {
            "exceptional": 700,
            "strong": 750,
            "fair": 700,
            "limited": 650
        }
    }
    
    parser = argparse.ArgumentParser(description='筛选期刊和文章')
    parser.add_argument('--subject', help='学科名称（不指定则遍历所有学科）')
    parser.add_argument('--connection-string', help='MongoDB连接字符串')
    parser.add_argument('--db-name', help='数据库名称')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--no-dry-run', action='store_true', help='实际执行筛选（默认为试运行）')
    parser.add_argument('--target-articles', help='目标文章数量JSON字符串')
    
    args = parser.parse_args()
    
    # 解析目标文章数量
    target_articles = None
    if args.target_articles:
        try:
            target_articles = json.loads(args.target_articles)
        except json.JSONDecodeError:
            print("错误: 无法解析目标文章数量JSON")
            sys.exit(1)
    elif args.subject and args.subject in SUBJECT_TARGETS:
        target_articles = SUBJECT_TARGETS[args.subject]
        print(f"使用预定义的目标文章数量: {target_articles}")
    
    asyncio.run(filter_journals(
        subject=args.subject,
        connection_string=args.connection_string,
        db_name=args.db_name,
        seed=args.seed,
        dry_run=not args.no_dry_run,
        target_articles=target_articles,
        subject_targets_lookup=SUBJECT_TARGETS
    ))
