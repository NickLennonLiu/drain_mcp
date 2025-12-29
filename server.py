import json
import logging
import os
import tempfile
import urllib.request
from os.path import dirname, join
from typing import List, Dict, Any, Optional, Annotated

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from drain3.masking import MaskingInstruction
from fastmcp import FastMCP

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 Drain 模型
_config_path = join(dirname(__file__), "drain_demos", "drain3.ini")
_state_file = join(dirname(__file__), "drain3_state.bin")
_config = TemplateMinerConfig()
_config.load(_config_path)
_config.profiling_enabled = False

# 使用文件持久化保存模型状态
_persistence = FilePersistence(_state_file)
_template_miner = TemplateMiner(_persistence, config=_config)

mcp = FastMCP("DrainMCPServer")

logger.info(f"Drain3 MCP Server initialized with {len(_config.masking_instructions)} masking instructions")


def _iter_clusters():
    """遍历当前 Drain 模型中的所有 cluster"""
    clusters = getattr(_template_miner.drain, "clusters", None)
    if clusters is None:
        return []
    if hasattr(clusters, "values"):
        return list(clusters.values())
    try:
        return list(clusters)
    except TypeError:
        return []


def _get_cluster_by_id(cluster_id: Optional[str]):
    """辅助函数：根据 cluster_id 获取 cluster 对象"""
    if cluster_id is None:
        return None
    clusters = getattr(_template_miner.drain, "clusters", None)
    if clusters is None:
        return None
    if hasattr(clusters, "get"):
        return clusters.get(cluster_id)
    try:
        for cluster in clusters:
            if getattr(cluster, "cluster_id", None) == cluster_id:
                return cluster
    except TypeError:
        return None
    return None


def _get_cluster_size(cluster_id: Optional[str]) -> Optional[int]:
    cluster = _get_cluster_by_id(cluster_id)
    if cluster is None:
        return None
    return getattr(cluster, "size", None)


def _download_file(url: str) -> str:
    """下载文件到临时目录并返回本地路径"""
    if url.startswith(("http://", "https://")):
        # 下载远程文件
        tmp_path = tempfile.mktemp(suffix='.log')
        try:
            urllib.request.urlretrieve(url, tmp_path)
            return tmp_path
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise Exception(f"Failed to download file from {url}: {str(e)}")
    elif os.path.isfile(url):
        # 本地文件路径
        return url
    else:
        raise Exception(f"File not found: {url}")


@mcp.tool(
    name="train_file",
    description="训练 Drain 模型，支持读取本地日志文件或通过 HTTP/HTTPS 下载的远程日志文件。该工具会逐行处理日志，自动提取日志模板并建立聚类。训练过程中会跳过空行，模型状态会自动持久化保存。",
)
def train_file(
    file_url: Annotated[str, "日志文件的绝对路径（如 /path/to/logfile.log）或可直接访问的 HTTP/HTTPS URL（如 https://example.com/logs/app.log）。支持 UTF-8 编码的文本文件。"]
) -> Dict[str, Any]:
    """
    训练 Drain 模型，从文件或 URL 读取日志数据进行批量训练。
    
    该工具会逐行读取日志文件，使用 Drain3 算法自动识别日志模板模式，将相似的日志归类到同一个 cluster 中。
    训练过程中会跳过空行，模型状态会自动保存到持久化存储中。
    
    Args:
        file_url: 日志文件的绝对路径或 HTTP/HTTPS URL。本地文件路径必须是绝对路径。
                  远程文件会被下载到临时目录，处理完成后自动删除。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - lines_processed: 成功处理的日志行数（不包括空行）
        - clusters_before: 训练前的 cluster 总数
        - clusters_after: 训练后的 cluster 总数
        - new_clusters: 本次训练新增的 cluster 数量
        - cluster_message_counts: 本次训练中涉及的 cluster 及其消息数量列表
        - error: 如果 status 为 "error"，则包含错误信息
        
    示例:
        train_file("/var/log/app.log")  # 本地文件
        train_file("https://example.com/logs/app.log")  # 远程文件
    """
    global _template_miner
    try:
        file_path = _download_file(file_url)
        line_count = 0
        clusters_before = len(_template_miner.drain.clusters)
        touched_clusters = set()
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip()
                if line:  # 跳过空行
                    result = _template_miner.add_log_message(line)
                    cluster_id = result.get("cluster_id")
                    if cluster_id is not None:
                        touched_clusters.add(cluster_id)
                    line_count += 1
        
        # 如果是临时文件，删除它
        if file_path != file_url and file_path.startswith(tempfile.gettempdir()):
            os.unlink(file_path)
        
        clusters_after = len(_template_miner.drain.clusters)
        cluster_message_counts = [
            {
                "cluster_id": cid,
                "message_count": _get_cluster_size(cid)
            }
            for cid in touched_clusters
        ]
        
        return {
            "status": "success",
            "lines_processed": line_count,
            "clusters_before": clusters_before,
            "clusters_after": clusters_after,
            "new_clusters": clusters_after - clusters_before,
            "cluster_message_counts": cluster_message_counts
        }
    except Exception as e:
        logger.error(f"Error training file: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="train_line",
    description="对单条日志进行增量训练，适用于实时日志流处理或逐条添加日志的场景。该工具会立即更新模型，将日志匹配到现有 cluster 或创建新的 cluster。模型状态会自动持久化保存。",
)
def train_line(
    line: Annotated[str, "需要被学习的单条日志原文，不应包含换行符。日志会被自动进行 masking 处理，提取模板并归类到相应的 cluster 中。"]
) -> Dict[str, Any]:
    """
    对单条日志进行增量训练，适用于实时日志流处理场景。
    
    该工具会立即处理单条日志，将其与现有 cluster 进行匹配。如果匹配成功，会更新对应 cluster 的统计信息；
    如果无法匹配，会创建新的 cluster。模型状态会自动保存到持久化存储中。
    
    Args:
        line: 单条日志原文，不应包含换行符。建议去除首尾空白字符。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - change_type: 变更类型（"cluster_created" 表示创建了新 cluster，"cluster_template_changed" 表示更新了模板，"none" 表示无变更）
        - template_mined: 提取的日志模板（变量部分会被替换为占位符）
        - parameters: 从日志中提取的参数列表（模板中的变量值）
        - cluster_id: 匹配到的或新创建的 cluster ID
        - clusters_before: 训练前的 cluster 总数
        - clusters_after: 训练后的 cluster 总数
        - cluster_size: 当前 cluster 包含的日志数量
        - error: 如果 status 为 "error"，则包含错误信息
        
    示例:
        train_line("2024-01-01 10:00:00 INFO User 12345 logged in from 192.168.1.1")
    """
    global _template_miner
    try:
        clusters_before = len(_template_miner.drain.clusters)
        result = _template_miner.add_log_message(line)
        clusters_after = len(_template_miner.drain.clusters)
        
        template = result.get("template_mined", "")
        params = _template_miner.extract_parameters(template, line) if template else []
        cluster_size = _get_cluster_size(result.get("cluster_id"))
        
        return {
            "status": "success",
            "change_type": result.get("change_type", "none"),
            "template_mined": template,
            "parameters": params,
            "cluster_id": result.get("cluster_id"),
            "clusters_before": clusters_before,
            "clusters_after": clusters_after,
            "cluster_size": cluster_size
        }
    except Exception as e:
        logger.error(f"Error training line: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="inference_line",
    description="对单条日志执行模板匹配推理，不更新模型状态。该工具会查找与日志最匹配的 cluster，返回对应的模板和提取的参数。适用于日志分类、异常检测等场景。",
)
def inference_line(
    line: Annotated[str, "需要推理的单条日志原文，不应包含换行符。工具会尝试在已训练的模型中查找匹配的 cluster，如果找到则返回模板和参数，否则返回未匹配状态。"]
) -> Dict[str, Any]:
    """
    对单条日志执行模板匹配推理，不更新模型状态。
    
    该工具使用已训练的 Drain 模型对日志进行匹配，查找最相似的 cluster。
    如果找到匹配的 cluster，会返回对应的模板和提取的参数；如果未找到匹配，说明该日志模式可能未被训练过。
    该操作是只读的，不会修改模型状态。
    
    Args:
        line: 单条日志原文，不应包含换行符。建议去除首尾空白字符。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - matched: 是否找到匹配的 cluster（布尔值）
        - cluster_id: 匹配到的 cluster ID（如果 matched 为 True）
        - template: 匹配到的日志模板（如果 matched 为 True）
        - parameters: 从日志中提取的参数列表（如果 matched 为 True）
        - cluster_size: 匹配到的 cluster 包含的日志数量（如果 matched 为 True）
        - message: 如果 matched 为 False，则包含 "No matching cluster found" 消息
        - error: 如果 status 为 "error"，则包含错误信息
        
    示例:
        inference_line("2024-01-01 10:00:00 ERROR Database connection failed")
    """
    global _template_miner
    try:
        cluster = _template_miner.match(line)
        if cluster is None:
            return {
                "status": "success",
                "matched": False,
                "message": "No matching cluster found",
                "cluster_size": None
            }
        else:
            template = cluster.get_template()
            params = _template_miner.get_parameter_list(template, line)
            cluster_size = getattr(cluster, "size", None)
            
            return {
                "status": "success",
                "matched": True,
                "cluster_id": cluster.cluster_id,
                "template": template,
                "parameters": params,
                "cluster_size": cluster_size
            }
    except Exception as e:
        logger.error(f"Error inferencing line: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="inference_file",
    description="对日志文件进行批量模板推理，统计匹配情况并返回详细结果。该工具会逐行处理日志文件，对每条日志执行匹配，统计匹配率和未匹配的日志。返回结果限制为前 100 条，适用于日志分析、异常检测等场景。",
)
def inference_file(
    file_url: Annotated[str, "需要推理的日志文件绝对路径（如 /path/to/logfile.log）或可直接访问的 HTTP/HTTPS URL（如 https://example.com/logs/app.log）。支持 UTF-8 编码的文本文件。"]
) -> Dict[str, Any]:
    """
    对日志文件进行批量模板推理，统计匹配情况并返回详细结果。
    
    该工具会逐行读取日志文件，对每条日志执行模板匹配推理，统计匹配成功的数量和未匹配的数量。
    对于匹配成功的日志，会返回对应的模板、参数和 cluster 信息；对于未匹配的日志，会标记为未匹配状态。
    该操作是只读的，不会修改模型状态。返回结果限制为前 100 条，以避免响应过大。
    
    Args:
        file_url: 日志文件的绝对路径或 HTTP/HTTPS URL。本地文件路径必须是绝对路径。
                 远程文件会被下载到临时目录，处理完成后自动删除。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - total_lines: 处理的日志总行数（不包括空行）
        - matched_count: 成功匹配的日志数量
        - unmatched_count: 未匹配的日志数量
        - match_rate: 匹配率（0.0 到 1.0 之间的浮点数）
        - results: 前 100 条日志的详细结果列表，每条包含：
          - line_number: 行号
          - line: 日志原文
          - matched: 是否匹配（布尔值）
          - cluster_id: 匹配到的 cluster ID（如果 matched 为 True）
          - template: 匹配到的日志模板（如果 matched 为 True）
          - parameters: 提取的参数列表（如果 matched 为 True）
          - cluster_size: cluster 包含的日志数量（如果 matched 为 True）
        - error: 如果 status 为 "error"，则包含错误信息
        
    示例:
        inference_file("/var/log/app.log")  # 本地文件
        inference_file("https://example.com/logs/app.log")  # 远程文件
    """
    global _template_miner
    try:
        file_path = _download_file(file_url)
        matched_count = 0
        unmatched_count = 0
        results = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip()
                if line:  # 跳过空行
                    cluster = _template_miner.match(line)
                    if cluster is None:
                        unmatched_count += 1
                        results.append({
                            "line_number": line_num,
                            "line": line,
                            "matched": False
                        })
                    else:
                        matched_count += 1
                        template = cluster.get_template()
                        params = _template_miner.get_parameter_list(template, line)
                        cluster_size = getattr(cluster, "size", None)
                        results.append({
                            "line_number": line_num,
                            "line": line,
                            "matched": True,
                            "cluster_id": cluster.cluster_id,
                            "template": template,
                            "parameters": params,
                            "cluster_size": cluster_size
                        })
        
        # 如果是临时文件，删除它
        if file_path != file_url and file_path.startswith(tempfile.gettempdir()):
            os.unlink(file_path)
        
        return {
            "status": "success",
            "total_lines": matched_count + unmatched_count,
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "match_rate": matched_count / (matched_count + unmatched_count) if (matched_count + unmatched_count) > 0 else 0,
            "results": results[:100]  # 限制返回前100条结果
        }
    except Exception as e:
        logger.error(f"Error inferencing file: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="model_stats",
    description="获取 Drain 模型的整体统计信息，包括当前聚类（cluster）总数和累计处理的日志消息总数。该工具用于快速了解模型的训练状态和规模，不修改模型状态。",
)
def model_stats() -> Dict[str, Any]:
    """
    获取 Drain 模型的整体统计信息。
    
    该工具返回模型的整体统计信息，包括：
    - 当前聚类（cluster）总数：表示模型识别出的不同日志模板模式数量
    - 累计处理的日志消息总数：表示模型训练过程中处理过的所有日志条数
    
    该操作是只读的，不会修改模型状态。
    
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - total_clusters: 当前模型中的 cluster 总数，表示识别出的不同日志模板模式数量
        - total_messages: 累计处理的日志消息总数，表示所有 cluster 中日志数量的总和
        - error: 如果 status 为 "error"，则包含错误信息
        
    示例:
        model_stats()  # 返回 {"status": "success", "total_clusters": 150, "total_messages": 10000}
    """
    try:
        clusters = _iter_clusters()
        total_clusters = len(clusters)
        total_messages = sum(getattr(cluster, "size", 0) for cluster in clusters)
        return {
            "status": "success",
            "total_clusters": total_clusters,
            "total_messages": total_messages
        }
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="list_clusters",
    description="分页查看 Drain 模型中的 cluster 列表及其统计信息。每个 cluster 代表一个日志模板模式，包含该模式匹配的日志数量和提取的模板。支持分页查询，便于浏览大量 cluster。",
)
def list_clusters(
    page: Annotated[int, "页码，从 1 开始。例如 page=1 表示第一页，page=2 表示第二页。"] = 1,
    page_size: Annotated[int, "每页返回的 cluster 数量，默认为 20。建议设置为 10-50 之间的值，避免单次返回数据过多。"] = 20
) -> Dict[str, Any]:
    """
    分页列出 Drain 模型的 cluster 信息。
    
    该工具按 cluster_id 排序后分页返回 cluster 列表。每个 cluster 代表一个日志模板模式，
    包含该模式匹配的日志数量（size）和提取的模板（template）。
    支持分页查询，便于浏览大量 cluster。
    
    Args:
        page: 页码，从 1 开始。例如 page=1 表示第一页。
        page_size: 每页返回的 cluster 数量，默认为 20。建议设置为 10-50 之间的值。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - page: 当前页码
        - page_size: 每页数量
        - total_clusters: 模型中的 cluster 总数
        - total_pages: 总页数（根据 total_clusters 和 page_size 计算）
        - clusters: 当前页的 cluster 列表，每个 cluster 包含：
          - cluster_id: cluster 的唯一标识符
          - size: 该 cluster 包含的日志数量
          - template: 该 cluster 对应的日志模板（变量部分被替换为占位符）
        - error: 如果 status 为 "error"，则包含错误信息（例如 "page 与 page_size 需要为正整数"）
        
    示例:
        list_clusters(page=1, page_size=10)  # 获取第一页，每页 10 条
        list_clusters(page=2, page_size=20)  # 获取第二页，每页 20 条
    """
    if page <= 0 or page_size <= 0:
        return {
            "status": "error",
            "error": "page 与 page_size 需要为正整数"
        }
    try:
        clusters = sorted(
            _iter_clusters(),
            key=lambda c: str(getattr(c, "cluster_id", ""))
        )
        total_clusters = len(clusters)
        total_pages = (total_clusters + page_size - 1) // page_size if page_size else 0
        start = (page - 1) * page_size
        end = start + page_size
        page_clusters = clusters[start:end]
        
        cluster_items = [
            {
                "cluster_id": getattr(cluster, "cluster_id", None),
                "size": getattr(cluster, "size", 0),
                "template": cluster.get_template() if hasattr(cluster, "get_template") else None
            }
            for cluster in page_clusters
        ]
        
        return {
            "status": "success",
            "page": page,
            "page_size": page_size,
            "total_clusters": total_clusters,
            "total_pages": total_pages,
            "clusters": cluster_items
        }
    except Exception as e:
        logger.error(f"Error listing clusters: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="add_masking",
    description="新增日志解析的 masking（掩码）规则。Masking 用于在日志模板提取前将特定模式（如 IP 地址、时间戳、数字等）替换为占位符，提高模板匹配的准确性。添加规则后模型会重新初始化，但已训练的 cluster 会被保留。",
)
def add_masking(
    regex_pattern: Annotated[str, "需要被替换的正则表达式模式。例如 '\\d+\\.\\d+\\.\\d+\\.\\d+' 用于匹配 IP 地址，'\\d{4}-\\d{2}-\\d{2}' 用于匹配日期。注意需要正确转义特殊字符。"],
    mask_with: Annotated[str, "用于替换匹配文本的占位符字符串。例如 '<IP>'、'<NUM>'、'<DATE>' 等。建议使用有意义的占位符名称，便于理解模板。"]
) -> Dict[str, Any]:
    """
    新增日志解析的 masking（掩码）规则。
    
    Masking 是 Drain 算法中的重要机制，用于在日志模板提取前将特定模式（如 IP 地址、时间戳、数字等）
    替换为占位符，从而提高模板匹配的准确性。例如，将 "192.168.1.1" 替换为 "<IP>"，
    使得所有包含 IP 地址的日志能够匹配到同一个模板。
    
    添加 masking 规则后，模型会重新初始化以应用新规则，但已训练的 cluster 会被保留。
    如果指定的正则表达式模式已存在，操作会失败并返回错误。
    
    Args:
        regex_pattern: 需要被替换的正则表达式模式。例如：
                       - '\\d+\\.\\d+\\.\\d+\\.\\d+' 用于匹配 IP 地址
                       - '\\d{4}-\\d{2}-\\d{2}' 用于匹配日期
                       - '\\d+' 用于匹配数字
                       注意需要正确转义特殊字符（如点号、括号等）。
        mask_with: 用于替换匹配文本的占位符字符串。例如：
                   - '<IP>' 用于 IP 地址
                   - '<NUM>' 用于数字
                   - '<DATE>' 用于日期
                   建议使用有意义的占位符名称，便于理解模板。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - message: 操作结果消息
        - total_masking_instructions: 当前 masking 规则总数
        - clusters_preserved: 保留的 cluster 数量（重新初始化后）
        - error: 如果 status 为 "error"，则包含错误信息（例如 "Masking pattern already exists"）
        
    示例:
        add_masking("\\d+\\.\\d+\\.\\d+\\.\\d+", "<IP>")  # 添加 IP 地址 masking
        add_masking("\\d+", "<NUM>")  # 添加数字 masking
    """
    global _template_miner
    try:
        # 检查是否已存在相同的 pattern
        for idx, instruction in enumerate(_config.masking_instructions):
            logger.info(f"Instruction {idx}: {instruction.pattern}, mask_with: {instruction.mask_with}")
            if instruction.pattern == regex_pattern:
                return {
                    "status": "error",
                    "error": f"Masking pattern already exists: {regex_pattern}"
                }
        
        # 保存当前模型状态
        _template_miner.save_state(snapshot_reason="add_masking")

        logger.info(f"Adding new instruction: {regex_pattern} -> {mask_with}")
        # 添加新的 masking 指令到配置
        # new_instruction = {"regex_pattern": regex_pattern, "mask_with": mask_with}
        new_instruction = MaskingInstruction(regex_pattern, mask_with)
        _config.masking_instructions.append(new_instruction)
        logger.info(f"New instruction: {new_instruction}")
        
        # 重新初始化 TemplateMiner 以应用新的 masking 指令
        # 状态会从持久化存储中恢复
        # global _template_miner
        _template_miner = TemplateMiner(persistence_handler=_persistence, config=_config)
        
        return {
            "status": "success",
            "message": f"Added masking instruction: {regex_pattern} -> {mask_with}",
            "total_masking_instructions": len(_config.masking_instructions),
            "clusters_preserved": len(_template_miner.drain.clusters)
        }
    except Exception as e:
        logger.error(f"Error adding masking: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="remove_masking",
    description="移除已存在的 masking（掩码）规则。删除规则后模型会重新初始化，但已训练的 cluster 会被保留。如果指定的正则表达式模式不存在，操作会失败并返回错误。",
)
def remove_masking(
    regex_pattern: Annotated[str, "需要删除的正则表达式模式，必须与添加时使用的模式完全一致。例如如果要删除 IP 地址 masking，应传入 '\\d+\\.\\d+\\.\\d+\\.\\d+'。"]
) -> Dict[str, Any]:
    """
    移除已存在的 masking（掩码）规则。
    
    该工具用于删除之前添加的 masking 规则。删除规则后，模型会重新初始化以应用变更，
    但已训练的 cluster 会被保留。如果指定的正则表达式模式不存在，操作会失败并返回错误。
    
    注意：删除 masking 规则可能会影响后续日志的模板匹配结果，因为之前被掩码的部分
    现在会被当作普通文本处理。
    
    Args:
        regex_pattern: 需要删除的正则表达式模式，必须与添加时使用的模式完全一致。
                       例如如果要删除 IP 地址 masking，应传入 '\\d+\\.\\d+\\.\\d+\\.\\d+'。
        
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - message: 操作结果消息
        - total_masking_instructions: 当前 masking 规则总数（删除后）
        - clusters_preserved: 保留的 cluster 数量（重新初始化后）
        - error: 如果 status 为 "error"，则包含错误信息（例如 "Masking pattern not found"）
        
    示例:
        remove_masking("\\d+\\.\\d+\\.\\d+\\.\\d+")  # 删除 IP 地址 masking
    """
    global _template_miner
    try:
        removed = False
        for i, instruction in enumerate(_config.masking_instructions):
            if instruction.pattern == regex_pattern:
                _config.masking_instructions.pop(i)
                removed = True
                break
        
        if not removed:
            return {
                "status": "error",
                "error": f"Masking pattern not found: {regex_pattern}"
            }
        
        # 保存当前模型状态
        _template_miner.save_state(snapshot_reason="remove_masking")
        
        # 重新初始化 TemplateMiner 以应用新的 masking 指令
        # 状态会从持久化存储中恢复
        # global _template_miner
        _template_miner = TemplateMiner(persistence_handler=_persistence, config=_config)
        
        return {
            "status": "success",
            "message": f"Removed masking instruction: {regex_pattern}",
            "total_masking_instructions": len(_config.masking_instructions),
            "clusters_preserved": len(_template_miner.drain.clusters)
        }
    except Exception as e:
        logger.error(f"Error removing masking: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="list_masking",
    description="列出当前已配置的所有 masking（掩码）规则。返回所有正则表达式模式及其对应的占位符，用于查看当前模型的 masking 配置。该操作是只读的，不修改模型状态。",
)
def list_masking() -> Dict[str, Any]:
    """
    列出当前已配置的所有 masking（掩码）规则。
    
    该工具返回模型中所有已配置的 masking 规则，包括正则表达式模式和对应的占位符。
    用于查看当前模型的 masking 配置，便于了解哪些模式会被替换为占位符。
    该操作是只读的，不会修改模型状态。
    
    Returns:
        包含以下字段的字典：
        - status: 操作状态（"success" 或 "error"）
        - count: masking 规则的总数
        - masking_instructions: masking 规则列表，每个规则包含：
          - regex_pattern: 正则表达式模式
          - mask_with: 对应的占位符字符串
        - error: 如果 status 为 "error"，则包含错误信息
        
    示例:
        list_masking()  # 返回所有 masking 规则
        # 返回示例：
        # {
        #   "status": "success",
        #   "count": 3,
        #   "masking_instructions": [
        #     {"regex_pattern": "\\d+\\.\\d+\\.\\d+\\.\\d+", "mask_with": "<IP>"},
        #     {"regex_pattern": "\\d+", "mask_with": "<NUM>"},
        #     {"regex_pattern": "\\d{4}-\\d{2}-\\d{2}", "mask_with": "<DATE>"}
        #   ]
        # }
    """
    try:
        masking_list = [
            {
                "regex_pattern": instr.pattern,
                "mask_with": instr.mask_with
            }
            for instr in _config.masking_instructions
        ]
        
        return {
            "status": "success",
            "count": len(masking_list),
            "masking_instructions": masking_list
        }
    except Exception as e:
        logger.error(f"Error listing masking: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8101,
        path="/mcp"
    )