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
    description="训练 Drain 模型，支持读取本地日志或可下载的日志文件",
)
def train_file(
    file_url: Annotated[str, "日志文件的绝对路径或可直接访问的 URL"]
) -> Dict[str, Any]:
    """
    Train the Drain model on a file from a URL or local path.
    
    Args:
        file_url: URL or local file path to train on
        
    Returns:
        Dictionary with training statistics
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
    description="训练 Drain 模型中的单条日志，便于增量更新",
)
def train_line(
    line: Annotated[str, "需要被学习的日志原文"]
) -> Dict[str, Any]:
    """
    Train the Drain model on a single line.
    
    Args:
        line: Log line to train on
        
    Returns:
        Dictionary with training result
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
    description="对单条日志执行模板匹配并返回模板与参数",
)
def inference_line(
    line: Annotated[str, "需要推理的日志原文"]
) -> Dict[str, Any]:
    """
    Perform inference on a single line using the trained Drain model.
    
    Args:
        line: Log line to infer
        
    Returns:
        Dictionary with inference result
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
    description="对日志文件进行模板推理，统计匹配情况并列出结果",
)
def inference_file(
    file_url: Annotated[str, "需要推理的日志文件绝对路径或可访问的 URL"]
) -> Dict[str, Any]:
    """
    Perform inference on a file from a URL or local path.
    
    Args:
        file_url: URL or local file path to infer on
        
    Returns:
        Dictionary with inference statistics
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
    description="查看 Drain 模型当前的聚类数量与累计解析的消息数",
)
def model_stats() -> Dict[str, Any]:
    """
    获取 Drain 模型的整体统计信息。
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
    description="分页查看 Drain 模型中的 cluster 列表及其统计信息",
)
def list_clusters(
    page: Annotated[int, "页码，从 1 开始"] = 1,
    page_size: Annotated[int, "每页返回的 cluster 数量"] = 20
) -> Dict[str, Any]:
    """
    分页列出 Drain 模型的 cluster 信息。
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
    description="新增日志解析的 masking 规则",
)
def add_masking(
    regex_pattern: Annotated[str, "需要被替换的正则表达式模式"],
    mask_with: Annotated[str, "用于替换匹配文本的占位符"]
) -> Dict[str, Any]:
    """
    Add a masking instruction to the Drain model.
    Note: Masking changes require reinitializing the model, which will preserve trained clusters.
    
    Args:
        regex_pattern: Regular expression pattern to mask
        mask_with: String to replace matched patterns with
        
    Returns:
        Dictionary with operation result
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
    description="移除已存在的 masking 规则",
)
def remove_masking(
    regex_pattern: Annotated[str, "需要删除的正则表达式模式"]
) -> Dict[str, Any]:
    """
    Remove a masking instruction from the Drain model.
    Note: Masking changes require reinitializing the model, which will preserve trained clusters.
    
    Args:
        regex_pattern: Regular expression pattern to remove
        
    Returns:
        Dictionary with operation result
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
    description="列出当前已配置的所有 masking 规则",
)
def list_masking() -> Dict[str, Any]:
    """
    List all masking instructions in the Drain model.
    
    Returns:
        Dictionary with list of masking instructions
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
        port=8000,
        path="/mcp"
    )