# file: agents/cognition/graph_storage.py
import json
import os
import networkx as nx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeCapsule:
    """
    知识胶囊：缓存期的宏观共识，避免频繁查询大模型。
    带有生命周期（生效时间和失效时间）。
    """
    topic: str
    content: str
    valid_at: float
    invalid_at: float
    
    def is_valid(self, current_time: float) -> bool:
        return self.valid_at <= current_time <= self.invalid_at


class GraphMemoryBank:
    """
    基于 NetworkX 维护的 Agent 个人知识图谱与胶囊缓存。
    """
    def __init__(self, agent_id: str, storage_dir: str = "data/graphs"):
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.graph = nx.DiGraph()
        self.capsules: Dict[str, KnowledgeCapsule] = {}
        
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
            
        self.file_path = os.path.join(self.storage_dir, f"graph_{self.agent_id}.graphml")
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.file_path):
            try:
                self.graph = nx.read_graphml(self.file_path)
            except Exception as e:
                logger.error(f"Failed to load graph for {self.agent_id}: {e}")
                self.graph = nx.DiGraph()

    def _save_graph(self):
        try:
            nx.write_graphml(self.graph, self.file_path)
        except Exception as e:
            logger.error(f"Failed to save graph for {self.agent_id}: {e}")

    def add_triplet(self, subject: str, predicate: str, object_: str, weight: float = 1.0):
        """添加或更新三元组关系"""
        if self.graph.has_edge(subject, object_):
            # 累加权重并更新谓词
            old_weight = self.graph[subject][object_].get('weight', 1.0)
            self.graph[subject][object_]['weight'] = old_weight + weight
            self.graph[subject][object_]['predicate'] = predicate
        else:
            self.graph.add_edge(subject, object_, predicate=predicate, weight=weight)
        
        self._save_graph()

    def add_capsule(self, topic: str, content: str, current_time: float, ttl_seconds: float = 3600):
        """添加带有生存周期的知识胶囊"""
        capsule = KnowledgeCapsule(
            topic=topic,
            content=content,
            valid_at=current_time,
            invalid_at=current_time + ttl_seconds
        )
        self.capsules[topic] = capsule

    def get_valid_capsules(self, current_time: float) -> List[str]:
        """获取当前依然存活的知识胶囊内容列表"""
        valid_contents = []
        expired_topics = []
        for topic, capsule in self.capsules.items():
            if capsule.is_valid(current_time):
                valid_contents.append(f"[{topic}] {capsule.content}")
            else:
                expired_topics.append(topic)
                
        # 顺便清理过期缓存
        for topic in expired_topics:
            del self.capsules[topic]
            
        return valid_contents

    def retrieve_subgraph(self, keywords: List[str], depth: int = 1, max_nodes: int = 20) -> str:
        """
        检索关键词的 N跳子图
        Returns: 拼接成文本的三元组描述
        """
        if not self.graph.nodes:
            return ""

        subgraph_nodes = set()
        for keyword in keywords:
            for node in self.graph.nodes:
                if keyword.lower() in str(node).lower():
                    # BFS 获取邻居节点
                    edges = nx.bfs_edges(self.graph, source=node, depth_limit=depth)
                    subgraph_nodes.add(node)
                    for u, v in edges:
                        subgraph_nodes.add(u)
                        subgraph_nodes.add(v)
                        if len(subgraph_nodes) >= max_nodes:
                            break
            if len(subgraph_nodes) >= max_nodes:
                break
                
        if not subgraph_nodes:
            return ""
            
        sub = self.graph.subgraph(subgraph_nodes)
        triplets = []
        for u, v, data in sub.edges(data=True):
            predicate = data.get('predicate', '相关联')
            triplets.append(f"({u} -> {predicate} -> {v})")
            
        return "; ".join(triplets)
