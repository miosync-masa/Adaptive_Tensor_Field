#!/usr/bin/env python3
"""
脅威インテリジェンスブロックチェーン実装
AdaptiveTensorFieldSecurity - Threat Intelligence Chain Module
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pickle
import gzip
from enum import Enum

# === データ構造定義 ===

class ThreatType(Enum):
    """脅威タイプの定義"""
    C2_SERVER = "command_and_control"
    MALWARE_HASH = "malware"
    PHISHING_URL = "phishing"
    EXPLOIT_KIT = "exploit"
    APT_INDICATOR = "apt"
    SUSPICIOUS_IP = "suspicious_ip"
    RANSOMWARE = "ransomware"

@dataclass
class ThreatIndicator:
    """脅威インジケータ（IOC）のデータ構造"""
    indicator_type: str  # ip, domain, hash, url, email
    value: str
    threat_type: ThreatType
    confidence: float  # 0.0-1.0
    severity: str  # low, medium, high, critical
    source: str  # 情報源
    tags: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    ttl: int = 2592000  # 30日（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatIntelBlock:
    """脅威情報ブロックの構造"""
    index: int
    timestamp: datetime
    threat_data: List[ThreatIndicator]
    correlations: Dict[str, Any]
    effectiveness: float  # この情報の有効性スコア
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """ブロックのハッシュ値を計算"""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "threat_data": [self._serialize_indicator(t) for t in self.threat_data],
            "correlations": self.correlations,
            "effectiveness": self.effectiveness,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _serialize_indicator(self, indicator: ThreatIndicator) -> dict:
        """ThreatIndicatorをシリアライズ"""
        return {
            "type": indicator.indicator_type,
            "value": indicator.value,
            "threat_type": indicator.threat_type.value,
            "confidence": indicator.confidence,
            "severity": indicator.severity,
            "source": indicator.source,
            "tags": indicator.tags,
            "first_seen": indicator.first_seen.isoformat(),
            "last_seen": indicator.last_seen.isoformat(),
            "ttl": indicator.ttl,
            "metadata": indicator.metadata
        }

# === メインクラス ===

class ThreatIntelligenceChain:
    """脅威インテリジェンスのブロックチェーン実装"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[ThreatIntelBlock] = []
        self.pending_threats: List[ThreatIndicator] = []
        self.difficulty = difficulty  # マイニング難易度
        
        # インデックス構造
        self.ioc_index: Dict[str, List[int]] = defaultdict(list)  # IOC値 -> ブロック番号
        self.source_index: Dict[str, List[int]] = defaultdict(list)  # ソース -> ブロック番号
        self.threat_type_index: Dict[ThreatType, List[int]] = defaultdict(list)
        
        # 統計情報
        self.source_reliability: Dict[str, float] = defaultdict(lambda: 0.5)
        self.detection_history: deque = deque(maxlen=10000)
        
        # キャッシュ
        self.threat_cache: Dict[str, ThreatIndicator] = {}
        self.prediction_cache: Dict[str, List[ThreatIndicator]] = {}
        
        # 初期ブロック作成
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """ジェネシスブロックの作成"""
        genesis_indicator = ThreatIndicator(
            indicator_type="system",
            value="genesis",
            threat_type=ThreatType.APT_INDICATOR,
            confidence=1.0,
            severity="info",
            source="system",
            tags=["genesis", "initialization"]
        )
        
        genesis_block = ThreatIntelBlock(
            index=0,
            timestamp=datetime.now(),
            threat_data=[genesis_indicator],
            correlations={},
            effectiveness=1.0,
            previous_hash="0" * 64
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def add_threat_intelligence(self, threat_data: List[ThreatIndicator], 
                              correlations: Optional[Dict[str, Any]] = None) -> ThreatIntelBlock:
        """新しい脅威情報をチェーンに追加"""
        # 重複チェック
        unique_threats = self._filter_duplicates(threat_data)
        
        if not unique_threats:
            return None
        
        # 相関情報の生成
        if correlations is None:
            correlations = self._generate_correlations(unique_threats)
        
        # 有効性スコアの計算
        effectiveness = self._calculate_effectiveness(unique_threats)
        
        # 新しいブロックの作成
        new_block = ThreatIntelBlock(
            index=len(self.chain),
            timestamp=datetime.now(),
            threat_data=unique_threats,
            correlations=correlations,
            effectiveness=effectiveness,
            previous_hash=self.chain[-1].hash
        )
        
        # マイニング（Proof of Work）
        new_block = self._mine_block(new_block)
        
        # チェーンに追加
        self.chain.append(new_block)
        
        # インデックス更新
        self._update_indices(new_block)
        
        # キャッシュ更新
        self._update_cache(unique_threats)
        
        # 信頼性スコア更新
        self._update_source_reliability(unique_threats)
        
        return new_block
    
    def _filter_duplicates(self, threats: List[ThreatIndicator]) -> List[ThreatIndicator]:
        """重複を除外"""
        unique_threats = []
        for threat in threats:
            cache_key = f"{threat.indicator_type}:{threat.value}"
            if cache_key not in self.threat_cache:
                unique_threats.append(threat)
        return unique_threats
    
    def _generate_correlations(self, threats: List[ThreatIndicator]) -> Dict[str, Any]:
        """脅威間の相関を自動生成"""
        correlations = {
            "threat_count": len(threats),
            "severity_distribution": self._calculate_severity_distribution(threats),
            "source_distribution": self._calculate_source_distribution(threats),
            "temporal_pattern": self._analyze_temporal_pattern(threats),
            "threat_clusters": self._cluster_threats(threats)
        }
        return correlations
    
    def _calculate_effectiveness(self, threats: List[ThreatIndicator]) -> float:
        """脅威情報の有効性スコアを計算"""
        factors = []
        
        # 1. ソースの信頼性
        source_scores = [self.source_reliability.get(t.source, 0.5) for t in threats]
        factors.append(np.mean(source_scores))
        
        # 2. 情報の新鮮さ
        now = datetime.now()
        freshness_scores = []
        for threat in threats:
            age_hours = (now - threat.first_seen).total_seconds() / 3600
            freshness = max(0, 1 - (age_hours / 720))  # 30日で0
            freshness_scores.append(freshness)
        factors.append(np.mean(freshness_scores))
        
        # 3. 信頼度の平均
        confidence_scores = [t.confidence for t in threats]
        factors.append(np.mean(confidence_scores))
        
        # 4. 重要度（severity）の重み付け平均
        severity_weights = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        severity_scores = [severity_weights.get(t.severity, 0.5) for t in threats]
        factors.append(np.mean(severity_scores))
        
        # 総合スコア
        return np.mean(factors)
    
    def _mine_block(self, block: ThreatIntelBlock) -> ThreatIntelBlock:
        """Proof of Workによるマイニング"""
        target = "0" * self.difficulty
        
        while True:
            block.hash = block.calculate_hash()
            if block.hash[:self.difficulty] == target:
                break
            block.nonce += 1
        
        return block
    
    def _update_indices(self, block: ThreatIntelBlock):
        """インデックスの更新"""
        for threat in block.threat_data:
            # IOCインデックス
            self.ioc_index[threat.value].append(block.index)
            
            # ソースインデックス
            self.source_index[threat.source].append(block.index)
            
            # 脅威タイプインデックス
            self.threat_type_index[threat.threat_type].append(block.index)
    
    def _update_cache(self, threats: List[ThreatIndicator]):
        """キャッシュの更新"""
        for threat in threats:
            cache_key = f"{threat.indicator_type}:{threat.value}"
            self.threat_cache[cache_key] = threat
    
    def _update_source_reliability(self, threats: List[ThreatIndicator]):
        """ソースの信頼性スコアを更新"""
        # ここでは簡易的な実装
        # 実際には、検出実績との照合で更新
        for threat in threats:
            # 高severity + 高confidenceの情報を提供するソースの信頼性を上げる
            if threat.severity in ["high", "critical"] and threat.confidence > 0.8:
                current = self.source_reliability[threat.source]
                self.source_reliability[threat.source] = min(1.0, current * 1.05)
    
    # === 検索・照合機能 ===
    
    def check_ioc(self, indicator_value: str, 
                  indicator_type: Optional[str] = None) -> Optional[ThreatIndicator]:
        """IOCをチェック"""
        cache_key = f"{indicator_type}:{indicator_value}" if indicator_type else indicator_value
        
        # キャッシュチェック
        if cache_key in self.threat_cache:
            threat = self.threat_cache[cache_key]
            # TTLチェック
            if (datetime.now() - threat.first_seen).total_seconds() < threat.ttl:
                return threat
        
        # ブロックチェーンを検索
        if indicator_value in self.ioc_index:
            block_indices = self.ioc_index[indicator_value]
            for idx in reversed(block_indices):  # 最新から検索
                block = self.chain[idx]
                for threat in block.threat_data:
                    if threat.value == indicator_value:
                        if (datetime.now() - threat.first_seen).total_seconds() < threat.ttl:
                            return threat
        
        return None
    
    def search_by_threat_type(self, threat_type: ThreatType, 
                            limit: int = 100) -> List[ThreatIndicator]:
        """脅威タイプで検索"""
        results = []
        block_indices = self.threat_type_index.get(threat_type, [])
        
        for idx in reversed(block_indices[-limit:]):
            block = self.chain[idx]
            for threat in block.threat_data:
                if threat.threat_type == threat_type:
                    results.append(threat)
        
        return results
    
    # === 分析・予測機能 ===
    
    def analyze_threat_trends(self, days: int = 30) -> Dict[str, Any]:
        """脅威トレンドの分析"""
        cutoff_date = datetime.now() - timedelta(days=days)
        trends = {
            "threat_types": defaultdict(int),
            "sources": defaultdict(int),
            "severity_timeline": defaultdict(lambda: defaultdict(int)),
            "top_tags": defaultdict(int),
            "growth_rate": {}
        }
        
        # 指定期間のブロックを分析
        for block in self.chain:
            if block.timestamp < cutoff_date:
                continue
            
            day_key = block.timestamp.strftime("%Y-%m-%d")
            
            for threat in block.threat_data:
                trends["threat_types"][threat.threat_type.value] += 1
                trends["sources"][threat.source] += 1
                trends["severity_timeline"][day_key][threat.severity] += 1
                
                for tag in threat.tags:
                    trends["top_tags"][tag] += 1
        
        # 成長率の計算
        if len(trends["severity_timeline"]) > 1:
            dates = sorted(trends["severity_timeline"].keys())
            first_week = sum(sum(trends["severity_timeline"][d].values()) 
                           for d in dates[:7])
            last_week = sum(sum(trends["severity_timeline"][d].values()) 
                          for d in dates[-7:])
            
            if first_week > 0:
                trends["growth_rate"]["weekly"] = (last_week - first_week) / first_week
        
        return dict(trends)
    
    def predict_threats(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """脅威の予測"""
        # 履歴型多系列ベイズ推論のシミュレーション
        recent_blocks = self.chain[-100:]  # 直近100ブロック
        
        # 時系列パターンの抽出
        hourly_patterns = defaultdict(lambda: defaultdict(int))
        for block in recent_blocks:
            hour = block.timestamp.hour
            for threat in block.threat_data:
                hourly_patterns[hour][threat.threat_type.value] += 1
        
        # 予測
        predictions = []
        current_time = datetime.now()
        
        for hour_offset in range(hours_ahead):
            future_time = current_time + timedelta(hours=hour_offset)
            hour = future_time.hour
            
            if hour in hourly_patterns:
                for threat_type, count in hourly_patterns[hour].items():
                    # ベイズ推論のシミュレーション
                    prior = count / 100  # 事前確率
                    likelihood = self._calculate_likelihood(threat_type, hour)
                    posterior = prior * likelihood
                    
                    if posterior > 0.3:  # 閾値
                        predictions.append({
                            "time": future_time,
                            "threat_type": threat_type,
                            "probability": posterior,
                            "confidence": "high" if posterior > 0.7 else "medium"
                        })
        
        return predictions
    
    def _calculate_likelihood(self, threat_type: str, hour: int) -> float:
        """尤度の計算（簡易版）"""
        # 実際にはもっと複雑な計算
        base_likelihood = 0.5
        
        # 深夜早朝は攻撃が多い
        if hour < 6 or hour > 22:
            base_likelihood *= 1.5
        
        # APT攻撃は業務時間外
        if threat_type == "apt" and (hour < 9 or hour > 18):
            base_likelihood *= 1.3
        
        return min(1.0, base_likelihood)
    
    # === オフライン機能 ===
    
    def export_for_offline(self, filepath: str):
        """オフライン用にエクスポート"""
        export_data = {
            "chain": [self._serialize_block(block) for block in self.chain],
            "indices": {
                "ioc": dict(self.ioc_index),
                "source": dict(self.source_index),
                "threat_type": {k.value: v for k, v in self.threat_type_index.items()}
            },
            "statistics": {
                "source_reliability": dict(self.source_reliability),
                "total_threats": sum(len(block.threat_data) for block in self.chain),
                "export_time": datetime.now().isoformat()
            }
        }
        
        # 圧縮して保存
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
    
    def import_from_offline(self, filepath: str):
        """オフラインデータをインポート"""
        with gzip.open(filepath, 'rb') as f:
            import_data = pickle.load(f)
        
        # 差分のみマージする実装が必要
        # ここでは簡易版
        return len(import_data["chain"])
    
    def _serialize_block(self, block: ThreatIntelBlock) -> dict:
        """ブロックのシリアライズ"""
        return {
            "index": block.index,
            "timestamp": block.timestamp.isoformat(),
            "threat_data": [self._serialize_indicator(t) for t in block.threat_data],
            "correlations": block.correlations,
            "effectiveness": block.effectiveness,
            "previous_hash": block.previous_hash,
            "hash": block.hash
        }
    
    def _serialize_indicator(self, indicator: ThreatIndicator) -> dict:
        """インジケータのシリアライズ"""
        return {
            "type": indicator.indicator_type,
            "value": indicator.value,
            "threat_type": indicator.threat_type.value,
            "confidence": indicator.confidence,
            "severity": indicator.severity,
            "source": indicator.source,
            "tags": indicator.tags,
            "first_seen": indicator.first_seen.isoformat(),
            "last_seen": indicator.last_seen.isoformat(),
            "ttl": indicator.ttl,
            "metadata": indicator.metadata
        }
    
    # === 統計・分析メソッド ===
    
    def _calculate_severity_distribution(self, threats: List[ThreatIndicator]) -> Dict[str, float]:
        """重要度の分布を計算"""
        distribution = defaultdict(int)
        for threat in threats:
            distribution[threat.severity] += 1
        
        total = len(threats)
        return {k: v/total for k, v in distribution.items()} if total > 0 else {}
    
    def _calculate_source_distribution(self, threats: List[ThreatIndicator]) -> Dict[str, float]:
        """ソースの分布を計算"""
        distribution = defaultdict(int)
        for threat in threats:
            distribution[threat.source] += 1
        
        total = len(threats)
        return {k: v/total for k, v in distribution.items()} if total > 0 else {}
    
    def _analyze_temporal_pattern(self, threats: List[ThreatIndicator]) -> Dict[str, Any]:
        """時間的パターンの分析"""
        if not threats:
            return {}
        
        timestamps = [t.first_seen for t in threats]
        intervals = []
        
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if intervals:
            return {
                "avg_interval": np.mean(intervals),
                "std_interval": np.std(intervals),
                "burst_detected": np.std(intervals) < np.mean(intervals) * 0.1
            }
        return {}
    
    def _cluster_threats(self, threats: List[ThreatIndicator]) -> List[List[int]]:
        """脅威をクラスタリング（簡易版）"""
        # 実際にはもっと高度なクラスタリング
        clusters = defaultdict(list)
        
        for i, threat in enumerate(threats):
            cluster_key = f"{threat.threat_type.value}_{threat.severity}"
            clusters[cluster_key].append(i)
        
        return list(clusters.values())
    
    def verify_chain_integrity(self) -> bool:
        """チェーンの整合性を検証"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # ハッシュの検証
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # ブロック自体のハッシュ検証
            if current_block.hash != current_block.calculate_hash():
                return False
        
        return True
    
    def get_chain_statistics(self) -> Dict[str, Any]:
        """チェーンの統計情報を取得"""
        total_threats = sum(len(block.threat_data) for block in self.chain)
        
        return {
            "total_blocks": len(self.chain),
            "total_threats": total_threats,
            "avg_threats_per_block": total_threats / len(self.chain) if self.chain else 0,
            "unique_sources": len(self.source_index),
            "unique_iocs": len(self.ioc_index),
            "chain_valid": self.verify_chain_integrity(),
            "last_update": self.chain[-1].timestamp.isoformat() if self.chain else None
        }

"""
# === 使用例 ===

if __name__ == "__main__":
    # 脅威インテリジェンスチェーンの初期化
    threat_chain = ThreatIntelligenceChain(difficulty=4)
    
    # サンプル脅威情報の追加
    sample_threats = [
        ThreatIndicator(
            indicator_type="ip",
            value="198.51.100.50",
            threat_type=ThreatType.C2_SERVER,
            confidence=0.95,
            severity="critical",
            source="RecordedFuture",
            tags=["APT29", "NOBELIUM", "active"],
            metadata={"geo": "RU", "asn": "AS12345"}
        ),
        ThreatIndicator(
            indicator_type="hash",
            value="a1b2c3d4e5f6789012345678901234567890123456789012",
            threat_type=ThreatType.MALWARE_HASH,
            confidence=0.99,
            severity="high",
            source="VirusTotal",
            tags=["emotet", "dropper"],
            metadata={"detections": "68/72", "first_submission": "2025-01-15"}
        ),
        ThreatIndicator(
            indicator_type="domain",
            value="malicious-site.example.com",
            threat_type=ThreatType.PHISHING_URL,
            confidence=0.87,
            severity="medium",
            source="PhishTank",
            tags=["phishing", "credential_theft", "office365"],
            metadata={"target": "microsoft.com", "kit": "kr3pto"}
        )
    ]
    
    # ブロックチェーンに追加
    block = threat_chain.add_threat_intelligence(sample_threats)
    print(f"新しいブロックを追加: Block #{block.index}, Hash: {block.hash[:16]}...")
    
    # IOCチェック
    result = threat_chain.check_ioc("198.51.100.50", "ip")
    if result:
        print(f"\n脅威検出: {result.value}")
        print(f"  タイプ: {result.threat_type.value}")
        print(f"  信頼度: {result.confidence}")
        print(f"  重要度: {result.severity}")
        print(f"  タグ: {', '.join(result.tags)}")
    
    # トレンド分析
    trends = threat_chain.analyze_threat_trends(days=30)
    print(f"\n過去30日の脅威トレンド:")
    print(f"  脅威タイプ: {dict(trends['threat_types'])}")
    print(f"  情報源: {dict(trends['sources'])}")
    
    # 予測
    predictions = threat_chain.predict_threats(hours_ahead=24)
    print(f"\n今後24時間の脅威予測:")
    for pred in predictions[:5]:  # 上位5件
        print(f"  {pred['time'].strftime('%H:%M')} - {pred['threat_type']} "
              f"(確率: {pred['probability']:.2f}, 信頼度: {pred['confidence']})")
    
    # チェーン統計
    stats = threat_chain.get_chain_statistics()
    print(f"\nチェーン統計:")
    print(f"  総ブロック数: {stats['total_blocks']}")
    print(f"  総脅威数: {stats['total_threats']}")
    print(f"  チェーン整合性: {'✓' if stats['chain_valid'] else '✗'}")
    
    # オフライン用エクスポート
    threat_chain.export_for_offline("threat_intel_backup.gz")
    print("\nオフライン用バックアップを作成しました。")
    """
