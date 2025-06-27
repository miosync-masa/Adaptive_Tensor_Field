# =====================================
# インシデント対応支援システム
# =====================================

import json
import hashlib
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================
# データ構造定義
# =====================================

@dataclass
class IncidentMetadata:
    """インシデントメタデータ"""
    incident_id: str
    detection_time: datetime
    severity: str  # critical, high, medium, low
    attack_type: str
    affected_users: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    status: str = "detected"  # detected, analyzing, contained, resolved
    
@dataclass
class ForensicEvidence:
    """フォレンジック証拠"""
    evidence_id: str
    incident_id: str
    timestamp: datetime
    evidence_type: str  # log, memory, network, file
    data: Dict[str, Any]
    hash_value: str = ""
    
    def __post_init__(self):
        if not self.hash_value:
            self.hash_value = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """証拠データのハッシュ値計算"""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

# =====================================
# Phase 1: 即時対応システム
# =====================================

class ImmediateResponseSystem:
    """即時対応システム（〜30秒）"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        self.response_queue = deque()
        self.containment_rules = self._init_containment_rules()
        
    def _init_containment_rules(self) -> Dict:
        """封じ込めルールの初期化"""
        return {
            "investigating": {
                "actions": ["enhance_logging", "notify_admin"],
                "auto_execute": True
            },
            "suspicious": {
                "actions": ["alert_soc", "capture_evidence", "limit_access"],
                "auto_execute": True
            },
            "critical": {
                "actions": ["isolate_network", "freeze_account", "kill_process"],
                "auto_execute": False,  # 承認必要
                "approval_timeout": 30  # 30秒
            }
        }
    
    async def process_incident(self, event: Dict, detection_result: Dict) -> Dict:
        """インシデントの即時処理"""
        start_time = datetime.now()
        
        # 1. インシデントメタデータ作成
        incident = self._create_incident(event, detection_result)
        
        # 2. 自動封じ込め判定
        containment_actions = await self._determine_containment(incident, event)
        
        # 3. 緊急アラート発報
        alert_result = self._send_emergency_alert(incident, containment_actions)
        
        # 4. 証拠保全開始
        evidence_task = asyncio.create_task(
            self._preserve_initial_evidence(incident, event)
        )
        
        # 5. レスポンス記録
        response_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "incident": incident,
            "containment_actions": containment_actions,
            "alert_result": alert_result,
            "evidence_task": evidence_task,
            "response_time": response_time,
            "phase": "immediate_response"
        }
    
    def _create_incident(self, event: Dict, detection_result: Dict) -> IncidentMetadata:
        """インシデントメタデータの作成"""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{event.get('user_id', 'unknown')}"
        
        return IncidentMetadata(
            incident_id=incident_id,
            detection_time=datetime.now(),
            severity=detection_result.get("status", "unknown"),
            attack_type=self._classify_attack_type(event),
            affected_users=[event.get("user_id", "unknown")],
            affected_systems=self._identify_affected_systems(event)
        )
    
    def _classify_attack_type(self, event: Dict) -> str:
        """攻撃タイプの分類"""
        operation = event.get("operation", "").lower()
        file_path = event.get("file_path", "").lower()
        process = event.get("process_name", "").lower()
        
        # 攻撃パターンマッチング
        if "powershell" in process or "cmd" in process:
            if "confidential" in file_path:
                return "data_exfiltration"
            return "privilege_escalation"
        elif operation in ["filedelete", "processeterminate"]:
            return "destructive_attack"
        elif operation == "filecopy" and event.get("file_size_kb", 0) > 100000:
            return "data_theft"
        elif event.get("cross_dept_warning"):
            return "lateral_movement"
        
        return "unknown_attack"
    
    def _identify_affected_systems(self, event: Dict) -> List[str]:
        """影響を受けたシステムの特定"""
        systems = []
        
        # ファイルパスからシステム推定
        file_path = event.get("file_path", "")
        if "\\sales\\" in file_path:
            systems.append("営業管理システム")
        if "\\finance\\" in file_path:
            systems.append("財務システム")
        if "\\hr\\" in file_path:
            systems.append("人事システム")
        
        # IPアドレスからシステム推定
        dest_ip = event.get("destination_ip", "")
        if dest_ip.startswith("192.168.1."):
            systems.append("営業部ネットワーク")
        elif dest_ip.startswith("192.168.3."):
            systems.append("経理部ネットワーク")
        
        return systems or ["不明なシステム"]
    
    async def _determine_containment(self, incident: IncidentMetadata, event: Dict) -> Dict:
        """封じ込め戦略の決定"""
        severity = incident.severity
        rules = self.containment_rules.get(severity, {})
        
        actions_to_execute = []
        
        if rules.get("auto_execute", False):
            # 自動実行
            for action in rules.get("actions", []):
                result = await self._execute_containment_action(action, incident, event)
                actions_to_execute.append({
                    "action": action,
                    "status": "executed",
                    "result": result
                })
        else:
            # 承認待ち
            for action in rules.get("actions", []):
                actions_to_execute.append({
                    "action": action,
                    "status": "pending_approval",
                    "timeout": rules.get("approval_timeout", 30)
                })
        
        return {
            "severity": severity,
            "actions": actions_to_execute,
            "timestamp": datetime.now()
        }
    
    async def _execute_containment_action(self, action: str, incident: IncidentMetadata, 
                                        event: Dict) -> Dict:
        """封じ込めアクションの実行"""
        # 実際の環境では各アクションに対応するAPIを呼び出す
        action_map = {
            "enhance_logging": self._enhance_logging,
            "notify_admin": self._notify_admin,
            "alert_soc": self._alert_soc,
            "capture_evidence": self._capture_evidence,
            "limit_access": self._limit_access,
            "isolate_network": self._isolate_network,
            "freeze_account": self._freeze_account,
            "kill_process": self._kill_process
        }
        
        handler = action_map.get(action)
        if handler:
            return await handler(incident, event)
        
        return {"status": "error", "message": f"Unknown action: {action}"}
    
    async def _enhance_logging(self, incident: IncidentMetadata, event: Dict) -> Dict:
        """ログ記録の強化"""
        # シミュレーション
        await asyncio.sleep(0.1)
        return {
            "status": "success",
            "enhanced_targets": incident.affected_systems,
            "log_level": "DEBUG"
        }
    
    async def _isolate_network(self, incident: IncidentMetadata, event: Dict) -> Dict:
        """ネットワーク分離"""
        # シミュレーション
        await asyncio.sleep(0.2)
        return {
            "status": "success",
            "isolated_ip": event.get("source_ip"),
            "vlan_id": "quarantine_vlan_999"
        }
    
    def _send_emergency_alert(self, incident: IncidentMetadata, 
                            containment_actions: Dict) -> Dict:
        """緊急アラートの送信"""
        alert = {
            "alert_id": f"ALERT-{incident.incident_id}",
            "severity": incident.severity,
            "title": f"{incident.attack_type} detected - {incident.incident_id}",
            "affected_users": incident.affected_users,
            "affected_systems": incident.affected_systems,
            "containment_status": containment_actions,
            "timestamp": datetime.now(),
            "channels": ["email", "slack", "pagerduty"]
        }
        
        # 実際の環境では通知APIを呼び出す
        print(f"🚨 EMERGENCY ALERT: {alert['title']}")
        
        return alert
    
    async def _preserve_initial_evidence(self, incident: IncidentMetadata, 
                                       event: Dict) -> ForensicEvidence:
        """初期証拠の保全"""
        evidence = ForensicEvidence(
            evidence_id=f"EVD-{incident.incident_id}-001",
            incident_id=incident.incident_id,
            timestamp=datetime.now(),
            evidence_type="initial_detection",
            data={
                "raw_event": event,
                "detection_context": {
                    "chain_blocks": self._get_relevant_blocks(event),
                    "user_history": self._get_user_history(event)
                }
            }
        )
        
        # 証拠の永続化（実際はデータベースや専用ストレージ）
        await self._store_evidence(evidence)
        
        return evidence
    
    def _get_relevant_blocks(self, event: Dict) -> List[Dict]:
        """関連するチェーンブロックの取得"""
        # チェーンマネージャーから関連ブロックを取得
        user_id = event.get("user_id")
        dept = event.get("department")
        
        blocks = []
        if dept and dept in self.chain_manager.department_chains:
            chain = self.chain_manager.department_chains[dept]
            # 最新20ブロックを取得
            for block in chain.blocks[-20:]:
                if block.metadata.get("user_id") == user_id:
                    blocks.append({
                        "index": block.index,
                        "timestamp": block.metadata.get("timestamp"),
                        "security_mode": block.metadata.get("security_mode"),
                        "divergence": block.divergence
                    })
        
        return blocks
    
    def _get_user_history(self, event: Dict) -> Dict:
        """ユーザー履歴の取得"""
        user_id = event.get("user_id")
        if user_id in self.chain_manager.user_chains:
            user_chain = self.chain_manager.user_chains[user_id]
            return {
                "baseline_behavior": user_chain.baseline_behavior,
                "recent_alerts": user_chain.get_recent_alerts(n=10)
            }
        return {}
    
    async def _store_evidence(self, evidence: ForensicEvidence):
        """証拠の保存"""
        # シミュレーション
        await asyncio.sleep(0.1)
        print(f"📦 Evidence preserved: {evidence.evidence_id}")

# =====================================
# Phase 2: 初動分析システム
# =====================================

class InitialAnalysisSystem:
    """初動分析システム（〜5分）"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        self.timeline_generator = IncidentTimeline()
        
    async def analyze_incident(self, incident: IncidentMetadata, 
                             initial_evidence: ForensicEvidence) -> Dict:
        """インシデントの初動分析"""
        print(f"\n🔍 Phase 2: 初動分析開始 - {incident.incident_id}")
        
        # 1. タイムライン自動生成
        timeline = await self.timeline_generator.generate_timeline(
            incident, 
            self.chain_manager
        )
        
        # 2. 影響範囲の特定
        impact_scope = self._identify_impact_scope(incident, initial_evidence)
        
        # 3. 攻撃手法の分類
        attack_classification = self._classify_attack_technique(
            incident, 
            timeline, 
            initial_evidence
        )
        
        return {
            "incident_id": incident.incident_id,
            "timeline": timeline,
            "impact_scope": impact_scope,
            "attack_classification": attack_classification,
            "phase": "initial_analysis",
            "analysis_time": datetime.now()
        }
    
    def _identify_impact_scope(self, incident: IncidentMetadata, 
                             evidence: ForensicEvidence) -> Dict:
        """影響範囲の特定"""
        scope = {
            "direct_impact": {
                "users": set(incident.affected_users),
                "systems": set(incident.affected_systems),
                "data_assets": []
            },
            "potential_impact": {
                "users": set(),
                "systems": set(),
                "departments": set()
            },
            "risk_level": "unknown"
        }
        
        # 証拠データから影響範囲を拡張
        event_data = evidence.data.get("raw_event", {})
        
        # ファイルアクセスの影響
        if "file_path" in event_data:
            file_path = event_data["file_path"]
            scope["direct_impact"]["data_assets"].append(file_path)
            
            # 共有フォルダの場合、他ユーザーへの影響可能性
            if "\\shared\\" in file_path or "\\fileserver\\" in file_path:
                scope["potential_impact"]["users"].update(
                    self._get_shared_folder_users(file_path)
                )
        
        # 横展開の可能性評価
        if incident.attack_type == "lateral_movement":
            scope["potential_impact"]["departments"].update(
                self._get_connected_departments(incident.affected_users[0])
            )
        
        # リスクレベルの計算
        total_affected = (
            len(scope["direct_impact"]["users"]) + 
            len(scope["direct_impact"]["systems"]) +
            len(scope["potential_impact"]["users"]) +
            len(scope["potential_impact"]["departments"])
        )
        
        if total_affected > 10:
            scope["risk_level"] = "critical"
        elif total_affected > 5:
            scope["risk_level"] = "high"
        elif total_affected > 2:
            scope["risk_level"] = "medium"
        else:
            scope["risk_level"] = "low"
        
        return scope
    
    def _get_shared_folder_users(self, file_path: str) -> List[str]:
        """共有フォルダのアクセスユーザー取得（シミュレーション）"""
        # 実際はActive DirectoryやファイルサーバーのACLから取得
        dept_users = {
            "\\sales\\": ["yamada_t", "suzuki_m", "tanaka_s"],
            "\\finance\\": ["ito_h", "nakamura_r", "ogawa_s"],
            "\\hr\\": ["kato_m", "ishida_j", "hayashi_r"]
        }
        
        for path_pattern, users in dept_users.items():
            if path_pattern in file_path:
                return users
        
        return []
    
    def _get_connected_departments(self, user_id: str) -> List[str]:
        """接続可能な部署の取得"""
        # ユーザーの部署から横展開可能な部署を推定
        user_dept_map = {
            "sales": ["sales", "finance"],
            "engineering": ["engineering", "sales"],
            "finance": ["finance", "hr"],
            "hr": ["hr", "finance"]
        }
        
        user_dept = self.chain_manager.user_department_map.get(user_id, "unknown")
        return user_dept_map.get(user_dept, [])
    
    def _classify_attack_technique(self, incident: IncidentMetadata, 
                                 timeline: Dict, evidence: ForensicEvidence) -> Dict:
        """攻撃手法の分類（MITRE ATT&CKベース）"""
        classification = {
            "tactics": [],
            "techniques": [],
            "confidence": 0.0,
            "mitre_mapping": []
        }
        
        # タイムラインから攻撃手法を推定
        if timeline.get("reconnaissance"):
            classification["tactics"].append("Reconnaissance")
            classification["techniques"].append("Active Scanning")
            classification["mitre_mapping"].append("T1595")
        
        if timeline.get("privilege_escalation"):
            classification["tactics"].append("Privilege Escalation")
            classification["techniques"].append("Valid Accounts")
            classification["mitre_mapping"].append("T1078")
        
        if timeline.get("lateral_movement"):
            classification["tactics"].append("Lateral Movement")
            classification["techniques"].append("Remote Services")
            classification["mitre_mapping"].append("T1021")
        
        if timeline.get("data_collection"):
            classification["tactics"].append("Collection")
            classification["techniques"].append("Data from Local System")
            classification["mitre_mapping"].append("T1005")
        
        # 信頼度の計算
        evidence_count = len([k for k, v in timeline.items() if v])
        classification["confidence"] = min(evidence_count * 0.25, 1.0)
        
        return classification

# =====================================
# タイムライン生成クラス
# =====================================

class IncidentTimeline:
    """インシデントタイムライン生成"""
    
    async def generate_timeline(self, incident: IncidentMetadata, 
                              chain_manager) -> Dict:
        """攻撃タイムラインの生成"""
        timeline = {
            "incident_id": incident.incident_id,
            "phases": {},
            "events": [],
            "visualization": None
        }
        
        # 攻撃フェーズの特定
        phases = await self._identify_attack_phases(incident, chain_manager)
        timeline["phases"] = phases
        
        # イベントの時系列整理
        events = self._extract_timeline_events(incident, chain_manager)
        timeline["events"] = sorted(events, key=lambda x: x["timestamp"])
        
        # 可視化データの生成
        timeline["visualization"] = self._create_timeline_visualization(
            timeline["phases"], 
            timeline["events"]
        )
        
        return timeline
    
    async def _identify_attack_phases(self, incident: IncidentMetadata, 
                                    chain_manager) -> Dict:
        """攻撃フェーズの特定"""
        phases = {
            "reconnaissance": None,
            "initial_access": None,
            "privilege_escalation": None,
            "lateral_movement": None,
            "data_collection": None,
            "data_exfiltration": None,
            "cleanup": None
        }
        
        # チェーンから関連イベントを抽出
        user_id = incident.affected_users[0] if incident.affected_users else None
        if not user_id:
            return phases
        
        # 部署チェーンから時系列イベントを取得
        dept = chain_manager.user_department_map.get(user_id)
        if dept and dept in chain_manager.department_chains:
            chain = chain_manager.department_chains[dept]
            
            for block in chain.blocks:
                if block.metadata.get("user_id") != user_id:
                    continue
                
                # フェーズ判定
                phase = self._determine_phase(block)
                if phase and not phases[phase]:
                    phases[phase] = {
                        "timestamp": block.metadata.get("timestamp"),
                        "block_index": block.index,
                        "evidence": block.metadata
                    }
        
        return phases
    
    def _determine_phase(self, block) -> Optional[str]:
        """ブロックから攻撃フェーズを判定"""
        metadata = block.metadata
        operation = metadata.get("operation", "").lower()
        mode = metadata.get("security_mode", "")
        
        # フェーズマッピング
        if mode == "normal" and operation in ["fileread", "processlist"]:
            return "reconnaissance"
        elif mode in ["investigating", "suspicious"] and operation == "login":
            return "initial_access"
        elif "powershell" in metadata.get("process_name", "").lower():
            return "privilege_escalation"
        elif metadata.get("cross_dept_warning"):
            return "lateral_movement"
        elif operation in ["filecopy", "fileread"] and metadata.get("file_size_kb", 0) > 10000:
            return "data_collection"
        elif metadata.get("destination_ip", "").startswith("203."):
            return "data_exfiltration"
        elif operation == "filedelete":
            return "cleanup"
        
        return None
    
    def _extract_timeline_events(self, incident: IncidentMetadata, 
                               chain_manager) -> List[Dict]:
        """タイムラインイベントの抽出"""
        events = []
        user_id = incident.affected_users[0] if incident.affected_users else None
        
        if not user_id:
            return events
        
        # ユーザーチェーンからイベント抽出
        if user_id in chain_manager.user_chains:
            user_chain = chain_manager.user_chains[user_id]
            
            for block in user_chain.blocks[-50:]:  # 最新50ブロック
                event = {
                    "timestamp": block.metadata.get("timestamp", ""),
                    "operation": block.metadata.get("operation", ""),
                    "severity": block.metadata.get("security_mode", ""),
                    "divergence": block.divergence,
                    "details": {
                        "file_path": block.metadata.get("file_path"),
                        "process": block.metadata.get("process_name"),
                        "destination": block.metadata.get("destination_ip")
                    }
                }
                events.append(event)
        
        return events
    
    def _create_timeline_visualization(self, phases: Dict, events: List[Dict]) -> Dict:
        """タイムライン可視化データの作成"""
        viz_data = {
            "type": "timeline",
            "phases": [],
            "events": [],
            "severity_map": {
                "normal": "🟢",
                "investigating": "🟡",
                "suspicious": "🟠",
                "critical": "🔴"
            }
        }
        
        # フェーズデータ
        for phase_name, phase_data in phases.items():
            if phase_data:
                viz_data["phases"].append({
                    "name": phase_name,
                    "timestamp": phase_data["timestamp"],
                    "icon": self._get_phase_icon(phase_name)
                })
        
        # イベントデータ（サンプリング）
        sampled_events = events[::max(1, len(events)//20)]  # 最大20イベント
        for event in sampled_events:
            viz_data["events"].append({
                "time": event["timestamp"].split()[1] if event["timestamp"] else "00:00:00",
                "operation": event["operation"],
                "severity_icon": viz_data["severity_map"].get(event["severity"], "⚪"),
                "divergence": event["divergence"]
            })
        
        return viz_data
    
    def _get_phase_icon(self, phase: str) -> str:
        """フェーズアイコンの取得"""
        icons = {
            "reconnaissance": "🔍",
            "initial_access": "🚪",
            "privilege_escalation": "⬆️",
            "lateral_movement": "➡️",
            "data_collection": "📊",
            "data_exfiltration": "📤",
            "cleanup": "🧹"
        }
        return icons.get(phase, "❓")

# =====================================
# Phase 3: 詳細調査システム
# =====================================

class DetailedInvestigationSystem:
    """詳細調査システム（〜30分）"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.risk_evaluator = LateralMovementRiskEvaluator()
        
    async def investigate_incident(self, incident: IncidentMetadata, 
                                 initial_analysis: Dict) -> Dict:
        """インシデントの詳細調査"""
        print(f"\n🔬 Phase 3: 詳細調査開始 - {incident.incident_id}")
        
        # 1. 根本原因分析
        root_cause = await self.root_cause_analyzer.analyze(
            incident, 
            initial_analysis,
            self.chain_manager
        )
        
        # 2. 横展開リスク評価
        lateral_risk = self.risk_evaluator.evaluate(
            incident,
            initial_analysis["impact_scope"],
            self.chain_manager
        )
        
        # 3. 対策優先度決定
        mitigation_priority = self._determine_mitigation_priority(
            root_cause,
            lateral_risk,
            initial_analysis
        )
        
        return {
            "incident_id": incident.incident_id,
            "root_cause_analysis": root_cause,
            "lateral_movement_risk": lateral_risk,
            "mitigation_priority": mitigation_priority,
            "phase": "detailed_investigation",
            "investigation_time": datetime.now()
        }
    
    def _determine_mitigation_priority(self, root_cause: Dict, 
                                     lateral_risk: Dict, 
                                     initial_analysis: Dict) -> List[Dict]:
        """対策優先度の決定"""
        mitigations = []
        
        # 根本原因に基づく対策
        if root_cause.get("vulnerability_type") == "weak_authentication":
            mitigations.append({
                "action": "enforce_mfa",
                "priority": "critical",
                "estimated_time": "30 minutes",
                "impact": "high"
            })
        
        if root_cause.get("vulnerability_type") == "unpatched_system":
            mitigations.append({
                "action": "emergency_patching",
                "priority": "high",
                "estimated_time": "2 hours",
                "impact": "medium"
            })
        
        # 横展開リスクに基づく対策
        if lateral_risk.get("risk_score", 0) > 0.7:
            mitigations.append({
                "action": "network_segmentation",
                "priority": "critical",
                "estimated_time": "1 hour",
                "impact": "high"
            })
        
        # 影響範囲に基づく対策
        if initial_analysis["impact_scope"]["risk_level"] == "critical":
            mitigations.append({
                "action": "full_system_isolation",
                "priority": "critical",
                "estimated_time": "15 minutes",
                "impact": "very_high"
            })
        
        # 優先度でソート
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        mitigations.sort(key=lambda x: priority_order.get(x["priority"], 999))
        
        return mitigations

# =====================================
# 根本原因分析クラス
# =====================================

class RootCauseAnalyzer:
    """根本原因分析システム"""
    
    async def analyze(self, incident: IncidentMetadata, 
                     initial_analysis: Dict, 
                     chain_manager) -> Dict:
        """根本原因の分析"""
        root_cause = {
            "incident_id": incident.incident_id,
            "initial_vector": None,
            "vulnerability_type": None,
            "contributing_factors": [],
            "confidence": 0.0,
            "evidence_chain": []
        }
        
        # タイムラインから初期侵入ベクトルを特定
        timeline = initial_analysis.get("timeline", {})
        phases = timeline.get("phases", {})
        
        # 初期アクセスフェーズの分析
        if phases.get("initial_access"):
            initial_access = phases["initial_access"]
            root_cause["initial_vector"] = self._analyze_initial_vector(
                initial_access["evidence"]
            )
        
        # 脆弱性の特定
        root_cause["vulnerability_type"] = self._identify_vulnerability(
            incident,
            initial_analysis,
            chain_manager
        )
        
        # 寄与要因の分析
        root_cause["contributing_factors"] = self._analyze_contributing_factors(
            incident,
            chain_manager
        )
        
        # 証拠チェーンの構築
        root_cause["evidence_chain"] = self._build_evidence_chain(
            timeline,
            chain_manager
        )
        
        # 信頼度の計算
        root_cause["confidence"] = self._calculate_confidence(root_cause)
        
        return root_cause
    
    def _analyze_initial_vector(self, evidence: Dict) -> str:
        """初期侵入ベクトルの分析"""
        operation = evidence.get("operation", "").lower()
        source_ip = evidence.get("source_ip", "")
        
        if operation == "login" and not source_ip.startswith("192.168."):
            return "external_remote_access"
        elif operation == "loginfailed":
            return "brute_force_attempt"
        elif "phishing" in evidence.get("file_path", "").lower():
            return "phishing_email"
        else:
            return "unknown_vector"
    
    def _identify_vulnerability(self, incident: IncidentMetadata, 
                               initial_analysis: Dict, 
                               chain_manager) -> str:
        """脆弱性タイプの特定"""
        # 攻撃分類から脆弱性を推定
        attack_class = initial_analysis.get("attack_classification", {})
        techniques = attack_class.get("techniques", [])
        
        if "Valid Accounts" in techniques:
            # アカウント関連の脆弱性チェック
            user_id = incident.affected_users[0] if incident.affected_users else None
            if user_id and user_id in chain_manager.user_chains:
                user_chain = chain_manager.user_chains[user_id]
                # 過去のログイン失敗をチェック
                failed_logins = sum(
                    1 for block in user_chain.blocks 
                    if block.metadata.get("operation") == "LoginFailed"
                )
                if failed_logins > 5:
                    return "weak_authentication"
        
        if "PowerShell" in str(initial_analysis):
            return "unrestricted_script_execution"
        
        if incident.attack_type == "lateral_movement":
            return "excessive_privileges"
        
        return "unknown_vulnerability"
    
    def _analyze_contributing_factors(self, incident: IncidentMetadata, 
                                    chain_manager) -> List[str]:
        """寄与要因の分析"""
        factors = []
        
        # ユーザー行動パターンの分析
        user_id = incident.affected_users[0] if incident.affected_users else None
        if user_id and user_id in chain_manager.user_chains:
            user_chain = chain_manager.user_chains[user_id]
            
            # 異常な時間帯のアクセス
            night_access = sum(
                1 for block in user_chain.blocks[-100:]
                if block.metadata.get("timestamp", "").split()[1].startswith(("00:", "01:", "02:", "03:", "04:", "05:"))
            )
            if night_access > 5:
                factors.append("unusual_access_hours")
            
            # 新しいシステムへのアクセス
            if user_chain.baseline_behavior:
                baseline_dirs = set(user_chain.baseline_behavior.get("common_directories", []))
                recent_dirs = set()
                for block in user_chain.blocks[-50:]:
                    file_path = block.metadata.get("file_path", "")
                    if file_path:
                        directory = "\\".join(file_path.split("\\")[:-1])
                        recent_dirs.add(directory)
                
                new_dirs = recent_dirs - baseline_dirs
                if new_dirs:
                    factors.append("access_to_new_systems")
        
        # 組織的要因
        if incident.attack_type == "data_exfiltration":
            factors.append("inadequate_dlp_controls")
        
        if "cross_dept_warning" in str(chain_manager.cross_department_access):
            factors.append("weak_access_controls")
        
        return factors
    
    def _build_evidence_chain(self, timeline: Dict, chain_manager) -> List[Dict]:
        """証拠チェーンの構築"""
        evidence_chain = []
        
        # タイムラインのフェーズごとに証拠を収集
        phases = timeline.get("phases", {})
        for phase_name, phase_data in phases.items():
            if phase_data:
                evidence_chain.append({
                    "phase": phase_name,
                    "timestamp": phase_data["timestamp"],
                    "block_index": phase_data["block_index"],
                    "key_evidence": self._extract_key_evidence(phase_data["evidence"])
                })
        
        return sorted(evidence_chain, key=lambda x: x["timestamp"] if x["timestamp"] else "")
    
    def _extract_key_evidence(self, evidence: Dict) -> Dict:
        """重要な証拠の抽出"""
        return {
            "operation": evidence.get("operation"),
            "file_path": evidence.get("file_path"),
            "process": evidence.get("process_name"),
            "security_mode": evidence.get("security_mode"),
            "divergence": evidence.get("divergence_score")
        }
    
    def _calculate_confidence(self, root_cause: Dict) -> float:
        """分析の信頼度計算"""
        confidence = 0.0
        
        # 各要素の存在で信頼度を加算
        if root_cause["initial_vector"] and root_cause["initial_vector"] != "unknown_vector":
            confidence += 0.3
        
        if root_cause["vulnerability_type"] and root_cause["vulnerability_type"] != "unknown_vulnerability":
            confidence += 0.3
        
        if len(root_cause["contributing_factors"]) > 0:
            confidence += 0.2
        
        if len(root_cause["evidence_chain"]) > 3:
            confidence += 0.2
        
        return min(confidence, 1.0)

# =====================================
# 横展開リスク評価クラス
# =====================================

class LateralMovementRiskEvaluator:
    """横展開リスク評価システム"""
    
    def evaluate(self, incident: IncidentMetadata, impact_scope: Dict, 
                chain_manager) -> Dict:
        """横展開リスクの評価"""
        risk_assessment = {
            "incident_id": incident.incident_id,
            "risk_score": 0.0,
            "affected_departments": [],
            "critical_systems_at_risk": [],
            "propagation_paths": [],
            "containment_recommendations": []
        }
        
        # 現在の影響部署
        current_dept = self._get_user_department(
            incident.affected_users[0] if incident.affected_users else None,
            chain_manager
        )
        
        # リスクのある部署の特定
        risk_assessment["affected_departments"] = self._identify_at_risk_departments(
            current_dept,
            impact_scope,
            chain_manager
        )
        
        # 重要システムのリスク評価
        risk_assessment["critical_systems_at_risk"] = self._identify_critical_systems(
            risk_assessment["affected_departments"]
        )
        
        # 伝播経路の分析
        risk_assessment["propagation_paths"] = self._analyze_propagation_paths(
            incident,
            chain_manager
        )
        
        # リスクスコアの計算
        risk_assessment["risk_score"] = self._calculate_risk_score(risk_assessment)
        
        # 封じ込め推奨事項
        risk_assessment["containment_recommendations"] = self._generate_containment_recommendations(
            risk_assessment
        )
        
        return risk_assessment
    
    def _get_user_department(self, user_id: str, chain_manager) -> str:
        """ユーザーの部署取得"""
        return chain_manager.user_department_map.get(user_id, "unknown")
    
    def _identify_at_risk_departments(self, current_dept: str, 
                                    impact_scope: Dict, 
                                    chain_manager) -> List[str]:
        """リスクのある部署の特定"""
        at_risk_depts = set()
        
        # 直接影響を受けた部署
        at_risk_depts.add(current_dept)
        
        # 潜在的影響の部署
        at_risk_depts.update(impact_scope.get("potential_impact", {}).get("departments", []))
        
        # 部署間アクセス履歴から追加
        for access_key in chain_manager.cross_department_access:
            parts = access_key.split("_")
            if len(parts) >= 3:
                from_dept = parts[-2]
                to_dept = parts[-1]
                if from_dept == current_dept:
                    at_risk_depts.add(to_dept)
        
        return list(at_risk_depts)
    
    def _identify_critical_systems(self, departments: List[str]) -> List[Dict]:
        """重要システムの特定"""
        critical_systems = []
        
        # 部署と重要システムのマッピング
        dept_critical_systems = {
            "finance": [
                {"name": "決済システム", "criticality": "extreme"},
                {"name": "財務管理システム", "criticality": "high"}
            ],
            "hr": [
                {"name": "給与システム", "criticality": "extreme"},
                {"name": "人事管理システム", "criticality": "high"}
            ],
            "sales": [
                {"name": "顧客管理システム", "criticality": "high"},
                {"name": "受注管理システム", "criticality": "medium"}
            ],
            "engineering": [
                {"name": "ソースコード管理", "criticality": "extreme"},
                {"name": "CI/CDパイプライン", "criticality": "high"}
            ]
        }
        
        for dept in departments:
            if dept in dept_critical_systems:
                critical_systems.extend(dept_critical_systems[dept])
        
        # 重複除去と重要度でソート
        unique_systems = {s["name"]: s for s in critical_systems}.values()
        criticality_order = {"extreme": 0, "high": 1, "medium": 2, "low": 3}
        
        return sorted(unique_systems, key=lambda x: criticality_order.get(x["criticality"], 999))
    
    def _analyze_propagation_paths(self, incident: IncidentMetadata, 
                                  chain_manager) -> List[Dict]:
        """伝播経路の分析"""
        paths = []
        
        # ネットワーク構造から経路を推定
        user_id = incident.affected_users[0] if incident.affected_users else None
        if not user_id:
            return paths
        
        # 共有リソース経由の伝播
        paths.append({
            "type": "shared_resource",
            "path": "User → Shared Folder → Other Users",
            "probability": 0.7,
            "speed": "fast"
        })
        
        # 認証情報経由の伝播
        if incident.attack_type in ["privilege_escalation", "lateral_movement"]:
            paths.append({
                "type": "credential_reuse",
                "path": "Compromised Account → Domain Controller → All Systems",
                "probability": 0.9,
                "speed": "very_fast"
            })
        
        # アプリケーション経由の伝播
        if "process" in str(incident.__dict__):
            paths.append({
                "type": "application_vulnerability",
                "path": "Infected Process → Connected Services → Backend Systems",
                "probability": 0.5,
                "speed": "medium"
            })
        
        return paths
    
    def _calculate_risk_score(self, risk_assessment: Dict) -> float:
        """リスクスコアの計算"""
        score = 0.0
        
        # 影響部署数によるスコア
        dept_count = len(risk_assessment["affected_departments"])
        score += min(dept_count * 0.2, 0.4)
        
        # 重要システムによるスコア
        critical_count = sum(
            1 for s in risk_assessment["critical_systems_at_risk"] 
            if s["criticality"] in ["extreme", "high"]
        )
        score += min(critical_count * 0.3, 0.6)
        
        # 伝播経路の危険度
        high_prob_paths = sum(
            1 for p in risk_assessment["propagation_paths"]
            if p["probability"] > 0.7
        )
        score += min(high_prob_paths * 0.2, 0.4)
        
        return min(score, 1.0)
    
    def _generate_containment_recommendations(self, risk_assessment: Dict) -> List[Dict]:
        """封じ込め推奨事項の生成"""
        recommendations = []
        
        # リスクスコアに基づく推奨
        if risk_assessment["risk_score"] > 0.8:
            recommendations.append({
                "action": "immediate_network_isolation",
                "target": "all_affected_departments",
                "urgency": "critical",
                "description": "影響部署の完全ネットワーク分離"
            })
        
        # 重要システム保護
        for system in risk_assessment["critical_systems_at_risk"]:
            if system["criticality"] == "extreme":
                recommendations.append({
                    "action": "system_protection",
                    "target": system["name"],
                    "urgency": "high",
                    "description": f"{system['name']}への全アクセスを一時遮断"
                })
        
        # 伝播経路の遮断
        for path in risk_assessment["propagation_paths"]:
            if path["probability"] > 0.7:
                recommendations.append({
                    "action": "block_propagation_path",
                    "target": path["type"],
                    "urgency": "high",
                    "description": f"{path['path']}の遮断"
                })
        
        return recommendations

# =====================================
# Phase 4: 復旧支援システム
# =====================================

class RecoveryAssistanceSystem:
    """復旧支援システム（〜数時間）"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        
    async def assist_recovery(self, incident: IncidentMetadata, 
                            investigation_results: Dict) -> Dict:
        """復旧支援の実行"""
        print(f"\n🔧 Phase 4: 復旧支援開始 - {incident.incident_id}")
        
        # 1. 封じ込め範囲の最適化
        optimized_containment = self._optimize_containment(
            incident,
            investigation_results
        )
        
        # 2. 復旧手順の生成
        recovery_procedures = self._generate_recovery_procedures(
            incident,
            investigation_results,
            optimized_containment
        )
        
        # 3. 再発防止策の実装提案
        prevention_measures = self._design_prevention_measures(
            investigation_results["root_cause_analysis"],
            investigation_results["lateral_movement_risk"]
        )
        
        return {
            "incident_id": incident.incident_id,
            "optimized_containment": optimized_containment,
            "recovery_procedures": recovery_procedures,
            "prevention_measures": prevention_measures,
            "phase": "recovery_assistance",
            "recovery_time": datetime.now()
        }
    
    def _optimize_containment(self, incident: IncidentMetadata, 
                            investigation_results: Dict) -> Dict:
        """封じ込め範囲の最適化"""
        optimization = {
            "current_containment": self._get_current_containment(incident),
            "recommended_adjustments": [],
            "business_impact_analysis": {}
        }
        
        # ビジネス影響の分析
        impact = self._analyze_business_impact(
            incident,
            optimization["current_containment"]
        )
        optimization["business_impact_analysis"] = impact
        
        # 封じ込め範囲の調整提案
        risk_score = investigation_results["lateral_movement_risk"]["risk_score"]
        
        if risk_score < 0.3:
            # 低リスク：封じ込めを緩和
            optimization["recommended_adjustments"].append({
                "action": "reduce_isolation",
                "target": "non_critical_systems",
                "rationale": "横展開リスクが低いため、業務影響を最小化"
            })
        elif risk_score > 0.7:
            # 高リスク：封じ込めを強化
            optimization["recommended_adjustments"].append({
                "action": "expand_isolation",
                "target": "connected_systems",
                "rationale": "横展開リスクが高いため、予防的隔離を実施"
            })
        
        return optimization
    
    def _get_current_containment(self, incident: IncidentMetadata) -> Dict:
        """現在の封じ込め状態を取得"""
        # 実際の環境では、ファイアウォールやEDRから情報取得
        return {
            "isolated_users": incident.affected_users,
            "isolated_systems": incident.affected_systems,
            "blocked_processes": ["powershell.exe", "cmd.exe"],
            "network_restrictions": ["external_access_blocked"]
        }
    
    def _analyze_business_impact(self, incident: IncidentMetadata, 
                               containment: Dict) -> Dict:
        """ビジネスへの影響分析"""
        impact = {
            "affected_business_processes": [],
            "estimated_downtime": 0,
            "revenue_impact": "unknown",
            "user_productivity_loss": 0
        }
        
        # 影響を受けるビジネスプロセスの特定
        for system in containment["isolated_systems"]:
            if "営業" in system:
                impact["affected_business_processes"].append("受注処理")
                impact["estimated_downtime"] += 2  # 時間
            elif "財務" in system:
                impact["affected_business_processes"].append("支払処理")
                impact["estimated_downtime"] += 4
            elif "人事" in system:
                impact["affected_business_processes"].append("勤怠管理")
                impact["estimated_downtime"] += 1
        
        # ユーザー生産性の損失
        impact["user_productivity_loss"] = len(containment["isolated_users"]) * 8  # 人時
        
        return impact
    
    def _generate_recovery_procedures(self, incident: IncidentMetadata,
                                    investigation_results: Dict,
                                    optimized_containment: Dict) -> List[Dict]:
        """復旧手順の生成"""
        procedures = []
        
        # 1. 初期確認手順
        procedures.append({
            "step": 1,
            "title": "インシデント封じ込めの確認",
            "actions": [
                "影響を受けたシステムのネットワーク分離状態を確認",
                "悪意のあるプロセスの停止を確認",
                "ログ収集の継続を確認"
            ],
            "estimated_time": "15分",
            "responsible_team": "SOC"
        })
        
        # 2. システムクリーンアップ
        procedures.append({
            "step": 2,
            "title": "影響システムのクリーンアップ",
            "actions": [
                "マルウェアスキャンの実行",
                "不正なファイル・レジストリの削除",
                "正規の設定へのロールバック"
            ],
            "estimated_time": "1時間",
            "responsible_team": "IT運用"
        })
        
        # 3. アカウントリセット
        if incident.attack_type in ["privilege_escalation", "lateral_movement"]:
            procedures.append({
                "step": 3,
                "title": "認証情報のリセット",
                "actions": [
                    "影響を受けたアカウントのパスワードリセット",
                    "関連するサービスアカウントの確認と更新",
                    "MFA（多要素認証）の再設定"
                ],
                "estimated_time": "30分",
                "responsible_team": "ID管理"
            })
        
        # 4. 段階的復旧
        procedures.append({
            "step": 4,
            "title": "システムの段階的復旧",
            "actions": [
                "重要度の低いシステムから順次接続を回復",
                "各段階でのセキュリティ監視の強化",
                "異常検知時の即時ロールバック準備"
            ],
            "estimated_time": "2時間",
            "responsible_team": "IT運用"
        })
        
        # 5. 検証とモニタリング
        procedures.append({
            "step": 5,
            "title": "復旧後の検証",
            "actions": [
                "全システムの正常動作確認",
                "セキュリティログの詳細レビュー",
                "24時間の強化監視体制"
            ],
            "estimated_time": "継続的",
            "responsible_team": "SOC"
        })
        
        return procedures
    
    def _design_prevention_measures(self, root_cause: Dict, 
                                  lateral_risk: Dict) -> List[Dict]:
        """再発防止策の設計"""
        measures = []
        
        # 根本原因に対する対策
        vulnerability = root_cause.get("vulnerability_type")
        
        if vulnerability == "weak_authentication":
            measures.append({
                "measure": "認証強化",
                "actions": [
                    "全ユーザーへのMFA義務化",
                    "パスワードポリシーの強化",
                    "定期的なパスワード変更の実施"
                ],
                "priority": "high",
                "implementation_time": "1週間"
            })
        
        elif vulnerability == "unrestricted_script_execution":
            measures.append({
                "measure": "スクリプト実行制御",
                "actions": [
                    "PowerShell実行ポリシーの制限",
                    "AppLockerによるアプリケーション制御",
                    "スクリプト実行の監査ログ強化"
                ],
                "priority": "high",
                "implementation_time": "3日"
            })
        
        # 横展開リスクに対する対策
        if lateral_risk.get("risk_score", 0) > 0.5:
            measures.append({
                "measure": "ネットワークセグメンテーション",
                "actions": [
                    "部署間ネットワークの論理分離",
                    "最小権限の原則の徹底",
                    "特権アカウントの利用制限"
                ],
                "priority": "critical",
                "implementation_time": "2週間"
            })
        
        # 検知能力の向上
        measures.append({
            "measure": "検知能力の強化",
            "actions": [
                "EDRソリューションの導入・更新",
                "SIEMルールの追加・調整",
                "脅威ハンティングの定期実施"
            ],
            "priority": "medium",
            "implementation_time": "1ヶ月"
        })
        
        # 教育・訓練
        measures.append({
            "measure": "セキュリティ意識向上",
            "actions": [
                "インシデント事例の共有",
                "標的型攻撃対応訓練の実施",
                "セキュリティ教育の強化"
            ],
            "priority": "medium",
            "implementation_time": "継続的"
        })
        
        return measures

# =====================================
# 可視化システム
# =====================================

class IncidentVisualizationSystem:
    """インシデント可視化システム"""
    
    def __init__(self):
        self.colors = {
            "normal": "#4CAF50",
            "investigating": "#FFC107",
            "suspicious": "#FF9800",
            "critical": "#F44336"
        }
    
    def create_timeline_visualization(self, timeline_data: Dict) -> go.Figure:
        """インタラクティブタイムラインの作成"""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("攻撃タイムライン", "リスクレベル推移")
        )
        
        # タイムラインプロット
        phases = timeline_data.get("phases", {})
        events = timeline_data.get("events", [])
        
        # フェーズのプロット
        phase_times = []
        phase_names = []
        phase_colors = []
        
        for phase_name, phase_data in phases.items():
            if phase_data:
                phase_times.append(phase_data["timestamp"])
                phase_names.append(phase_name)
                phase_colors.append(self._get_phase_color(phase_name))
        
        fig.add_trace(
            go.Scatter(
                x=phase_times,
                y=[1] * len(phase_times),
                mode='markers+text',
                name='攻撃フェーズ',
                text=phase_names,
                textposition="top center",
                marker=dict(size=20, color=phase_colors),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # イベントのプロット
        event_times = [e["timestamp"] for e in events if e.get("timestamp")]
        event_severities = [e["severity"] for e in events if e.get("severity")]
        event_colors = [self.colors.get(s, "#999999") for s in event_severities]
        
        fig.add_trace(
            go.Scatter(
                x=event_times,
                y=[0.5] * len(event_times),
                mode='markers',
                name='セキュリティイベント',
                marker=dict(size=10, color=event_colors),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # リスクレベル推移
        divergences = [e.get("divergence", 0) for e in events]
        
        fig.add_trace(
            go.Scatter(
                x=event_times,
                y=divergences,
                mode='lines+markers',
                name='Divergenceスコア',
                line=dict(color='red', width=2),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # レイアウト設定
        fig.update_layout(
            title="インシデント タイムライン分析",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="時刻", row=2, col=1)
        fig.update_yaxes(title_text="レベル", range=[0, 1.5], row=1, col=1)
        fig.update_yaxes(title_text="スコア", row=2, col=1)
        
        return fig
    
    def create_impact_heatmap(self, impact_data: Dict, time_range: int = 24) -> plt.Figure:
        """部署別リスクヒートマップの作成"""
        # サンプルデータ生成（実際はchain_managerから取得）
        departments = ["営業部", "経理部", "開発部", "人事部"]
        hours = list(range(time_range))
        
        # リスクマトリックスの生成
        risk_matrix = np.random.rand(len(departments), len(hours))
        
        # 影響データの反映
        affected_depts = impact_data.get("affected_departments", [])
        for i, dept in enumerate(departments):
            if dept in affected_depts:
                risk_matrix[i, :] *= 2  # 影響を受けた部署のリスクを上げる
        
        # ヒートマップの作成
        fig, ax = plt.subplots(figsize=(15, 8))
        
        im = ax.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # 軸の設定
        ax.set_xticks(np.arange(len(hours)))
        ax.set_yticks(np.arange(len(departments)))
        ax.set_xticklabels([f"{h:02d}:00" for h in hours])
        ax.set_yticklabels(departments)
        
        # グリッドの追加
        ax.set_xticks(np.arange(len(hours)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(departments)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", size=0)
        
        # カラーバーの追加
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('リスクレベル', rotation=270, labelpad=20)
        
        # タイトル
        plt.title('部署別セキュリティリスク ヒートマップ（24時間）', fontsize=16, pad=20)
        plt.xlabel('時刻', fontsize=12)
        plt.ylabel('部署', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_attack_path_3d(self, incident_data: Dict) -> go.Figure:
        """3D攻撃経路の可視化"""
        # ネットワークグラフの作成
        G = nx.Graph()
        
        # ノードの追加（影響を受けたシステム）
        affected_systems = incident_data.get("affected_systems", [])
        for i, system in enumerate(affected_systems):
            G.add_node(system, pos=(i, i, 0), node_type="system")
        
        # 攻撃者ノード
        attacker = incident_data.get("attacker", "Attacker")
        G.add_node(attacker, pos=(-1, -1, 1), node_type="attacker")
        
        # エッジの追加（攻撃経路）
        attack_paths = incident_data.get("attack_paths", [])
        for path in attack_paths:
            if len(path) >= 2:
                for i in range(len(path)-1):
                    G.add_edge(path[i], path[i+1])
        
        # 3D座標の生成
        pos = nx.spring_layout(G, dim=3, seed=42)
        
        # Plotlyでの3D可視化
        edge_trace = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_trace.append(go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(width=4, color='red'),
                hoverinfo='none'
            ))
        
        # ノードのプロット
        node_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='markers+text',
            name='Systems',
            marker=dict(
                size=20,
                color=[],
                colorscale='Viridis',
                line=dict(width=2)
            ),
            text=[],
            textposition="top center",
            hoverinfo='text'
        )
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['z'] += tuple([z])
            node_trace['text'] += tuple([node])
            
            # ノードタイプによる色分け
            if G.nodes[node].get('node_type') == 'attacker':
                node_trace['marker']['color'] += tuple(['red'])
            else:
                node_trace['marker']['color'] += tuple(['blue'])
        
        # 図の作成
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="3D攻撃経路マップ",
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False)
            ),
            margin=dict(b=20, l=5, r=5, t=40),
            height=600
        )
        
        return fig
    
    def _get_phase_color(self, phase: str) -> str:
        """フェーズに応じた色の取得"""
        phase_colors = {
            "reconnaissance": "#2196F3",
            "initial_access": "#9C27B0",
            "privilege_escalation": "#F44336",
            "lateral_movement": "#FF9800",
            "data_collection": "#FFC107",
            "data_exfiltration": "#FF5722",
            "cleanup": "#795548"
        }
        return phase_colors.get(phase, "#999999")

# =====================================
# フォレンジック支援システム
# =====================================

class ForensicAssistanceSystem:
    """フォレンジック支援システム"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        self.evidence_store = {}
        
    async def preserve_evidence(self, incident: IncidentMetadata) -> Dict:
        """証拠保全の実行"""
        evidence_package = {
            "incident_id": incident.incident_id,
            "preservation_time": datetime.now(),
            "evidence_items": []
        }
        
        # 1. 揮発性データの保存
        volatile_data = await self._capture_volatile_data(incident)
        evidence_package["evidence_items"].append(volatile_data)
        
        # 2. チェーンブロックの保存
        chain_evidence = self._export_chain_blocks(incident)
        evidence_package["evidence_items"].append(chain_evidence)
        
        # 3. ログファイルの保存
        log_evidence = await self._collect_logs(incident)
        evidence_package["evidence_items"].append(log_evidence)
        
        # 4. ハッシュ値の計算と保存
        evidence_hash = self._calculate_package_hash(evidence_package)
        evidence_package["hash"] = evidence_hash
        
        # 5. 改竄防止パッケージの作成
        sealed_package = self._create_forensic_container(evidence_package)
        
        return sealed_package
    
    async def _capture_volatile_data(self, incident: IncidentMetadata) -> ForensicEvidence:
        """揮発性データのキャプチャ"""
        volatile_data = {
            "capture_time": datetime.now().isoformat(),
            "active_connections": self._get_active_connections(),
            "running_processes": self._get_running_processes(),
            "memory_snapshot": self._capture_memory_snapshot(),
            "affected_users": incident.affected_users,
            "affected_systems": incident.affected_systems
        }
        
        return ForensicEvidence(
            evidence_id=f"VOLATILE-{incident.incident_id}",
            incident_id=incident.incident_id,
            timestamp=datetime.now(),
            evidence_type="volatile",
            data=volatile_data
        )
    
    def _export_chain_blocks(self, incident: IncidentMetadata) -> ForensicEvidence:
        """関連するチェーンブロックのエクスポート"""
        blocks_data = []
        
        # ユーザーチェーンから収集
        for user_id in incident.affected_users:
            if user_id in self.chain_manager.user_chains:
                user_chain = self.chain_manager.user_chains[user_id]
                for block in user_chain.blocks[-100:]:  # 最新100ブロック
                    blocks_data.append({
                        "chain_type": "user",
                        "user_id": user_id,
                        "block_index": block.index,
                        "hash": block.hash,
                        "timestamp": block.metadata.get("timestamp"),
                        "security_mode": block.metadata.get("security_mode"),
                        "divergence": block.divergence,
                        "metadata": self._sanitize_metadata(block.metadata)
                    })
        
        # 部署チェーンから収集
        for dept_name, chain in self.chain_manager.department_chains.items():
            relevant_blocks = [
                block for block in chain.blocks[-200:]
                if any(user in str(block.metadata) for user in incident.affected_users)
            ]
            
            for block in relevant_blocks:
                blocks_data.append({
                    "chain_type": "department",
                    "department": dept_name,
                    "block_index": block.index,
                    "hash": block.hash,
                    "timestamp": block.metadata.get("timestamp"),
                    "security_mode": block.metadata.get("security_mode"),
                    "divergence": block.divergence,
                    "metadata": self._sanitize_metadata(block.metadata)
                })
        
        return ForensicEvidence(
            evidence_id=f"BLOCKS-{incident.incident_id}",
            incident_id=incident.incident_id,
            timestamp=datetime.now(),
            evidence_type="blockchain",
            data={"blocks": blocks_data, "total_blocks": len(blocks_data)}
        )
    
    async def _collect_logs(self, incident: IncidentMetadata) -> ForensicEvidence:
        """関連ログの収集"""
        logs = {
            "security_logs": [],
            "application_logs": [],
            "system_logs": [],
            "network_logs": []
        }
        
        # タイムウィンドウの設定（インシデント検知の前後1時間）
        start_time = incident.detection_time - timedelta(hours=1)
        end_time = incident.detection_time + timedelta(hours=1)
        
        # セキュリティログの収集（シミュレーション）
        logs["security_logs"] = [
            {
                "timestamp": (start_time + timedelta(minutes=i*5)).isoformat(),
                "event_id": f"SEC-{i}",
                "severity": "high" if i % 3 == 0 else "medium",
                "description": f"Security event related to {incident.affected_users[0] if incident.affected_users else 'unknown'}"
            }
            for i in range(24)
        ]
        
        return ForensicEvidence(
            evidence_id=f"LOGS-{incident.incident_id}",
            incident_id=incident.incident_id,
            timestamp=datetime.now(),
            evidence_type="logs",
            data=logs
        )
    
    def _create_forensic_container(self, evidence_package: Dict) -> Dict:
        """改竄防止フォレンジックコンテナの作成"""
        container = {
            "format_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "evidence_package": evidence_package,
            "integrity": {
                "hash_algorithm": "SHA256",
                "package_hash": evidence_package.get("hash", ""),
                "signature": self._create_digital_signature(evidence_package)
            },
            "chain_of_custody": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "evidence_collected",
                    "performed_by": "ForensicAssistanceSystem",
                    "hash": evidence_package.get("hash", "")
                }
            ]
        }
        
        return container
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """メタデータのサニタイズ（機密情報の除去）"""
        sanitized = metadata.copy()
        
        # 機密フィールドの除去または匿名化
        sensitive_fields = ["password", "token", "secret", "key"]
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "[REDACTED]"
        
        return sanitized
    
    def _calculate_package_hash(self, package: Dict) -> str:
        """証拠パッケージのハッシュ計算"""
        # 辞書を安定したJSON文字列に変換
        package_str = json.dumps(package, sort_keys=True, default=str)
        return hashlib.sha256(package_str.encode()).hexdigest()
    
    def _create_digital_signature(self, data: Dict) -> str:
        """デジタル署名の作成（シミュレーション）"""
        # 実際の環境では秘密鍵を使用して署名
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha512(data_str.encode()).hexdigest()
    
    def _get_active_connections(self) -> List[Dict]:
        """アクティブな接続情報の取得（シミュレーション）"""
        return [
            {
                "local_address": "192.168.1.100:445",
                "remote_address": "192.168.1.200:50123",
                "state": "ESTABLISHED",
                "process": "svchost.exe"
            }
        ]
    
    def _get_running_processes(self) -> List[Dict]:
        """実行中プロセスの取得（シミュレーション）"""
        return [
            {
                "pid": 1234,
                "name": "powershell.exe",
                "user": "attacker",
                "cpu_percent": 45.2,
                "memory_mb": 128
            }
        ]
    
    def _capture_memory_snapshot(self) -> Dict:
        """メモリスナップショットの取得（シミュレーション）"""
        return {
            "snapshot_id": f"MEM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "size_mb": 8192,
            "capture_method": "live_capture",
            "compressed": True
        }

# =====================================
# 相関分析システム
# =====================================

class CorrelationAnalysisSystem:
    """インシデント相関分析システム"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        
    def analyze_correlations(self, incident: IncidentMetadata) -> Dict:
        """インシデントの相関分析"""
        correlations = {
            "incident_id": incident.incident_id,
            "time_correlation": self._analyze_time_correlation(incident),
            "user_correlation": self._analyze_user_correlation(incident),
            "technique_correlation": self._analyze_technique_correlation(incident),
            "summary": {}
        }
        
        # 相関サマリーの生成
        correlations["summary"] = self._generate_correlation_summary(correlations)
        
        return correlations
    
    def _analyze_time_correlation(self, incident: IncidentMetadata) -> Dict:
        """時間的相関の分析"""
        time_correlation = {
            "concurrent_incidents": [],
            "historical_patterns": [],
            "periodic_pattern": None
        }
        
        # 同時期の他の異常を検出
        detection_window = timedelta(hours=1)
        start_time = incident.detection_time - detection_window
        end_time = incident.detection_time + detection_window
        
        # 全部署のチェーンをチェック
        for dept_name, chain in self.chain_manager.department_chains.items():
            for block in chain.blocks:
                block_time_str = block.metadata.get("timestamp", "")
                if block_time_str:
                    try:
                        block_time = datetime.strptime(block_time_str, "%Y-%m-%d %H:%M:%S")
                        if start_time <= block_time <= end_time:
                            if block.metadata.get("security_mode") in ["suspicious", "critical"]:
                                time_correlation["concurrent_incidents"].append({
                                    "department": dept_name,
                                    "user": block.metadata.get("user_id"),
                                    "timestamp": block_time_str,
                                    "severity": block.metadata.get("security_mode")
                                })
                    except:
                        continue
        
        # 過去の類似パターンを検索
        time_correlation["historical_patterns"] = self._find_historical_patterns(incident)
        
        # 周期性の検出
        time_correlation["periodic_pattern"] = self._detect_periodicity(incident)
        
        return time_correlation
    
    def _analyze_user_correlation(self, incident: IncidentMetadata) -> Dict:
        """ユーザー相関の分析"""
        user_correlation = {
            "same_ip_users": [],
            "behavior_similarity": [],
            "access_pattern_match": []
        }
        
        if not incident.affected_users:
            return user_correlation
        
        primary_user = incident.affected_users[0]
        
        # 同一IPからの他ユーザーを検索
        if primary_user in self.chain_manager.user_chains:
            primary_chain = self.chain_manager.user_chains[primary_user]
            
            # 最新のIPアドレスを取得
            recent_ips = set()
            for block in primary_chain.blocks[-20:]:
                ip = block.metadata.get("source_ip")
                if ip:
                    recent_ips.add(ip)
            
            # 他のユーザーチェーンで同じIPをチェック
            for user_id, user_chain in self.chain_manager.user_chains.items():
                if user_id != primary_user:
                    for block in user_chain.blocks[-20:]:
                        if block.metadata.get("source_ip") in recent_ips:
                            user_correlation["same_ip_users"].append({
                                "user_id": user_id,
                                "shared_ip": block.metadata.get("source_ip"),
                                "timestamp": block.metadata.get("timestamp")
                            })
                            break
        
        # 行動パターンの類似性分析
        user_correlation["behavior_similarity"] = self._calculate_behavior_similarity(
            primary_user
        )
        
        return user_correlation
    
    def _analyze_technique_correlation(self, incident: IncidentMetadata) -> Dict:
        """攻撃手法の相関分析"""
        technique_correlation = {
            "known_apt_match": [],
            "internal_threat_score": 0.0,
            "technique_combinations": []
        }
        
        # 既知のAPTパターンとの照合
        apt_patterns = {
            "APT28": {
                "techniques": ["powershell", "lateral_movement", "data_exfiltration"],
                "confidence": 0.0
            },
            "APT29": {
                "techniques": ["privilege_escalation", "stealth", "long_term_access"],
                "confidence": 0.0
            }
        }
        
        # インシデントの特徴抽出
        incident_features = self._extract_incident_features(incident)
        
        # APTパターンとのマッチング
        for apt_name, apt_data in apt_patterns.items():
            match_score = self._calculate_technique_match(
                incident_features,
                apt_data["techniques"]
            )
            if match_score > 0.3:
                technique_correlation["known_apt_match"].append({
                    "apt_group": apt_name,
                    "confidence": match_score,
                    "matched_techniques": apt_data["techniques"]
                })
        
        # 内部脅威スコアの計算
        technique_correlation["internal_threat_score"] = self._calculate_internal_threat_score(
            incident,
            incident_features
        )
        
        return technique_correlation
    
    def _find_historical_patterns(self, incident: IncidentMetadata) -> List[Dict]:
        """過去の類似パターンの検索"""
        patterns = []
        
        # 過去30日間の類似インシデントを検索
        lookback_days = 30
        lookback_time = incident.detection_time - timedelta(days=lookback_days)
        
        # 簡易的なパターンマッチング
        for dept_name, chain in self.chain_manager.department_chains.items():
            similar_blocks = []
            
            for block in chain.blocks:
                if block.metadata.get("security_mode") == incident.severity:
                    if block.metadata.get("attack_type") == incident.attack_type:
                        similar_blocks.append(block)
            
            if similar_blocks:
                patterns.append({
                    "department": dept_name,
                    "occurrence_count": len(similar_blocks),
                    "first_seen": similar_blocks[0].metadata.get("timestamp"),
                    "last_seen": similar_blocks[-1].metadata.get("timestamp")
                })
        
        return patterns
    
    def _detect_periodicity(self, incident: IncidentMetadata) -> Optional[Dict]:
        """周期性の検出"""
        # 同じ攻撃タイプの発生時刻を収集
        occurrence_times = []
        
        for dept_name, chain in self.chain_manager.department_chains.items():
            for block in chain.blocks:
                if block.metadata.get("attack_type") == incident.attack_type:
                    timestamp_str = block.metadata.get("timestamp")
                    if timestamp_str:
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            occurrence_times.append(timestamp)
                        except:
                            continue
        
        if len(occurrence_times) < 3:
            return None
        
        # 時間間隔の計算
        occurrence_times.sort()
        intervals = []
        for i in range(1, len(occurrence_times)):
            interval = (occurrence_times[i] - occurrence_times[i-1]).total_seconds() / 3600
            intervals.append(interval)
        
        # 周期性の判定（簡易版）
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            std_interval = np.std(intervals)
            
            if std_interval < avg_interval * 0.3:  # 変動が30%以内
                return {
                    "detected": True,
                    "average_interval_hours": avg_interval,
                    "confidence": 1 - (std_interval / avg_interval)
                }
        
        return {"detected": False}
    
    def _calculate_behavior_similarity(self, target_user: str) -> List[Dict]:
        """行動パターンの類似性計算"""
        similarities = []
        
        if target_user not in self.chain_manager.user_chains:
            return similarities
        
        target_chain = self.chain_manager.user_chains[target_user]
        target_baseline = target_chain.baseline_behavior
        
        # 他のユーザーとの比較
        for user_id, user_chain in self.chain_manager.user_chains.items():
            if user_id != target_user:
                similarity_score = self._compare_baselines(
                    target_baseline,
                    user_chain.baseline_behavior
                )
                
                if similarity_score > 0.6:
                    similarities.append({
                        "user_id": user_id,
                        "similarity_score": similarity_score,
                        "matching_patterns": self._get_matching_patterns(
                            target_baseline,
                            user_chain.baseline_behavior
                        )
                    })
        
        return sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)
    
    def _extract_incident_features(self, incident: IncidentMetadata) -> List[str]:
        """インシデントの特徴抽出"""
        features = []
        
        # 攻撃タイプから特徴抽出
        if incident.attack_type == "privilege_escalation":
            features.extend(["powershell", "admin_tools"])
        elif incident.attack_type == "lateral_movement":
            features.extend(["lateral_movement", "network_discovery"])
        elif incident.attack_type == "data_exfiltration":
            features.extend(["data_exfiltration", "large_transfer"])
        
        return features
    
    def _calculate_technique_match(self, incident_features: List[str], 
                                 apt_techniques: List[str]) -> float:
        """技術的特徴のマッチング率計算"""
        if not apt_techniques:
            return 0.0
        
        matches = sum(1 for tech in apt_techniques if tech in incident_features)
        return matches / len(apt_techniques)
    
    def _calculate_internal_threat_score(self, incident: IncidentMetadata, 
                                       features: List[str]) -> float:
        """内部脅威スコアの計算"""
        score = 0.0
        
        # 内部脅威の指標
        if incident.affected_users:
            user_id = incident.affected_users[0]
            if user_id in self.chain_manager.user_chains:
                user_chain = self.chain_manager.user_chains[user_id]
                
                # 長期間の正常アクセス後の異常
                if len(user_chain.blocks) > 100:
                    normal_blocks = sum(
                        1 for block in user_chain.blocks[:80]
                        if block.metadata.get("security_mode") == "normal"
                    )
                    if normal_blocks > 70:
                        score += 0.3
                
                # 通常の業務時間内のアクセス
                if hasattr(user_chain, "baseline_behavior"):
                    typical_hours = user_chain.baseline_behavior.get("typical_hours", [])
                    if 9 <= incident.detection_time.hour <= 17:
                        score += 0.2
                
                # 既知の内部システムへのアクセス
                if "lateral_movement" in features:
                    score += 0.3
        
        return min(score, 1.0)
    
    def _compare_baselines(self, baseline1: Dict, baseline2: Dict) -> float:
        """ベースライン行動の比較"""
        if not baseline1 or not baseline2:
            return 0.0
        
        similarity = 0.0
        comparison_count = 0
        
        # 活動時間帯の比較
        hours1 = set(baseline1.get("typical_hours", []))
        hours2 = set(baseline2.get("typical_hours", []))
        if hours1 and hours2:
            similarity += len(hours1 & hours2) / len(hours1 | hours2)
            comparison_count += 1
        
        # アクセスディレクトリの比較
        dirs1 = set(baseline1.get("common_directories", []))
        dirs2 = set(baseline2.get("common_directories", []))
        if dirs1 and dirs2:
            similarity += len(dirs1 & dirs2) / len(dirs1 | dirs2)
            comparison_count += 1
        
        # 使用プロセスの比較
        procs1 = set(baseline1.get("common_processes", []))
        procs2 = set(baseline2.get("common_processes", []))
        if procs1 and procs2:
            similarity += len(procs1 & procs2) / len(procs1 | procs2)
            comparison_count += 1
        
        return similarity / comparison_count if comparison_count > 0 else 0.0
    
    def _get_matching_patterns(self, baseline1: Dict, baseline2: Dict) -> List[str]:
        """一致するパターンの取得"""
        patterns = []
        
        # 共通の活動時間
        hours1 = set(baseline1.get("typical_hours", []))
        hours2 = set(baseline2.get("typical_hours", []))
        common_hours = hours1 & hours2
        if common_hours:
            patterns.append(f"活動時間帯: {sorted(common_hours)}")
        
        # 共通のディレクトリ
        dirs1 = set(baseline1.get("common_directories", []))
        dirs2 = set(baseline2.get("common_directories", []))
        common_dirs = dirs1 & dirs2
        if common_dirs:
            patterns.append(f"アクセスディレクトリ: {len(common_dirs)}箇所")
        
        return patterns
    
    def _generate_correlation_summary(self, correlations: Dict) -> Dict:
        """相関分析のサマリー生成"""
        summary = {
            "risk_level": "low",
            "key_findings": [],
            "recommendations": []
        }
        
        # 時間相関の評価
        concurrent_count = len(correlations["time_correlation"]["concurrent_incidents"])
        if concurrent_count > 3:
            summary["risk_level"] = "high"
            summary["key_findings"].append(
                f"同時期に{concurrent_count}件の異常を検出 - 組織的攻撃の可能性"
            )
            summary["recommendations"].append(
                "全社的なセキュリティ監視の強化"
            )
        
        # ユーザー相関の評価
        if correlations["user_correlation"]["same_ip_users"]:
            summary["key_findings"].append(
                "複数ユーザーが同一IPを使用 - アカウント侵害の可能性"
            )
            summary["recommendations"].append(
                "影響ユーザーの認証情報リセット"
            )
        
        # 技術相関の評価
        if correlations["technique_correlation"]["known_apt_match"]:
            apt_match = correlations["technique_correlation"]["known_apt_match"][0]
            summary["risk_level"] = "critical"
            summary["key_findings"].append(
                f"{apt_match['apt_group']}との手法一致率: {apt_match['confidence']:.0%}"
            )
            summary["recommendations"].append(
                "脅威インテリジェンスに基づく対策の実施"
            )
        
        return summary

# =====================================
# 統合インシデント対応システム
# =====================================

class IntegratedIncidentResponseSystem:
    """統合インシデント対応システム"""
    
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager
        self.immediate_response = ImmediateResponseSystem(chain_manager)
        self.initial_analysis = InitialAnalysisSystem(chain_manager)
        self.detailed_investigation = DetailedInvestigationSystem(chain_manager)
        self.recovery_assistance = RecoveryAssistanceSystem(chain_manager)
        self.forensic_assistance = ForensicAssistanceSystem(chain_manager)
        self.correlation_analysis = CorrelationAnalysisSystem(chain_manager)
        self.visualization = IncidentVisualizationSystem()
        
        # インシデント履歴
        self.incident_history = {}
        
    async def respond_to_incident(self, event: Dict, detection_result: Dict) -> Dict:
        """インシデント対応の統合実行"""
        print(f"\n{'='*60}")
        print(f"🚨 インシデント対応開始")
        print(f"{'='*60}")
        
        response_timeline = []
        
        # Phase 1: 即時対応（〜30秒）
        phase1_start = datetime.now()
        phase1_result = await self.immediate_response.process_incident(
            event, 
            detection_result
        )
        response_timeline.append({
            "phase": "immediate_response",
            "duration": (datetime.now() - phase1_start).total_seconds(),
            "result": phase1_result
        })
        
        incident = phase1_result["incident"]
        
        # Phase 2: 初動分析（〜5分）
        phase2_start = datetime.now()
        phase2_result = await self.initial_analysis.analyze_incident(
            incident,
            phase1_result["evidence_task"]
        )
        response_timeline.append({
            "phase": "initial_analysis",
            "duration": (datetime.now() - phase2_start).total_seconds(),
            "result": phase2_result
        })
        
        # Phase 3: 詳細調査（〜30分）
        phase3_start = datetime.now()
        phase3_result = await self.detailed_investigation.investigate_incident(
            incident,
            phase2_result
        )
        response_timeline.append({
            "phase": "detailed_investigation",
            "duration": (datetime.now() - phase3_start).total_seconds(),
            "result": phase3_result
        })
        
        # Phase 4: 復旧支援（〜数時間）
        phase4_start = datetime.now()
        phase4_result = await self.recovery_assistance.assist_recovery(
            incident,
            phase3_result
        )
        response_timeline.append({
            "phase": "recovery_assistance",
            "duration": (datetime.now() - phase4_start).total_seconds(),
            "result": phase4_result
        })
        
        # 追加分析
        correlation_result = self.correlation_analysis.analyze_correlations(incident)
        forensic_result = await self.forensic_assistance.preserve_evidence(incident)
        
        # レポート生成
        final_report = self._generate_incident_report(
            incident,
            response_timeline,
            correlation_result,
            forensic_result
        )
        
        # インシデント履歴に保存
        self.incident_history[incident.incident_id] = final_report
        
        # 可視化
        self._create_visualizations(final_report)
        
        return final_report
    
    def _generate_incident_report(self, incident: IncidentMetadata,
                                response_timeline: List[Dict],
                                correlation_result: Dict,
                                forensic_result: Dict) -> Dict:
        """最終インシデントレポートの生成"""
        report = {
            "incident_summary": {
                "id": incident.incident_id,
                "detection_time": incident.detection_time.isoformat(),
                "severity": incident.severity,
                "attack_type": incident.attack_type,
                "status": incident.status,
                "affected_scope": {
                    "users": incident.affected_users,
                    "systems": incident.affected_systems
                }
            },
            "response_timeline": response_timeline,
            "analysis_results": {
                "root_cause": response_timeline[2]["result"]["root_cause_analysis"],
                "lateral_risk": response_timeline[2]["result"]["lateral_movement_risk"],
                "correlations": correlation_result
            },
            "mitigation_actions": {
                "immediate_containment": response_timeline[0]["result"]["containment_actions"],
                "recovery_procedures": response_timeline[3]["result"]["recovery_procedures"],
                "prevention_measures": response_timeline[3]["result"]["prevention_measures"]
            },
            "forensic_evidence": {
                "evidence_id": forensic_result.get("evidence_package", {}).get("incident_id"),
                "preservation_time": forensic_result.get("created_at"),
                "hash": forensic_result.get("integrity", {}).get("package_hash")
            },
            "total_response_time": sum(phase["duration"] for phase in response_timeline),
            "report_generated": datetime.now().isoformat()
        }
        
        return report
    
    def _create_visualizations(self, report: Dict):
        """レポートの可視化"""
        # タイムライン可視化
        timeline_data = report["analysis_results"]["root_cause"].get("evidence_chain", [])
        if timeline_data:
            timeline_viz = self.visualization.create_timeline_visualization({
                "phases": {item["phase"]: item for item in timeline_data},
                "events": []
            })
            # timeline_viz.show()  # 実際の環境では表示
        
        # インパクトヒートマップ
        impact_data = report["incident_summary"]["affected_scope"]
        heatmap = self.visualization.create_impact_heatmap(impact_data)
        # plt.show()  # 実際の環境では表示
        
        print("\n📊 可視化レポートが生成されました")

# =====================================
# 使用例とテスト
# =====================================

async def test_incident_response():
    """インシデント対応システムのテスト"""
    
    # ダミーのchain_managerを作成
    class DummyChainManager:
        def __init__(self):
            self.department_chains = {}
            self.user_chains = {}
            self.user_department_map = {"suzuki_m": "sales"}
            self.cross_department_access = {}
    
    chain_manager = DummyChainManager()
    
    # テストイベント（攻撃シナリオ）
    test_event = {
        "timestamp": "2025-03-14 10:14:00",
        "user_id": "suzuki_m",
        "department": "sales",
        "operation": "FileCopy",
        "file_path": "\\\\fileserver\\sales\\confidential\\customer_list.xlsx",
        "file_size_kb": 45000,
        "process_name": "powershell.exe",
        "destination_ip": "203.0.113.50",
        "source_ip": "192.168.1.120",
        "status": "SUCCESS"
    }
    
    # 検知結果（chain_managerからの出力を想定）
    detection_result = {
        "status": "critical",
        "divergence": 45.8,
        "alert_level": "HIGH",
        "cross_dept_warning": True
    }
    
    # 統合システムの初期化
    response_system = IntegratedIncidentResponseSystem(chain_manager)
    
    # インシデント対応の実行
    response = await response_system.respond_to_incident(test_event, detection_result)
    
    # 結果のサマリー表示
    print(f"\n{'='*60}")
    print("📋 インシデント対応完了")
    print(f"{'='*60}")
    print(f"インシデントID: {response['incident_summary']['id']}")
    print(f"深刻度: {response['incident_summary']['severity']}")
    print(f"攻撃タイプ: {response['incident_summary']['attack_type']}")
    print(f"総対応時間: {response['total_response_time']:.2f}秒")
    print(f"\n根本原因: {response['analysis_results']['root_cause']['vulnerability_type']}")
    print(f"横展開リスク: {response['analysis_results']['lateral_risk']['risk_score']:.2f}")
    
    # 推奨対策の表示
    print("\n🛡️ 推奨対策:")
    for measure in response['mitigation_actions']['prevention_measures']:
        print(f"  - {measure['measure']} (優先度: {measure['priority']})")
    
    return response

# メイン実行
if __name__ == "__main__":
    # 非同期実行
    asyncio.run(test_incident_response())
