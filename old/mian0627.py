# -------------------------------
# 🌐 NumPy（BLAS/LAPACK）Code
# -------------------------------

# 🌐 標準ライブラリ
import os
import sys
import json
import time
import random
import pickle
import gzip
import hashlib

from pathlib import Path
from collections import deque, OrderedDict, Counter,defaultdict
from datetime import datetime, timedelta

from functools import lru_cache
from multiprocessing import Pool, Queue
from typing import Tuple, Dict, Any, Optional
import asyncio
import concurrent.futures

# 🧮 数値・データ処理
import math
import numpy as np
import pandas as pd
from numba import njit, prange

# 別名エイリアス（datetime衝突時のみ!）
from datetime import datetime as dtmod

# 🧪 機械学習/統計系
from cuml.cluster import KMeans
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# 🛠️ 設定・型安全
from dataclasses import dataclass, field

# -------------------------------
# 🧬config関数
# -------------------------------
@dataclass
class ChainTuneConfig:

    # === スケーラビリティ ===
    enable_scalable_backend: bool = True
    enable_user_scalable_backend: bool = True

    # === EMA/重み係数（組織用・個人用） ===
    ema_alpha_org:  float = 0.3        # 組織EMA平滑化係数
    ema_alpha_user: float = 0.6        # 個人EMA平滑化係数
    alpha_org: float = 1.2             # 組織：normal_div 重み
    beta_org: float = 0.7              # 組織：severity_level重み
    gamma_org: float = 0.5             # 組織：action_magnitude重み
    alpha_user: float = 1.3            # 個人：normal_div 重み
    beta_user: float = 0.9             # 個人：severity_level重み
    gamma_user: float = 0.8            # 個人：action_magnitude重み

    # === 判定しきい値（suspicious, investigating, critical） ===
    threshold_suspicious_org: float = 15.0      # 組織：suspicious判定しきい値
    threshold_suspicious_user: float = 20.0     # 個人：suspicious判定しきい値
    threshold_investigating_org: float = 8.0   # 組織：investigating判定しきい値（※追加）
    threshold_investigating_user: float = 10.0  # 個人：investigating判定しきい値（※追加）
    trust_error_threshold_org: float = 0.5      # 組織：信頼スコアでcritical
    trust_error_threshold_user: float = 0.65    # 個人：信頼スコアでcritical
    threshold_div_suspicious : float = 14.0
    threshold_div_crittcal : float = 25.0

    # === Divergence×信頼スコアの組み合わせによる追加判定 ===
    trust_score_investigating_1: float = 0.65   # investigating判定その1（信頼スコア）
    trust_score_investigating_2: float = 0.70   # investigating判定その2（信頼スコア）
    normal_div_investigating_1: float = 10.0    # investigating判定その1（normal_div）
    normal_div_investigating_2: float = 8.0     # investigating判定その2（normal_div）

    # === フォーク・同期関連しきい値 ===
    fork_threshold_user: float = 12.0           # 個人：チェーン分岐閾値
    fork_threshold_org: float = 15.0            # 組織：チェーン分岐閾値

    # 🌟 キャッシュ・ストレージ
    cache_size:        int = 10_000
    max_memory_mb:     int = 1_000
    block_history_min: int = 60
    block_history_hr:  int = 24
    block_history_day: int = 30

    # 🌟 サンプリング
    target_sample_rate: float = 0.10
    high_load_thresh:   float = 0.80

    # 🌟 部署ごとのしきい値
    dept_threshold_table: dict = field(default_factory=lambda: {
        "sales":       {"morning": 12.0, "afternoon": 10.0, "evening": 11.0},
        "engineering": {"morning": 18.0, "afternoon": 10.0, "evening": 20.0},
        "finance":     {"morning": 12.0, "afternoon":  9.0, "evening": 11.0},
        "hr":          {"morning": 14.0, "afternoon":  9.0, "evening": 12.0},
        "executive":   {"morning": 15.0, "afternoon": 12.0, "evening": 18.0}
    })

    # 🌟 ファイルパス一元管理
    security_chain_filepath:  str = "realistic_security_logs.json"
    error_chain_filepath:     str = "attack_only_logs.json"
    suspicious_events_file:   str = "chain_attackments.json"
    zero_day_events_file:     str = "attackments.json"
    normal_test_events_file:  str = "normal_test_events.json"
    attack_patterns_file:     str = "attack_patterns.json"


    # 🌟 意味テンソルの重み係数（L1/L2）
    weights_l1: dict = field(default_factory=lambda: {
        "severity_level": 1.0,
        "action_magnitude": 0.5,
        "trust_score": 1.0,
        "threat_context": 0.7,
        "event_score": 0.4,
        "security_mode": 0.6,
        "is_confidential_access": 2.5,
        "is_cross_dept_access": 2.0,
        "is_external_network": 2.5,
        "permission_level_diff": 1.5,
        "operation_success_flag": 1.8
    })
    weights_l2: dict = field(default_factory=lambda: {
        "severity_level": 0.7,
        "action_magnitude": 0.3,
        "trust_score": 0.9,
        "threat_context": 1.0,
        "event_score": 0.7,
        "security_mode": 0.8,
        "is_confidential_access": 2.5,
        "is_cross_dept_access": 2.0,
        "is_external_network": 2.5,
        "permission_level_diff": 1.5,
        "operation_success_flag": 1.8
    })

    def to_org_dict(self):
        return {
            "alpha": self.alpha_org,
            "beta": self.beta_org,
            "gamma": self.gamma_org,
            "ema_alpha": self.ema_alpha_org,
            "threshold_suspicious": self.threshold_suspicious_org,
            "trust_error_threshold": self.trust_error_threshold_org,
        }

    def to_user_dict(self):
        return {
            "alpha": self.alpha_user,
            "beta": self.beta_user,
            "gamma": self.gamma_user,
            "ema_alpha": self.ema_alpha_user,
            "threshold_suspicious": self.threshold_suspicious_user,
            "trust_error_threshold": self.trust_error_threshold_user,
        }

    # 🌟 Divergence
    divergence_max: float = 120.0

    # 🌟 セキュリティモードのマッピング
    security_mode_mapping: dict = field(default_factory=lambda: {
        "normal": 0.0,
        "suspicious": 1.0,
        "critical": 2.0,
        "investigating": 1.5,
        "latent_suspicious": 1.2  # ←追加！
    })

    reverse_security_mode_mapping: dict = field(init=False)

    def __post_init__(self):
        # REVERSE_SECURITY_MODE_MAPPINGの自動生成
        self.reverse_security_mode_mapping = {v: k for k, v in self.security_mode_mapping.items()}

# ===== security_constants.py =====

# --- 1. 操作ごとの基本スコア ---
OPERATION_SCORES = {
    "FileRead": 15, "FileWrite": 30, "FileCopy": 60, "FileDelete": 50,
    "FileMove": 35, "ProcessCreate": 40, "ProcessTerminate": 20,
    "NetworkConnect": 35, "NetworkListen": 30, "Login": 10,
    "LoginFailed": 40, "Logout": 2
}

# --- 2. パス種別→加点値 ---
PATH_SCORES = {
    "high": [
        ("\\system32\\", 50), ("\\syswow64\\", 50), ("\\admin\\", 50),
        ("\\config\\", 50), ("\\windows\\", 50), ("\\program files\\", 50), ("\\programdata\\", 50),
    ],
    "sensitive": [
        ("\\finance\\", 45), ("\\hr\\", 35), ("\\payroll\\", 35), ("\\経理\\", 45), ("\\人事\\", 35),
        ("\\給与\\", 35), ("\\役員\\", 45), ("\\executive\\", 45), ("\\confidential\\", 50),
    ],
    "normal": [
        ("\\sales\\", 12), ("\\営業\\", 12), ("\\顧客\\", 12), ("\\customer\\", 12),
        ("\\契約\\", 12), ("\\contract\\", 12), ("\\source\\", 12),
        ("\\backup\\", 18), ("\\audit\\", 18), ("\\log\\", 18), ("\\archive\\", 18),
    ]
}

# --- 3. ファイルサイズ閾値 ---
FILE_SIZE_THRESHOLDS = [
    (1000, 60),
    (500, 50),
    (300, 40),
    (100, 35),
    (50, 15)
]

# --- 4. プロセス加点 ---
PROCESS_SCORES = {
    "dangerous": [("powershell", 90), ("cmd", 85), ("wmic", 70), ("psexec", 70), ("net.exe", 70), ("reg.exe", 70)],
    "admin": [("mmc.exe", 20), ("regedit", 20), ("services.exe", 20), ("taskmgr", 20)],
    "archiver": [("7z", 25), ("winrar", 25), ("winzip", 25), ("gpg", 25), ("truecrypt", 25)],
}

# --- 5. 時間帯・週末 ---
TIME_PENALTY_SCORES = [
    ((0, 5), 30),
    ((5, 7), 20),
    ((22, 24), 25),
    ((12, 13), -5),
]
TIME_ANOMALY_SCORES = [
    ((9, 18), 0.0),
    ((7, 9), 0.2),
    ((18, 20), 0.2),
    ((6, 7), 0.4),
    ((20, 22), 0.4),
    ((5, 6), 0.6),
    ((22, 23), 0.6),
    ((0, 5), 0.9),   # 深夜帯
]
WEEKEND_BONUS = 0.3
MONDAY_MORNING = (0, 6, 8, 0.8)
FRIDAY_NIGHT = (4, 19, 22, 0.8)
BUSINESS_HOURS = {"start": 6, "end": 18}
HOURS_SINCE_ACCESS = 720

# --- 6. 重大度レベル ---
SEVERITY_LEVEL_THRESHOLDS = [
    (85, 1.0),
    (70, 0.9),
    (50, 0.7),
    (30, 0.5),
    (15, 0.3),
    (0,  0.1),
]

# --- 7. ベクトル変換 ---
ACTION_OPERATION_VECTORS = {
    "Login": np.array([1.0, 0.0, 0.0]),
    "Logout": np.array([0.0, 0.0, 1.0]),
    "LoginFailed": np.array([3.0, 0.0, 0.0]),
    "FileRead": np.array([0.0, 1.0, 0.0]),
    "NetworkConnect": np.array([0.0, 2.0, 0.0]),
    "NetworkListen": np.array([0.0, 1.0, 0.0]),
    "ProcessCreate": np.array([0.0, 0.8, 0.2]),
    "FileWrite": np.array([0.0, 0.0, 1.0]),
    "FileDelete": np.array([0.0, 0.0, 1.0]),
    "FileCopy": np.array([0.0, 0.3, 0.7]),
    "FileMove": np.array([0.0, 0.0, 1.0]),
    "ProcessTerminate": np.array([0.0, 0.0, 1.0]),
}
ACTION_DEFAULT_VECTOR = np.array([0.33, 0.33, 0.34])

def operation_type_vector(operation):
    return ACTION_OPERATION_VECTORS.get(operation, ACTION_DEFAULT_VECTOR)

# --- 8. 部署×リソース ---
DEPT_RESOURCE_SCORES = {
    ("sales", ("\\source\\", "\\dev\\", "\\git\\")): (15, -0.1),
    ("engineering", ("\\finance\\", "\\accounting\\")): (15, -0.1),
}

# --- 9. 外部IPスコア ---
EXTERNAL_IP_SCORES = {
    "internal": 20,
    "external": 60,
}

TRUSTED_HOSTNAMES = [
    "crm.company.com", "bank.example.com", "intranet.company.com"
]
TRUSTED_IPS = [
    "192.168.1.0/24", "172.16.0.0/12"
]

# --- 10. プライベートIP, ファイル拡張子 ---
PRIVATE_IP_PREFIXES = ("192.168.", "10.", "172.")
FILE_TYPE_MAPPING = {
    "office": (".docx", ".xlsx", ".pdf", ".pptx"),
    "exec": (".exe", ".bat", ".cmd", ".msi"),
}

# --- 11. 業務操作セット ---
BUSINESS_OPERATIONS = {
    "Login", "Logout", "FileRead", "FileWrite", "ProcessCreate", "NetworkConnect"
}

# --- 12. 機密フォルダ権限部署 ---
CONFIDENTIAL_ACCESS_DEPTS = ["engineering", "executive"]

# --- 13. セキュリティ重み ---
THREAT_CONTEXT_WEIGHTS = {
    "time_anomaly": 0.3,
    "outside_active_hours": 0.1,
    "external_network": 0.4,
    "gateway_access": 0.2,
    "dangerous_process": 0.3,
    "unknown_process": 0.1,
    "operation_failed": 0.2,
    "system_path_access": 0.2,
    "max_threat": 1.0,
}

# --- 14. 信頼度ロジック ---
TRUST_SCORES = {
    "operation_penalty": {
        "FileRead": 0.0,
        "FileWrite": 0.0,
        "FileDelete": -0.02,
        "FileCopy": -0.02,
        "ProcessCreate": -0.01,
        "LoginFailed": -0.4,
    },
    "first_access_penalty": 0.1,
    "recency_divisor": 8000.0,
    "recency_penalty_cap": 0.3,
    "internal_ip_penalty": 0.15,
    "external_ip_penalty": 0.35,
    "time_penalty": 0.1,
    "min_trust": 0.3,
    "max_trust": 1.0,
    "business_context_year_end_bonus": 0.1,
    "business_context_cross_dept_bonus": 0.2,
}

# --- 15. 判定閾値 ---
THRESHOLDS = {
    "failed_attempts_critical": 3,
    "large_file_kb": 100_000,
    "abnormal_access_count": 100,
    "pattern_count_threshold": 8,
    "repeat_count_threshold": 12,
    "anomaly_score_threshold": 3.0,
    "normal_div_threshold": 6.5,
    "trust_score_threshold": 0.80,
    "trust_score_high": 0.8,
    "trust_score_medium": 0.7,
    "trust_score_low": 0.72,
    "normal_div_high": 8.5,
    "pattern_accumulation_limit": 8
}

# --- 16. 部署と機密・業務パス ---
DEPT_PATH_RULES = {
    "sales": {
        # アクセス制限パス
        "restricted": ["\\finance\\", "\\hr\\", "\\payroll\\", "\\accounting\\", "\\bank\\"],
        "patterns": ["\\sales\\", "\\営業\\", "\\crm\\", "\\customer\\", "\\proposal\\"],  # ← 追加分
        "message": "営業が経理/人事リソースにアクセス"
    },
    "engineering": {
        "restricted": ["\\payroll\\", "\\hr\\", "\\finance\\"],
        "patterns": ["\\dev\\", "\\source\\", "\\repos\\", "\\開発\\", "\\git\\"],
        "message": "エンジニアが経理・人事リソースにアクセス"
    },
    "finance": {
        "restricted": ["\\source\\", "\\dev\\", "\\engineering\\", "\\ci_cd_pipeline\\"],
        "patterns": ["\\finance\\", "\\経理\\", "\\accounting\\", "\\財務\\", "\\budget\\"],
        "message": "経理が開発リソースにアクセス"
    },
    "hr": {
        "restricted": ["\\source\\", "\\dev\\", "\\engineering\\", "\\finance\\"],
        "patterns": ["\\hr\\", "\\人事\\", "\\payroll\\", "\\employee\\", "\\recruitment\\"],
        "message": "人事が開発・経理リソースにアクセス"
    },
    # 必要なら executive も追加
}

# --- 17. モード判定用イベントスコア閾値 ---
MODE_THRESHOLDS = {
    "critical": 40,
    "suspicious": 20,
    "investigating": 10,
}

# --- 18. 高リスクパスリスト ---
HIGH_RISK_PATHS = [
    "\\admin\\", "\\administrator\\", "\\config\\",
    "\\audit\\", "\\backup\\", "\\system32\\"
]

# --- 20. 部署ごとのIPレンジ ---
DEPT_IP_RANGES = {
    "sales": "192.168.1.",
    "engineering": "192.168.2.",
    "finance": "192.168.3.",
    "hr": "192.168.4."
}

# --- 21. ダイバージェンス/クラスタリング関連定数（追加分） ---

# 逸脱スコアの設定
DEVIATION_SCORES = {
    "hour_diff_divisor": 12.0,
    "hour_weight": 0.2,
    "unknown_directory": 0.3,
    "high_risk_directory": 0.3,
    "abnormal_file_size": 0.2,
    "unknown_process": 0.2,
    "dangerous_process": 0.3,
    "new_operation": 0.2,
    "rare_operation": 0.1,
    "unknown_internal_ip": 0.1,
    "unknown_external_ip": 0.3,
    "max_score": 1.0
}

FREQUENT_DIRECTORY_ACCESS_THRESHOLD = 10  # そのディレクトリが10回以上アクセスされたら“頻繁”と見なす
OPERATION_FREQUENCY_THRESHOLD = 0.01 # 操作頻度の閾値
RECENT_ACCESS_HOURS = 24 # 最近のアクセス時間（時間）
MIN_ACCESS_COUNT = 10  # アクセス頻度判定の閾値（例値。要調整）
NORMAL_ACCESS_GRACE_PERIOD = 24  # 正常アクセス猶予時間（h）
HIGH_FREQUENCY_OPERATION_RATIO = 0.2 # 高頻度操作の比率
HIGH_ALERT_MODES = ["critical", "suspicious", "latent_suspicious"]

# Divergence調整係数
DIVERGENCE_MULTIPLIERS = {
    "recent_normal_access": 0.7,
    "frequent_user": 0.8,
    "frequent_directory": 0.8,
    "safe_process": 0.9
}

# ビジネスコンテキスト係数
BUSINESS_CONTEXT_MULTIPLIERS = {
    "year_end": 0.7,
    "quarter_end": 0.7,
    "cross_dept_allowed": 0.6,
    "emergency": 0.5,
    "maintenance": 0.6
}

# Divergence調整係数（個人用）
DIVERGENCE_MULTIPLIERS.update({
    "personal_frequent_directory": 0.7,
    "recent_access": 0.85,
    "trusted_user_process": 0.8,
    "frequent_operation": 0.75
})

THRESHOLDS["default_adaptive"] = 8.0 # デフォルト適応閾値

# 重要度順序マッピング（評価やクラスタ分布表示用）
SEVERITY_ORDER = {
    "normal": 0,
    "investigating": 1,
    "suspicious": 2,
    "critical": 3
}

CRITICAL_STATUSES = ["suspicious", "critical"] # クリティカル扱いとなるステータス群

# 安全とみなすプロセス名（ホワイトリスト）
SAFE_PROCESSES = [
    "excel.exe", "winword.exe", "word.exe", "outlook.exe",
    "chrome.exe", "firefox.exe", "docker.exe", "Adobe.exe", "git.exe"
]

# 初期学習の閾値
INITIAL_LEARNING_THRESHOLD = 100
INITIAL_CLUSTER_COUNT = 10

global_chain_tune_config = ChainTuneConfig()
CONFIG = ChainTuneConfig()
THRESHOLD_SUSPICIOUS_USER = CONFIG.threshold_suspicious_user
THRESHOLD_INVESTIGATING_USER = CONFIG.threshold_investigating_user
TRUST_ERROR_THRESHOLD_USER = CONFIG.trust_error_threshold_user

THRESHOLD_SUSPICIOUS_ORG = CONFIG.threshold_suspicious_org
THRESHOLD_INVESTIGATING_ORG = CONFIG.threshold_investigating_org
TRUST_ERROR_THRESHOLD_ORG = CONFIG.trust_error_threshold_org

TRUST_SCORE_INVESTIGATING_1 = CONFIG.trust_score_investigating_1
TRUST_SCORE_INVESTIGATING_2 = CONFIG.trust_score_investigating_2
NORMAL_DIV_INVESTIGATING_1 = CONFIG.normal_div_investigating_1
NORMAL_DIV_INVESTIGATING_2 = CONFIG.normal_div_investigating_2
THRESHOLD_DIV_SUSPICIOUS = CONFIG.threshold_div_suspicious
THRESHOLD_DIV_CRITTCAL = CONFIG.threshold_div_crittcal

# -------------------------------
# 🧬ユーティリティ関数
# -------------------------------
HIGH_RISK_PATHS = [p for (p, _) in PATH_SCORES["high"]]
SENSITIVE_PATHS = [p for (p, _) in PATH_SCORES["sensitive"]]

def safe_values(v):
    """値をNumPy配列に変換（JIT対応）"""
    if isinstance(v, dict):
        return np.array(list(v.values()), dtype=np.float32)
    elif isinstance(v, np.ndarray):
        return v.astype(np.float32)
    else:
        return np.array(list(v), dtype=np.float32)

def vectorize_state(state_params):
    # 必要な意味特徴量キーのリスト
    FEATURE_KEYS = [
        "severity_level",
        "action_magnitude",
        "threat_context",
        "trust_score",
        "security_mode",
        "event_score",
        "is_confidential_access",
        "is_cross_dept_access",
        "is_external_network",
        "permission_level_diff",
        "operation_success_flag"
    ]
    vec = []
    for k in FEATURE_KEYS:
        v = state_params.get(k, 0)
        try:
            vec.append(float(v))
        except Exception:
            vec.append(0.0)
    return np.array(vec, dtype=float)

def is_global_ip(ip: str) -> bool:
    if not ip:
        return False  # Noneや空文字はローカル扱いで安全
    return not ip.startswith(PRIVATE_IP_PREFIXES)

def get_filetype(file_path: str) -> str:
    """ファイルパスからファイルタイプを判定"""
    ext = os.path.splitext(file_path)[1].lower()
    for file_type, extensions in FILE_TYPE_MAPPING.items():
        if ext in extensions:
            return file_type
    return "other"

def get_hour_from_timestamp(timestamp: str) -> Optional[int]:
    try:
        # "2025-03-14 08:15:00" → "08"
        return int(timestamp.split()[1].split(":")[0])
    except Exception:
        return None

def is_business_hours(hour: Optional[int]) -> bool:
    """業務時間内かどうかを判定"""
    if hour is None:
        return True  # 時間不明の場合は業務時間とみなす
    return BUSINESS_HOURS["start"] <= hour <= BUSINESS_HOURS["end"]

def assess_path_risk(file_path):
    """パスのリスクレベルを判定"""
    if not file_path:  # Noneや""でも問題なく
        return "normal", 0
    path_lower = file_path.lower()
    if any(p in path_lower for p in HIGH_RISK_PATHS):
        return "high", 50
    elif any(p in path_lower for p in SENSITIVE_PATHS):
        return "sensitive", 35
    return "normal", 0

def create_state_dict(
    severity: int,
    action_mag: float,
    threat: str,
    trust: float,
    mode: str,
    score: float,
    extra: dict = None
) -> dict:
    """状態辞書を作成する共通関数"""
    state = {
        "severity_level": severity,
        "action_magnitude": action_mag,
        "threat_context": threat,
        "trust_score": trust,
        "security_mode": mode,
        "event_score": score,
    }
    if extra:
        state.update(extra)
    return state
# -----------------
# 🔐 Security Log Data Handler
# -----------------
class SecurityLogData:
    """
    強化版 SecurityLogData: 組織構造・振る舞い・許容リソースを疑似的に管理
    """
    def __init__(self):
        # ユーザーごとの「所属・許容IPレンジ」定義（appsは省略）
        self.user_profiles = {
            "yamada_t":    {"dept": "sales",       "ips": ["192.168.1.100", "192.168.1.101"]},
            "suzuki_m":    {"dept": "sales",       "ips": ["192.168.1.120"]},
            "tanaka_s":    {"dept": "sales",       "ips": ["192.168.1.105"]},
            "sato_k":      {"dept": "engineering", "ips": ["192.168.2.50"]},
            "kato_m":      {"dept": "hr",          "ips": ["192.168.4.20"]},
            "ito_h":       {"dept": "finance",     "ips": ["192.168.3.30"]},
            "nakamura_r":  {"dept": "finance",     "ips": ["192.168.3.35"]},
            "honda_m":     {"dept": "engineering", "ips": ["192.168.2.65"]},
            "kimura_t":    {"dept": "engineering", "ips": ["192.168.2.55"]},
            "ito_a":       {"dept": "sales",       "ips": ["192.168.1.110"]},
            "fujita_k":    {"dept": "finance",     "ips": ["192.168.3.45"]},
            "kobayashi_t": {"dept": "finance",     "ips": ["192.168.3.40"]},
            "ogawa_s":     {"dept": "finance",     "ips": ["192.168.3.25"]},
            "ishida_j":    {"dept": "hr",          "ips": ["192.168.4.22"]},
            "hayashi_r":   {"dept": "hr",          "ips": ["192.168.4.35"]},
            "morita_n":    {"dept": "hr",          "ips": ["192.168.4.30"]},
            # 必要なユーザーをここに追加...
        }
        self.dept_ip_ranges = {
            "sales":       "192.168.1.",
            "engineering": "192.168.2.",
            "finance":     "192.168.3.",
            "hr":          "192.168.4.",
        }
        self.allowed_cross_dept = {
            "sales":       ["sales"],
            "engineering": ["engineering", "sales"],
            "finance":     ["finance"],
            "hr":          ["hr"],
            # 必要に応じて追加
        }
        self.safe_processes = [
            "excel.exe", "winword.exe", "outlook.exe", "chrome.exe", "firefox.exe", "docker.exe"
        ]
        self.dangerous_processes = [
            "powershell.exe", "cmd.exe", "wmic.exe", "unknown.exe"
        ]

    def get_user_common_locations(self, user_id):
        return self.user_profiles.get(user_id, {}).get("ips", [])

    def get_user_department(self, user_id):
        return self.user_profiles.get(user_id, {}).get("dept", "unknown")

    def get_dept_ip_range(self, dept):
        return self.dept_ip_ranges.get(dept, "")

    def is_ip_allowed_for_user(self, user_id, ip):
        dept = self.get_user_department(user_id)
        # 部署IPレンジ＋個別登録IPを許容
        return (ip.startswith(self.get_dept_ip_range(dept)) or ip in self.get_user_common_locations(user_id))

    def is_cross_dept_access_allowed(self, from_dept, to_dept):
        # 部署間アクセスの可否
        return to_dept in self.allowed_cross_dept.get(from_dept, [])

    def get_user_common_applications(self, user_id):
        return self.user_profiles.get(user_id, {}).get("apps", [])

    def is_process_safe(self, process_name):
        proc = process_name.lower()
        return any(safe in proc for safe in self.safe_processes)

    def is_process_dangerous(self, process_name):
        proc = process_name.lower()
        return any(danger in proc for danger in self.dangerous_processes)

    # ↓以下はダミーのまま
    def get_last_access_timestamp(self, user_id, file_path):
        hours_ago = random.randint(1, 720)
        return datetime.now() - timedelta(hours=hours_ago)

    def get_file_access_count(self, user_id, file_path, days=30):
        return random.randint(0, 100)

    def get_login_failure_rate(self, user_id, days=7):
        return random.uniform(0.0, 0.3)

    def get_user_active_hours(self, user_id):
        return {
            "weekday_hours": list(range(8, 19)),
            "weekend_hours": []
        }

    def user_has_permission(self, user_id, permission_type):
        executive_users = ["executive", "ceo", "cfo"]
        if user_id.lower() in executive_users:
            return True
        return False

    def get_directory_access_stats(self, directory_path):
        return {
            "total_users": random.randint(5, 50),
            "avg_daily_access": random.randint(10, 200),
            "common_operations": ["FileRead", "FileWrite"]
        }

# -------------------------------
# 🧬 クラスタリング関数
# -------------------------------
def fit_normal_clusters(state_vectors, n_clusters=5, save_examples=True):

    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(state_vectors)
    cluster_info = {}

    for i in range(n_clusters):
        cluster_points = state_vectors[km.labels_ == i]
        center = km.cluster_centers_[i]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        radius = np.percentile(distances, 95)
        # ★分散も計算
        cov = np.cov(cluster_points, rowvar=False)
        # ★代表パターン
        idx_nearest = np.argmin(distances)
        typical = cluster_points[idx_nearest] if save_examples else None

        cluster_info[i] = {
            "center": center,
            "radius": radius,
            "cov": cov,
            "size": len(cluster_points),
            "typical": typical,
        }
    return {"model": km, "info": cluster_info}

def compute_information_tensor(state_vector_list, epsilon=1e-6, max_norm=100):
    """情報テンソルを計算（JIT非対応:NumPy）"""
    matrix = np.array([safe_values(v) for v in state_vector_list])
    variances = np.var(matrix, axis=0)
    info_tensor = 1.0 / (variances + epsilon)
    return info_tensor * min(1.0, max_norm / np.max(info_tensor))

# ===== ダイバージェンス計算関数 =====
def get_adaptive_weight(diff, key, weights_l1, weights_l2):
    if diff < 0.5:
        return weights_l1.get(key, 1.0)
    elif diff < 2.0:
        t = (diff - 0.5) / 1.5
        return (1 - t) * weights_l1.get(key, 1.0) + t * weights_l2.get(key, 1.0)
    else:
        return weights_l2.get(key, 1.0)

def adaptive_metric(diff):
    """適応的メトリック"""
    if diff < 0.5:
        return diff
    elif diff < 2.0:
        return diff ** 2
    elif diff < 6.0:
        return 4 + (diff - 2) ** 2.2
    else:
        return 30 * np.log(diff - 4 + 2)

def compute_security_divergence(
    prev_state, curr_state,
    weights_l1=None, weights_l2=None,
    metric="weighted_hybrid",
    info_tensor=None,
    hybrid_ratio=0.7,
    config=None
):
    divergence = 0.0
    all_keys = set(prev_state.keys()) | set(curr_state.keys())

    # ★ config優先で重みセット
    if config is not None:
        weights_l1 = config.weights_l1
        weights_l2 = config.weights_l2

    for key in all_keys:
        v1, v2 = prev_state.get(key, 0), curr_state.get(key, 0)
        diff = np.linalg.norm(np.array(v1) - np.array(v2)) if isinstance(v1, np.ndarray) else abs(v1 - v2)
        w = get_adaptive_weight(diff, key, weights_l1, weights_l2)

        if metric == "l1":
            delta = diff
        elif metric == "l2":
            delta = diff ** 2
        elif metric == "cosine":
            delta = 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        elif metric == "zscore" and info_tensor is not None:
            mean = info_tensor["mean"].get(key, 0)
            std = info_tensor["std"].get(key, 1e-6)
            z = (v2 - mean) / std
            delta = np.abs(z) * w
        elif metric == "hybrid_z":
            mean = info_tensor["mean"].get(key, 0)
            std = info_tensor["std"].get(key, 1e-6)
            z = (v2 - mean) / std
            delta = hybrid_ratio * (w * adaptive_metric(diff)) + (1-hybrid_ratio) * (np.abs(z) * w)
        else:
            delta = w * adaptive_metric(diff)
        divergence += delta

    return divergence

# -------------------------------
# 🧬 イベント評価関数
# -------------------------------
def evaluate_security_event_by_cluster(
    new_state, model, info_tensor, weights_l1=None, weights_l2=None,
    metric="weighted_hybrid", config=None
):
    """クラスタによるセキュリティイベント評価"""
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    # ★ config優先で重みを取得
    if config is not None:
        if weights_l1 is None:
            weights_l1 = config.weights_l1.copy()
        if weights_l2 is None:
            weights_l2 = config.weights_l2.copy()

    state_dict = new_state if isinstance(new_state, dict) else {f"dim_{j}": v for j, v in enumerate(new_state)}

    min_divergence, best_cluster = float('inf'), -1
    for i, center in enumerate(model.cluster_centers_):
        center_dict = {k: v for k, v in zip(state_dict.keys(), center)}
        divergence = compute_security_divergence(
            center_dict, state_dict,
            weights_l1, weights_l2, metric,
            config=config
        )

        if divergence < min_divergence:
            min_divergence, best_cluster = divergence, i

    divergence_max = config.divergence_max if config and hasattr(config, 'divergence_max') else 100.0
    normalized_divergence = min(min_divergence, divergence_max)
    return normalized_divergence, best_cluster


def classify_event_by_multi_clusters(
    new_state,
    normal_model, exceptional_model, anomaly_model,
    info_tensor,
    config=None,
    weights_l1=None, weights_l2=None
):
    if config is not None:
        if weights_l1 is None:
            weights_l1 = config.weights_l1.copy()
        if weights_l2 is None:
            weights_l2 = config.weights_l2.copy()

    norm_div, norm_cl = evaluate_security_event_by_cluster(
        new_state, normal_model, info_tensor, weights_l1, weights_l2, config=config
    )

    if exceptional_model is not None:
        exc_div, exc_cl = evaluate_security_event_by_cluster(
            new_state, exceptional_model, info_tensor, weights_l1, weights_l2, config=config
        )
    else:
        exc_div, exc_cl = float('inf'), -1

    if anomaly_model is not None:
        ano_div, ano_cl = evaluate_security_event_by_cluster(
            new_state, anomaly_model, info_tensor, weights_l1, weights_l2, config=config
        )
    else:
        ano_div, ano_cl = float('inf'), -1

    min_div = min(norm_div, exc_div, ano_div)
    if min_div == ano_div and anomaly_model is not None:
        verdict = "anomaly"
        cluster = ano_cl
    elif min_div == exc_div and exceptional_model is not None:
        verdict = "exceptional"
        cluster = exc_cl
    else:
        verdict = "normal"
        cluster = norm_cl

    print(f"[XAI] verdict={verdict}, normal_div={norm_div:.2f}, exc_div={exc_div:.2f}, ano_div={ano_div:.2f}")

    return {
        "verdict": verdict,
        "cluster": cluster,
        "divergences": {
            "normal": norm_div,
            "exceptional": exc_div,
            "anomaly": ano_div
        },
        "best_divergence": min_div
    }

# -------------------------------
# 🧬例外の追加セット
# -------------------------------
def random_private_ip():
    """PRIVATE_IP_PREFIXESから社内向けIPをランダム生成"""
    prefix = random.choice(PRIVATE_IP_PREFIXES)
    return f"{prefix}{random.randint(1, 4)}.{random.randint(1, 254)}"

def create_realistic_security_event(
    user_id, department, operation, timestamp,
    file_path=None, file_size_kb=None,
    process_name=None, destination_ip=None,
    status="SUCCESS", business_context=None
):
    """
    セキュリティイベントのリアルな辞書データを生成
    Args:
        user_id: ユーザーID
        department: 部署名
        operation: 操作種別（例: Login, FileRead, ...）
        timestamp: datetime型（自動でフォーマット）
        ...（略）
    Returns:
        event: dict
    """
    event = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "department": department,
        "operation": operation,
        "status": status,
        "source_ip": random_private_ip()
    }
    if file_path is not None:
        event["file_path"] = file_path
    if file_size_kb is not None:
        event["file_size_kb"] = file_size_kb
    if process_name is not None:
        event["process_name"] = process_name
    if destination_ip is not None:
        event["destination_ip"] = destination_ip
    if business_context is not None:
        event["_business_context"] = business_context
    return event

def generate_morning_patterns(dept_name):
    """指定部署の朝の正常な業務パターンを生成"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns
    base_date = datetime(2025, 3, 10, 0, 0, 0)
    for user in dept_info["users"][:3]:
        login_time = base_date.replace(hour=random.randint(8, 9), minute=random.randint(0, 59))
        patterns.append(create_realistic_security_event(
            user, dept_name, "Login", login_time
        ))
        mail_time = login_time + timedelta(minutes=random.randint(5, 15))
        patterns.append(create_realistic_security_event(
            user, dept_name, "ProcessCreate", mail_time,
            process_name="outlook.exe"
        ))
    return patterns

def generate_noon_patterns(dept_name):
    """指定部署の昼の正常な業務パターンを生成"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns
    base_date = datetime(2025, 3, 10, 0, 0, 0)
    for user in dept_info["users"][:2]:
        logout_time = base_date.replace(hour=12, minute=random.randint(0, 30))
        patterns.append(create_realistic_security_event(
            user, dept_name, "Logout", logout_time
        ))
        login_time = logout_time + timedelta(hours=1)
        patterns.append(create_realistic_security_event(
            user, dept_name, "Login", login_time
        ))
    return patterns


def generate_evening_patterns(dept_name):
    """夕方の正常なファイル保存＋ログアウトパターンを生成"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns

    base_date = datetime(2025, 3, 10, 0, 0, 0)
    for user in dept_info["users"][:3]:
        # ファイルパス・プロセス名分岐（業務ルールに応じて拡張しやすい設計！）
        save_time = base_date.replace(hour=random.randint(17, 18), minute=random.randint(0, 59))
        file_path_map = {
            "sales": f"\\\\fileserver\\sales\\reports\\daily_report_{user}.xlsx",
            "engineering": f"\\\\fileserver\\engineering\\docs\\progress_{user}.md",
            "finance": f"\\\\fileserver\\finance\\reports\\summary_{user}.xlsx",
        }
        file_path = file_path_map.get(dept_name, f"\\\\fileserver\\{dept_name}\\docs\\report_{user}.docx")
        proc_name = "EXCEL.EXE" if dept_name in ["sales", "finance"] else "WINWORD.EXE"
        patterns.append(create_realistic_security_event(
            user, dept_name, "FileWrite", save_time,
            file_path=file_path,
            file_size_kb=random.randint(50, 500),
            process_name=proc_name
        ))
        logout_time = save_time + timedelta(minutes=random.randint(5, 30))
        patterns.append(create_realistic_security_event(
            user, dept_name, "Logout", logout_time
        ))
    return patterns

def generate_year_end_patterns(dept_name):
    """年度末の繁忙＆部門横断・特権操作含むリアル業務パターンを生成"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns

    base_date = datetime(2025, 3, 29, 0, 0, 0)  # 年度末の土曜日

    # 週末出社ログイン
    for user in dept_info["users"]:
        login_time = base_date.replace(hour=random.randint(9, 11), minute=random.randint(0, 59))
        patterns.append(create_realistic_security_event(
            user, dept_name, "Login", login_time,
            business_context=["year_end", "weekend_work"]
        ))

    # 部署ごとの特殊な協力パターン
    if dept_name == "finance":
        # 経理は営業データを確認
        for user in dept_info["users"][:2]:  # 代表2名
            event_time = datetime(2025, 3, 30, random.randint(14, 16), random.randint(0, 59), 0)

            # 営業の売上レポートを読む
            patterns.append(create_realistic_security_event(
                user, dept_name, "FileRead", event_time,
                file_path="\\\\fileserver\\sales\\reports\\2025\\03\\monthly_sales_202503.xlsx",
                file_size_kb=random.randint(2000, 5000),
                process_name="EXCEL.EXE",
                business_context=["year_end", "cross_dept_allowed"]
            ))

            # 営業の契約システムにアクセス
            event_time = event_time.replace(hour=random.randint(16, 18))
            patterns.append(create_realistic_security_event(
                user, dept_name, "FileRead", event_time,
                file_path="\\\\fileserver\\sales\\contracts\\2025\\Q4\\contract_summary.xlsx",
                file_size_kb=random.randint(1000, 3000),
                process_name="EXCEL.EXE",
                business_context=["year_end", "cross_dept_allowed"]
            ))

    elif dept_name == "sales":
        # 営業は経理システムに売上データを入力
        for user in dept_info["users"][:2]:
            event_time = datetime(2025, 3, 31, random.randint(10, 12), random.randint(0, 59), 0)

            # 経理の年度末レポートに書き込み
            patterns.append(create_realistic_security_event(
                user, dept_name, "FileWrite", event_time,
                file_path="\\\\fileserver\\finance\\reports\\2025\\03\\年度末決算_売上入力.xlsx",
                file_size_kb=random.randint(3000, 8000),
                process_name="EXCEL.EXE",
                business_context=["year_end", "cross_dept_allowed", "data_submission"]
            ))

            # 財務システムへのネットワーク接続
            patterns.append(create_realistic_security_event(
                user, dept_name, "NetworkConnect", event_time,
                destination_ip="192.168.3.100",  # 財務サーバー
                process_name="FinancePortal.exe",
                business_context=["year_end", "cross_dept_allowed"]
            ))

    elif dept_name == "hr":
        # HRは全部署の残業データを収集
        for user in dept_info["users"][:1]:  # 代表1名
            base_time = datetime(2025, 3, 28, random.randint(17, 19), random.randint(0, 59), 0)

            # 各部署の勤怠データを読み込み
            dept_paths = {
                "sales": "\\\\fileserver\\sales\\attendance\\2025\\03\\overtime_report.csv",
                "engineering": "\\\\fileserver\\engineering\\timesheet\\2025\\03\\dev_hours.xlsx",
                "finance": "\\\\fileserver\\finance\\attendance\\2025\\03\\working_hours.xlsx"
            }

            for dept, path in dept_paths.items():
                patterns.append(create_realistic_security_event(
                    user, dept_name, "FileRead", base_time,
                    file_path=path,
                    file_size_kb=random.randint(100, 500),
                    process_name="EXCEL.EXE",
                    business_context=["year_end", "cross_dept_allowed", "hr_audit"]
                ))
                base_time = base_time + timedelta(minutes=random.randint(5, 15))

    elif dept_name == "engineering":
        # エンジニアは年度末のシステム監査対応
        for user in dept_info["users"][:1]:
            event_time = datetime(2025, 3, 27, random.randint(20, 22), random.randint(0, 59), 0)

            # 監査ログへの特権アクセス
            patterns.append(create_realistic_security_event(
                user, dept_name, "FileRead", event_time,
                file_path="\\\\auditserver\\logs\\system\\2025\\security_audit_log.txt",
                file_size_kb=random.randint(10000, 50000),
                process_name="notepad++.exe",
                business_context=["year_end", "security_audit", "elevated_access"]
            ))

            # 監査レポートの作成
            patterns.append(create_realistic_security_event(
                user, dept_name, "FileWrite", event_time + timedelta(minutes=30),
                file_path="\\\\fileserver\\engineering\\audit\\2025\\annual_security_report.docx",
                file_size_kb=random.randint(500, 2000),
                process_name="WINWORD.EXE",
                business_context=["year_end", "security_audit"]
            ))

            # システム管理ツールの起動
            patterns.append(create_realistic_security_event(
                user, dept_name, "ProcessCreate", event_time,
                process_name="mmc.exe",  # Microsoft Management Console
                business_context=["year_end", "security_audit", "elevated_access"]
            ))

    return patterns

# -------------------------------
# 🧬 リスク判定強化
# -------------------------------
# ===== メイン判定関数 =====
def unified_security_judge(
    event: Dict[str, Any],
    user_history: Dict[str, Any],
    chain_context: Optional[Dict[str, Any]] = None,
    normal_div: Optional[float] = None,
    trust_score: Optional[float] = None
) -> Tuple[str, str]:

    # イベント情報の抽出
    op = event.get("operation", "")
    status = event.get("status", "")
    file_path = event.get("file_path", "")
    dept = event.get("department", "")
    process = event.get("process_name", "").lower()
    source_ip = event.get("source_ip", "")
    dest_ip = event.get("destination_ip", "")
    file_size_kb = event.get("file_size_kb", 0)
    hour = get_hour_from_timestamp(event.get("timestamp", ""))

    # ===== 基本判定ロジック =====
    verdict, reason = _basic_security_rules(
        op, status, file_path, dept, process, source_ip, dest_ip,
        file_size_kb, hour, user_history
    )

    # chain_contextがない場合は基本判定のみ返す
    if chain_context is None:
        return verdict, reason

    # ===== 拡張判定ロジック（パターン分析） =====
    return _enhanced_security_analysis(
        verdict, reason, event, op, dept, hour,
        chain_context, normal_div, trust_score
    )

def _basic_security_rules(
    op: str, status: str, file_path: str, dept: str,
    process: str, source_ip: str, dest_ip: str,
    file_size_kb: int, hour: Optional[int],
    user_history: Dict[str, Any]
) -> Tuple[str, str]:
    """基本的なセキュリティルールによる判定"""

    # 1. 部署とパスの制限チェック
    if op in ("FileRead", "FileWrite") and dept in DEPT_PATH_RULES:
        rules = DEPT_PATH_RULES[dept]
        if any(restricted in file_path for restricted in rules["restricted"]):
            return "suspicious", rules["message"]

    # 2. 機密フォルダ操作
    if op in ("FileRead", "FileCopy") and "\\confidential\\" in file_path.lower():
        if dept not in CONFIDENTIAL_ACCESS_DEPTS:
            return "suspicious", "非権限者による機密ファイル操作"

    # 3. 業務時間外のログイン/ログアウト
    if op in ("Login", "Logout") and hour is not None:
        if hour < BUSINESS_HOURS["start"] or hour > BUSINESS_HOURS["end"] + 1:
            if op == "Login":
                return "investigating", "深夜Login"
            if op == "Logout" and status == "SUCCESS":
                return "normal", "深夜Logout特例"

    # 4. 失敗ステータスのチェック
    if status == "FAILED":
        failed_attempts = user_history.get("failed_attempts", 0)
        if failed_attempts >= THRESHOLDS["failed_attempts_critical"]:
            return "suspicious", "短期間で失敗多発"
        else:
            return "investigating", "操作失敗"

    # 5. 大容量ファイルまたは外部送信
    if file_size_kb > THRESHOLDS["large_file_kb"]:
        return "critical", "巨大ファイル操作"

    if op == "NetworkConnect" and dest_ip and is_global_ip(dest_ip):
        return "critical", "外部ネットワークへの接続"

    # 6. 危険なプロセス
    DANGEROUS_PROCESSES = [name for name, _ in PROCESS_SCORES["dangerous"]]
    if any(d in process for d in DANGEROUS_PROCESSES):
        return "investigating", "危険プロセス利用"

    # 7. 異常なアクセス頻度
    if user_history.get("access_count", 0) > THRESHOLDS["abnormal_access_count"]:
        return "latent_suspicious", "異常なアクセス頻度"

    # 8. 管理領域への書き込み
    admin_paths = ["\\admin\\", "\\config\\"]
    if op == "FileWrite" and any(path in file_path.lower() for path in admin_paths):
        return "critical", "管理・設定ファイルの書き込み"

    # 9. グローバルIPからのアクセス
    if op in ("Login", "NetworkConnect"):
        if source_ip and is_global_ip(source_ip):
            return "critical", "海外IPからの操作"
        if dest_ip and is_global_ip(dest_ip):
            return "critical", "海外宛て接続"

    return "normal", "通常"


def _enhanced_security_analysis(
    verdict: str, reason: str, event: Dict[str, Any],
    op: str, dept: str, hour: Optional[int],
    chain_context: Dict[str, Any],
    normal_div: Optional[float],
    trust_score: Optional[float]
) -> Tuple[str, str]:
    """パターン分析による拡張セキュリティ判定"""

    # パターンキーの生成と履歴更新
    pattern_key = f"{op}_{get_filetype(event.get('file_path', ''))}_{dept}"
    _update_pattern_history(chain_context, pattern_key)

    recent_patterns = sum(1 for v in chain_context['pattern_history'].values() if v > 1)

    # 正常判定の緩和条件チェック
    if verdict == "normal":
        if _should_relax_normal_verdict(
            recent_patterns, chain_context, normal_div, trust_score, op, hour
        ):
            return "normal", "正常（業務div＋頻度＋業務時間）"

    # investigating/latent_suspiciousの緩和
    if verdict == "investigating":
        if _should_relax_investigating(
            op, trust_score, normal_div, chain_context, hour
        ):
            return "normal", "初回/業務divヒット(調整)"

    if verdict == "latent_suspicious":
        if _should_relax_latent_suspicious(
            op, trust_score, normal_div, chain_context
        ):
            return "normal", "初回/業務divヒット(調整)"

    # 特定の理由に対する累積チェック
    if reason in ("非権限者による機密ファイル操作", "営業が経理/人事リソースにアクセス"):
        count = chain_context['pattern_history'].get(pattern_key, 0)
        if (count < THRESHOLDS["pattern_accumulation_limit"] and
            trust_score is not None and trust_score > THRESHOLDS["trust_score_medium"] and
            normal_div is not None and normal_div > THRESHOLDS["normal_div_threshold"]):
            return "investigating", f"{reason}（緩和:信頼スコア/初回系）"
        return "suspicious", reason

    # critical/suspiciousの一段階緩和
    if verdict in ("suspicious", "critical"):
        if (trust_score and trust_score > THRESHOLDS["trust_score_low"] and
            normal_div and normal_div > THRESHOLDS["normal_div_high"]):
            return "investigating", "高信頼div: 一段階緩和"

    return verdict, reason

def _update_pattern_history(chain_context: Dict[str, Any], pattern_key: str) -> None:
    # パターン履歴の初期化安全対応
    chain_context.setdefault('pattern_history', {})
    chain_context['pattern_history'][pattern_key] = \
        chain_context['pattern_history'].get(pattern_key, 0) + 1

    # 連続カウントの更新
    last_pattern = chain_context.get("last_pattern")
    if last_pattern == pattern_key:
        chain_context['repeat_count'] = chain_context.get('repeat_count', 1) + 1
    else:
        chain_context['repeat_count'] = 1
    chain_context['last_pattern'] = pattern_key

def _should_relax_normal_verdict(
    recent_patterns: int, chain_context: Dict[str, Any],
    normal_div: Optional[float], trust_score: Optional[float],
    op: str, hour: Optional[int]
) -> bool:
    """正常判定を緩和すべきかどうかを判定"""
    pattern_condition = (
        recent_patterns > THRESHOLDS["pattern_count_threshold"] or
        chain_context.get('anomaly_score', 0.0) > THRESHOLDS["anomaly_score_threshold"]
    )
    repeat_condition = chain_context['repeat_count'] > THRESHOLDS["repeat_count_threshold"]

    if pattern_condition or repeat_condition:
        return (
            normal_div is not None and normal_div < THRESHOLDS["normal_div_threshold"] and
            trust_score is not None and trust_score > THRESHOLDS["trust_score_threshold"] and
            op in BUSINESS_OPERATIONS and
            is_business_hours(hour)
        )
    return False

def _should_relax_investigating(
    op: str, trust_score: Optional[float], normal_div: Optional[float],
    chain_context: Dict[str, Any], hour: Optional[int]
) -> bool:
    """investigating判定を緩和すべきかどうかを判定"""
    if op not in BUSINESS_OPERATIONS:
        return False

    # 初回アクセスの場合
    if chain_context.get('repeat_count', 1) <= 2:
        return (
            trust_score is not None and trust_score > THRESHOLDS["trust_score_threshold"] and
            normal_div is not None and normal_div < THRESHOLDS["normal_div_threshold"]
        )

    # 操作失敗の場合
    if chain_context.get('repeat_count', 1) < 3:
        return (
            normal_div is not None and normal_div < THRESHOLDS["normal_div_high"] and
            trust_score is not None and trust_score > THRESHOLDS["trust_score_medium"] and
            is_business_hours(hour)
        )

    return False

def _should_relax_latent_suspicious(
    op: str, trust_score: Optional[float], normal_div: Optional[float],
    chain_context: Dict[str, Any]
) -> bool:
    """latent_suspicious判定を緩和すべきかどうかを判定"""
    return (
        op in BUSINESS_OPERATIONS and
        trust_score is not None and trust_score > THRESHOLDS["trust_score_high"] and
        normal_div is not None and normal_div < THRESHOLDS["normal_div_threshold"] and
        chain_context.get('repeat_count', 1) <= 2
    )

def is_internal_destination(dest):
    # IP（文字列）→プレフィックス一致
    if any(dest.startswith(prefix) for prefix in TRUSTED_IPS):
        return True
    # ホワイトリストFQDN一致
    if dest in TRUSTED_HOSTNAMES:
        return True
    return False
# -------------------------------
# 🧬 モード分類関数
# -------------------------------
def classify_security_mode_auto(
    divergence,
    state_params,
    previous_ema=None,
    config=None,
    threshold_override=None,
    operation=None,
    is_user_chain=False  # 🆕 ここで追加！
):
    if config is None:
        raise ValueError("configは必須です！")

    severity_level = state_params.get("severity_level", 0.0)
    action_vector = state_params.get("action_vector", np.array([1.0, 0.0, 0.0]))
    action_magnitude = np.linalg.norm(action_vector)
    trust_score = state_params.get("trust_score", 1.0)
    operation = operation or state_params.get("operation", "")
    timestamp = state_params.get("timestamp")

    # 🔄 通常のスコア算出
    weighted_score = (
        config["alpha"] * divergence +
        config["beta"] * severity_level +
        config["gamma"] * action_magnitude
    )
    if previous_ema is not None:
        weighted_score = (
            config["ema_alpha"] * weighted_score +
            (1 - config["ema_alpha"]) * previous_ema
        )

    threshold_suspicious = threshold_override or config["threshold_suspicious"]
    threshold_investigating = threshold_suspicious * 0.5

    # 🧭 意味密度スコアに基づく判定
    if is_user_chain:
        threshold_suspicious = THRESHOLD_SUSPICIOUS_USER
        threshold_investigating = THRESHOLD_INVESTIGATING_USER
        trust_error_threshold = TRUST_ERROR_THRESHOLD_USER
    else:
        threshold_suspicious = THRESHOLD_SUSPICIOUS_ORG
        threshold_investigating = THRESHOLD_INVESTIGATING_ORG
        trust_error_threshold = TRUST_ERROR_THRESHOLD_ORG

    # このあとのブロックでは、"統一された名前"を使う
    if trust_score < trust_error_threshold:
        mode = "critical"
    elif divergence > THRESHOLD_DIV_CRITTCAL:
        mode = "critical"
    elif weighted_score > threshold_suspicious:
        mode = "suspicious"
    elif divergence > THRESHOLD_DIV_SUSPICIOUS:
        mode = "suspicious"
    elif weighted_score > threshold_investigating:
        mode = "investigating"
    elif trust_score < TRUST_SCORE_INVESTIGATING_1 and divergence > NORMAL_DIV_INVESTIGATING_1:
        mode = "investigating"
    elif trust_score < TRUST_SCORE_INVESTIGATING_2 and divergence > NORMAL_DIV_INVESTIGATING_2:
        mode = "investigating"
    else:
        mode = "normal"

    return mode, weighted_score

def calculate_event_score_from_real_log(event, security_log):
    """
    LanScope Catログから取得可能なデータのみでスコア計算
    """
    # --- 基本セットアップ ---
    operation = event.get("operation", event.get("event_type", ""))
    file_path = event.get("file_path", "")
    process = event.get("process_name", "").lower()
    destination = event.get("destination_ip", "")
    status = event.get("status", "SUCCESS")
    dept = event.get("department", "")
    file_size_kb = event.get("file_size_kb", 0)
    file_size_mb = file_size_kb / 1024
    timestamp = event.get("timestamp", event.get("event_time", ""))
    user_id = event.get("user_id", "unknown")
    source_ip = event.get("source_ip", "")
    score = OPERATION_SCORES.get(operation, 10)
    trust_score = 1.0

    # --- 1. Login時のIP・ユーザー認証 ---
    if operation == "Login":
        if source_ip and not source_ip.startswith(PRIVATE_IP_PREFIXES):
            score += EXTERNAL_IP_SCORES["external"]
        if user_id == "unknown":
            score += 40
        try:
            approved = security_log.get_user_approved_devices(user_id)
            if event.get("workstation_name") in approved:
                score = max(score - 10, 0)
        except Exception:
            pass

    # --- 2. 時間帯ペナルティ ---
    hour = None
    if timestamp:
        try:
            hour = int(timestamp.split()[1].split(":")[0])
        except Exception:
            pass
    if hour is not None:
        for (start, end), add in TIME_PENALTY_SCORES:
            if start <= hour < end:
                score += add

    # --- 3. Logout特例 ---
    if operation == "Logout":
        score = 0
        if hour in [12, 17, 18, 19, 20, 21, 22]:
            score = 0
        if dept == "executive":
            score = 0

    # --- 4. 通常外IPからのアクセス ---
    if source_ip and user_id != "unknown":
        try:
            typical_ips = security_log.get_user_common_locations(user_id)
            if typical_ips and source_ip not in typical_ips:
                if source_ip.startswith(PRIVATE_IP_PREFIXES):
                    score += EXTERNAL_IP_SCORES["internal"]
                else:
                    score += EXTERNAL_IP_SCORES["external"]
        except Exception:
            pass

    # --- 5. ファイルパス種別ごとの加点 ---
    def path_score_by_type(path):
        path_lower = path.lower()
        for key in ["high", "sensitive", "normal"]:
            for pattern, add in PATH_SCORES[key]:
                if pattern in path_lower:
                    return add
        return 0

    if file_path:
        score += path_score_by_type(file_path)
        path_lower = file_path.lower()

        # 顧客リスト・大容量ファイル検出
        if operation == "FileRead" and any(
            key in path_lower for key in [
                "customer", "顧客", "full_customer_list", "bonus_plan", "employee_list"
            ]
        ):
            if file_size_mb > 20:
                score += 40
            elif file_size_mb > 5:
                score += 20

        # Confidential判定と容量依存加点
        if "\\confidential\\" in path_lower:
            user_has_confidential_access = security_log.user_has_permission(user_id, "confidential")
            is_executive = dept.lower() == "executive"
            if file_size_mb > 50:
                score += 20 if (user_has_confidential_access or is_executive) else 80
            else:
                score += 5 if (user_has_confidential_access or is_executive) else 40

    # --- 6. ファイルサイズ加点 ---
    for threshold, add in FILE_SIZE_THRESHOLDS:
        if file_size_mb > threshold:
            score += add
            break

    # --- 7. 週末加点 ---
    if timestamp:
        try:
            import datetime as dtmod
            date_str = timestamp.split()[0]
            if dtmod.datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5:
                score += 10  # WEEKEND_BONUSで統一してもOK
        except Exception:
            pass

    # --- 8. プロセス名加点 ---
    def match_process(process, procs):
        return any(tool in process for tool in procs)

    # 危険プロセス
    dangerous_procs = [name for name, _ in PROCESS_SCORES["dangerous"]]
    admin_procs = [name for name, _ in PROCESS_SCORES["admin"]]
    archiver_procs = [name for name, _ in PROCESS_SCORES["archiver"]]

    if match_process(process, dangerous_procs):
        score += 70
    elif match_process(process, admin_procs):
        score += 20
    elif match_process(process, archiver_procs):
        score += 25

    # --- 9. 外部宛ネットワーク ---
    if destination and not destination.startswith(PRIVATE_IP_PREFIXES):
        score += 50
        if destination.startswith(("198.51.", "203.0.113.")):
            score += 30

    # --- 10. 部署×リソース判定 ---
    if destination and not is_internal_destination(destination):
        score += 50
        if destination.startswith(("198.51.", "203.0.113.")):
            score += 30

    # --- 11. ステータス依存加点 ---
    if status == "FAILED":
        if operation == "LoginFailed":
            score += 10
        elif operation == "NetworkConnect":
            score += 25
        else:
            score += 15

    # --- 12. 文書作成の例外減点 ---
    if operation == "FileWrite" and file_path:
        if any(doc in file_path.lower() for doc in [".docx", ".xlsx", ".pptx", "proposals", "proposal_"]):
            score = max(score - 25, 2)

    # --- スコア上限 ---
    return min(score, 100)

def calculate_severity_level(event_score):
    """
    イベントスコア（0-100）から重大度レベル（0.0-1.0）を計算
    """
    for threshold, level in SEVERITY_LEVEL_THRESHOLDS:
        if event_score >= threshold:
            return level
    return SEVERITY_LEVEL_THRESHOLDS[-1][1]  # 最小値で返す

def calculate_action_vector_from_operation(operation):
    """
    操作タイプから行動ベクトルを返す
    """
    return ACTION_OPERATION_VECTORS.get(operation, ACTION_DEFAULT_VECTOR)

def calculate_user_trust_from_history(event, security_log, prev_trust=1.0):
    user_id = event.get("user_id", "unknown")
    file_path = event.get("file_path", "")
    operation = event.get("operation", "")
    status = event.get("status", "SUCCESS")
    base_trust = prev_trust

    # --- 履歴・活動ベースの減点 ---
    try:
        login_failure_rate = security_log.get_login_failure_rate(user_id) or 0.0
        base_trust -= login_failure_rate
    except Exception:
        pass

    # ファイルアクセス履歴（recency, first access）
    if file_path:
        try:
            last_access = security_log.get_last_access_timestamp(user_id, file_path)
            if last_access:
                hours_since_access = (datetime.now() - last_access).total_seconds() / 3600
            else:
                hours_since_access = HOURS_SINCE_ACCESS
            access_count = security_log.get_file_access_count(user_id, file_path) or 0
        except Exception:
            hours_since_access = HOURS_SINCE_ACCESS
            access_count = 0
    else:
        hours_since_access = HOURS_SINCE_ACCESS
        access_count = 0

    is_first_access = access_count == 0
    first_access_penalty = TRUST_SCORES["first_access_penalty"] if is_first_access else 0.0
    recency_penalty = min(hours_since_access / TRUST_SCORES["recency_divisor"], TRUST_SCORES["recency_penalty_cap"])

    # --- 操作ごとの信頼変動 ---
    operation_trust = TRUST_SCORES["operation_penalty"].get(operation, 0.0)

    # 異常・失敗系ペナルティを累積型で追加
    trust_delta = 0.0
    if status == "FAILED":
        if operation in ["Login", "LoginFailed"]:
            trust_delta -= 0.45
        elif operation in ["FileWrite", "FileDelete", "ProcessCreate"]:
            trust_delta -= 0.25
        elif operation == "NetworkConnect":
            trust_delta -= 0.30
        else:
            trust_delta -= 0.15
    if event.get("critical_flag", False):
        trust_delta -= 0.4
    if event.get("external_access_flag", False):
        trust_delta -= 0.25

    # 日常業務は微回復（または0.0）
    if status == "SUCCESS" and operation in ["Login", "FileRead", "FileWrite"]:
        trust_delta += 0.01

    # IP逸脱ペナルティ
    source_ip = event.get("source_ip", "")
    if source_ip and user_id != "unknown":
        try:
            typical_ips = security_log.get_user_common_locations(user_id) or []
            if typical_ips and source_ip not in typical_ips:
                if source_ip.startswith(PRIVATE_IP_PREFIXES):
                    trust_delta -= TRUST_SCORES["internal_ip_penalty"]
                else:
                    trust_delta -= TRUST_SCORES["external_ip_penalty"]
        except Exception:
            pass

    # 時間帯逸脱
    timestamp = event.get("timestamp", "")
    if timestamp:
        try:
            hour = int(timestamp.split()[1].split(":")[0])
            active_hours = security_log.get_user_active_hours(user_id)
            if "weekday_hours" in active_hours and hour not in active_hours["weekday_hours"]:
                trust_delta -= TRUST_SCORES["time_penalty"]
        except Exception:
            pass

    # ビジネス文脈によるボーナス
    contexts = event.get("_business_context", [])
    if "year_end" in contexts or "quarter_end" in contexts:
        trust_delta += TRUST_SCORES["business_context_year_end_bonus"]
    if "cross_dept_allowed" in contexts:
        trust_delta += TRUST_SCORES["business_context_cross_dept_bonus"]

    # --- 総合スコア計算 ---
    trust_score = base_trust * (1.0 - recency_penalty - first_access_penalty) + operation_trust + trust_delta

    # 範囲制限
    trust_score = max(TRUST_SCORES["min_trust"], min(TRUST_SCORES["max_trust"], trust_score))
    return trust_score

def calculate_threat_context_from_log(event, security_log):
    threat_score = 0.0

    # 1. 時間帯の異常
    timestamp = event.get("timestamp", "")
    if timestamp:
        time_anomaly = calculate_time_anomaly(timestamp)
        threat_score += time_anomaly * THREAT_CONTEXT_WEIGHTS["time_anomaly"]

        user_id = event.get("user_id", "unknown")
        hour = int(timestamp.split()[1].split(":")[0])
        try:
            active_hours = security_log.get_user_active_hours(user_id)
            weekday_hours = active_hours.get("weekday_hours", [])
        except Exception:
            weekday_hours = []
        if hour not in weekday_hours:
            threat_score += THREAT_CONTEXT_WEIGHTS["outside_active_hours"]

    # 2. ネットワークの異常（外部通信）
    destination_ip = event.get("destination_ip", "")
    if destination_ip:
        if not destination_ip.startswith(PRIVATE_IP_PREFIXES):
            threat_score += THREAT_CONTEXT_WEIGHTS["external_network"]
        # GATEWAY_IP_SUFFIXESが定義されていない場合は空リストでOK
        if 'GATEWAY_IP_SUFFIXES' in globals():
            if destination_ip.endswith(tuple(GATEWAY_IP_SUFFIXES)):
                threat_score += THREAT_CONTEXT_WEIGHTS["gateway_access"]

    # 3. プロセスの異常度
    process_name = event.get("process_name", "").lower()
    user_id = event.get("user_id", "unknown")
    try:
        common_apps = [app.lower() for app in security_log.get_user_common_applications(user_id) or []]
    except Exception:
        common_apps = []

    DANGEROUS_PROCESSES = [name for name, _ in PROCESS_SCORES["dangerous"]]
    if any(proc in process_name for proc in DANGEROUS_PROCESSES):
        threat_score += THREAT_CONTEXT_WEIGHTS["dangerous_process"]
    elif process_name and process_name not in common_apps:
        threat_score += THREAT_CONTEXT_WEIGHTS["unknown_process"]

    # 4. 操作の失敗
    if event.get("status") == "FAILED":
        threat_score += THREAT_CONTEXT_WEIGHTS["operation_failed"]

    # 5. ファイルパスの異常（システム領域へのアクセス）
    file_path = event.get("file_path", "").lower()
    SYSTEM_PATHS = [p for (p, _) in PATH_SCORES["high"]]
    if file_path:
        if any(path in file_path for path in SYSTEM_PATHS):
            threat_score += THREAT_CONTEXT_WEIGHTS["system_path_access"]

    return min(threat_score, THREAT_CONTEXT_WEIGHTS["max_threat"])

def security_event_to_state(event, security_log, config=None):
    """
    ログから意味構造テンソル状態を生成
    """
    config = config or getattr(self, "tune_conf", None)
    event_score = calculate_event_score_from_real_log(event, security_log)
    severity_level = calculate_severity_level(event_score)

    operation = event.get("operation", "FileRead")
    action_vector = calculate_action_vector_from_operation(operation)
    action_magnitude = np.linalg.norm(action_vector)

    threat_context = calculate_threat_context_from_log(event, security_log)
    trust_score = calculate_user_trust_from_history(event, security_log)

    # 🌟 モード判定はグローバル定数優先
    mode_map = getattr(config, "security_mode_mapping")
    thresholds = MODE_THRESHOLDS if 'MODE_THRESHOLDS' in globals() else {"critical": 60, "suspicious": 30, "investigating": 15}
    if event_score > thresholds["critical"]:
        mode_value = mode_map["critical"]
    elif event_score > thresholds["suspicious"]:
        mode_value = mode_map["suspicious"]
    elif event_score > thresholds["investigating"]:
        mode_value = mode_map["investigating"]
    else:
        mode_value = mode_map["normal"]

    # 🌟 extra拡張フィールドも将来拡張しやすく
    extra = {
        "vector": vectorize_state({
            "severity_level": severity_level,
            "action_magnitude": action_magnitude,
            "threat_context": threat_context,
            "trust_score": trust_score,
            "security_mode": mode_value,
            "event_score": event_score,
        }),
        # 追加フィールドをここに追記OK
    }

    state = create_state_dict(
        severity=severity_level,
        action_mag=action_magnitude,
        threat=threat_context,
        trust=trust_score,
        mode=mode_value,
        score=event_score,
        extra=extra
    )
    return state

def calculate_time_anomaly(timestamp_str):
    """
    ⏰ 時間異常スコアの計算 - テーブル＆定数利用版
    """
    try:
        from datetime import datetime
        if isinstance(timestamp_str, str):
            date_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        else:
            date_time = timestamp_str

        hour = date_time.hour
        weekday = date_time.weekday()

        # 時間帯スコア
        base_score = 0.0
        for (start, end), score in TIME_ANOMALY_SCORES:
            if start <= hour < end or (end < start and (hour >= start or hour < end)):
                base_score = score
                break

        # 週末加算
        if weekday >= 5:  # 土日
            base_score = min(base_score + WEEKEND_BONUS, 1.0)

        # 月曜早朝
        if weekday == MONDAY_MORNING[0] and MONDAY_MORNING[1] <= hour <= MONDAY_MORNING[2]:
            base_score *= MONDAY_MORNING[3]
        # 金曜夜
        if weekday == FRIDAY_NIGHT[0] and FRIDAY_NIGHT[1] <= hour <= FRIDAY_NIGHT[2]:
            base_score *= FRIDAY_NIGHT[3]

        return base_score

    except Exception:
        return 0.3

# ===== サポートクラス =====
class EMAFilter:
    """指数移動平均フィルタ（動的α対応Λ³拡張）"""
    def __init__(self, alpha=0.3):
        self.default_alpha = alpha
        self.value = None

    def _compute_alpha(self, new_value, context):
        previous_ema = self.value if self.value is not None else new_value
        change_magnitude = abs(new_value - previous_ema)
        alpha = self.default_alpha
        if change_magnitude > 30:
            alpha = 0.8
        elif change_magnitude < 5:
            alpha = 0.3
        if context:
            if context.get('is_critical_operation', False):
                alpha = min(0.9, alpha * 1.2)
            if context.get('is_routine_operation', False):
                alpha = max(0.2, alpha * 0.8)
            if context.get('sync_rate', 1.0) < 0.7:
                alpha = max(0.2, alpha * 0.85)
            if context.get('semantic_density', 1.0) > 1.5:
                alpha = min(0.95, alpha * 1.2)
        return alpha

    def update(self, new_value, context=None):
        alpha = self._compute_alpha(new_value, context)
        if self.value is None:
            self.value = new_value
        else:
            self.value = alpha * new_value + (1 - alpha) * self.value
        return self.value

# -------------------------------
# 🔥 SecurityBlock クラス
# -------------------------------

class SecurityBlock:
    def __init__(self, index, data, previous_hash, state_params, divergence, metadata=None):
        self.index = index
        self.data = data
        self.previous_hash = previous_hash
        self.state_params = state_params
        self.divergence = divergence
        self.metadata = metadata or {}
        self.hash = self.hashing()

    def hashing(self):
        key = hashlib.sha256()
        key.update(str(self.index).encode('utf-8'))
        key.update(str(self.data).encode('utf-8'))
        key.update(str(self.previous_hash).encode('utf-8'))
        for k in sorted(self.state_params.keys()):
            key.update(str(k).encode('utf-8'))
            val = self.state_params[k]
            key.update(val.tobytes() if isinstance(val, np.ndarray) else str(val).encode('utf-8'))
        for k in sorted(self.metadata.keys()):
            key.update(str(k).encode('utf-8'))
            key.update(str(self.metadata[k]).encode('utf-8'))
        key.update(str(self.divergence).encode('utf-8'))
        return key.hexdigest()

    def verify(self):
        return self.hash == self.hashing()

def add_block(self, data, state_params, divergence, metadata=None, fork_threshold=None):
    prev_block = self.blocks[-1] if self.blocks else None
    event_hash = ScalableSecurityChain.compute_event_hash(event)
    new_block = SecurityBlock(
        index=len(self.blocks),
        data=data,
        previous_hash=prev_block.hash if prev_block else "0"*64,
        state_params=state_params,
        divergence=divergence,
        metadata=metadata or {}
    )

    # --- フォーク判定 ---
    prev_vector = prev_block.state_params.get("vector") if prev_block else None
    new_vector  = state_params.get("vector")
    forked = False
    # 🌟個人/組織ごと自動切替
    if fork_threshold is None and hasattr(self, 'tune_conf'):
        if hasattr(self, 'user_id'):  # UserSecurityChain判定
            fork_threshold = getattr(self.tune_conf, 'fork_threshold_user', 12.0)
        else:  # 組織
            fork_threshold = getattr(self.tune_conf, 'fork_threshold_org', 15.0)
        if prev_vector is not None and new_vector is not None and fork_threshold:
            diff = np.linalg.norm(np.array(new_vector) - np.array(prev_vector))
            if diff > fork_threshold:
                forked = True
                new_block.metadata["forked"] = True
                alert_fork(self, prev_block, new_block)
            else:
                new_block.metadata["forked"] = False
        else:
            self.blocks.append(new_block)
            new_block.metadata["forked"] = False

def alert_fork(chain, prev_block, new_block, threshold=12.0):
    """
    フォーク（分岐）を検知した際に呼ばれるアラート関数。
    threshold: この値未満のdivergenceは無視（アラート出さない）
    """
    divergence = new_block.divergence
    if divergence < threshold:
        # 微小フォークはアラート出さない（静かにスルー）
        return  # 何もせず終了

    user_id = new_block.metadata.get("user_id", "unknown")
    dept = new_block.metadata.get("department", "unknown")
    index = new_block.index
    ts = new_block.metadata.get("timestamp", "N/A")

    print(f"🚨 [FORK ALERT] Block {index} in chain (User: {user_id}, Dept: {dept})")
    print(f"　時刻: {ts}")
    print(f"　逸脱スコア: {divergence:.2f}")
    print(f"　前ブロック: {prev_block.index} との分岐発生！")

    if hasattr(chain, "fork_alerts"):
        fork_alert = {
            "block_index": index,
            "user_id": user_id,
            "department": dept,
            "timestamp": ts,
            "divergence": divergence,
            "prev_block_index": prev_block.index,
            "event_data": new_block.data,
            "vector_diff": np.linalg.norm(
                np.array(new_block.state_params.get("vector")) -
                np.array(prev_block.state_params.get("vector"))
            ) if new_block.state_params.get("vector") is not None and prev_block.state_params.get("vector") is not None else None
        }
        chain.fork_alerts.append(fork_alert)
        print(f"　[LOG] fork_alertsに追加: {fork_alert}")
    else:
        print("　[WARN] chainにfork_alerts属性が無いよ！")

    # ここで通知やメール、Webhook/Slack等に連携可能
    # send_fork_alert_to_slack(user_id, dept, index, divergence, ts) みたいに

def cross_dept_alert_sync(manager, window_minutes=180):
    """
    全部署の重大アラート（critical, suspicious）が
    指定window_minutes以内で複数部門に跨って同時多発していないか検出！
    """
    all_alerts = []
    # 1. 全部署チェーンを走査
    for dept, chain in manager.department_chains.items():
        for block in chain.blocks:
            # セキュリティモード＆時刻取得
            mode = block.metadata.get("security_mode", "")
            timestamp = block.metadata.get("timestamp", None)
            if not timestamp and hasattr(block, "event"):
                timestamp = block.event.get("timestamp", None)
            if timestamp and isinstance(timestamp, str):
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
            if mode in ["critical", "suspicious"] and timestamp:
                all_alerts.append({
                    "dept": dept,
                    "time": timestamp,
                    "mode": mode,
                    "user": block.metadata.get("user_id", "")
                })
    # 2. 時系列でソート
    all_alerts.sort(key=lambda a: a["time"])

    # 3. window_minutes以内で「部署が違う」アラート同士を探索
    found = False
    for i in range(len(all_alerts)):
        for j in range(i + 1, len(all_alerts)):
            # window外はbreak
            if (all_alerts[j]["time"] - all_alerts[i]["time"]) > timedelta(minutes=window_minutes):
                break
            if all_alerts[i]["dept"] == all_alerts[j]["dept"]:
                continue
            # 組み合わせがヒット
            print(f"⚡️ Cross-dept alert: {all_alerts[i]['dept']} ({all_alerts[i]['user']}, {all_alerts[i]['mode']}, {all_alerts[i]['time']}) "
                  f"<-> {all_alerts[j]['dept']} ({all_alerts[j]['user']}, {all_alerts[j]['mode']}, {all_alerts[j]['time']})")
            found = True

    if not found:
        print("✅ 部署横断の同期アラートはありません")

def calc_event_priority(event):
    """
    イベントの優先度を計算する統一関数
    各クラスから呼び出される
    """
    op = event.get("operation", "").lower()
    status = event.get("status", "SUCCESS")
    file_size_kb = event.get("file_size_kb", 0)
    destination = event.get("destination_ip", "")
    source_ip = event.get("source_ip", "")
    base_score = 0.1

    # 操作の危険性に応じて明確に差別化！
    if op == "filedelete":
        base_score += 0.5  # 削除は極めて重要
    elif op == "filecopy":
        base_score += 0.6  # ファイルコピーはさらに重要（外部送信の可能性）
    elif op == "filewrite":
        base_score += 0.4  # 書き込みは重要だがコピーより低めに
    elif op == "networkconnect":
        base_score += 0.3  # 外部通信ベーススコアを強化
    elif op in ["processcreate", "processterminate"]:
        base_score += 0.2  # プロセス系は微増で差別化

    # 外部への通信なら追加でさらに強化
    if destination and not destination.startswith(("192.168.", "10.", "172.")):
        base_score += 0.3

    # 特大ファイルをより重要視
    if file_size_kb and file_size_kb > 50000:
        base_score += 0.3  # 特大ファイル加点を強化（0.2→0.3）

    # 時間帯（深夜・早朝）
    try:
        hour = int(event.get("timestamp", "00:00:00").split()[1].split(":")[0])
        if hour < 6 or hour > 22:
            base_score += 0.2  # 深夜帯は変更なし
    except Exception:
        pass

    # 失敗ステータスはそのまま
    if status == "FAILED":
        base_score += 0.15

    # 最大1.0に制限
    return min(base_score, 1.0)

# -------------------------------
# 🔥 組織のSecurityEventChain クラス（リファクタリング版）
# -------------------------------
class SecurityEventChain:
    def __init__(self, genesis_state_params=None, metadata=None,
                 tune_conf: ChainTuneConfig = ChainTuneConfig()):
        self.tune_conf = tune_conf # configでON時だけ生成
        self.scalable_backend = ScalableSecurityChain(tune_conf) if tune_conf.enable_scalable_backend else None
        self.feature_dim = len(genesis_state_params) if genesis_state_params else 5
        self.blocks = [self.get_genesis_block(genesis_state_params, metadata)]
        self.ema_filter = EMAFilter(alpha=tune_conf.ema_alpha_org)
        self.info_tensor = None
        self.cluster_model = None
        self.department = metadata.get("department") if metadata else None
        self.user_histories = {}
        self.resource_access_patterns = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.exceptional_model = None
        self.anomaly_model = None
        self.chain_context = { "pattern_history": {}, "anomaly_score": 0.0 }
        self.fork_alerts = []

    # === スケーラビリティ機能API（組織チェーン用・改良版） ===

    def compress_chain_history(self):
        if not self.tune_conf.enable_scalable_backend or not self.scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        if not self.blocks:
            print("⚠️ 組織履歴データが空です")
            return None
        compressor = self.scalable_backend.implement_time_series_compression()
        data = np.array([block.state_params for block in self.blocks[-1000:]])
        if data.size == 0:
            print("⚠️ 圧縮データが空です")
            return None
        return compressor.compress_block(data)

    def aggregate_chain_stats(self, window=1000):
        if not self.tune_conf.enable_scalable_backend or not self.scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        if not self.blocks:
            return {"mean": 0, "std": 0}
        data = np.array([block.state_params for block in self.blocks[-window:]])
        if data.size == 0:
            return {"mean": 0, "std": 0}
        return { "mean": np.mean(data, axis=0), "std": np.std(data, axis=0) }

    def run_chain_mini_batch_clustering(self, batch_size=500):
        if not self.tune_conf.enable_scalable_backend or not self.scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        if len(self.blocks) < 2:
            print("⚠️ 組織クラスタリングに十分なデータがありません")
            return None
        clusters = self.scalable_backend.hierarchical_clustering()
        data = [block.state_params for block in self.blocks[-batch_size*5:]]
        return clusters.fit_incremental(data, batch_size=batch_size)

    def check_chain_bloom_duplicate(self, known_events, new_event):
        if not self.tune_conf.enable_scalable_backend or not self.scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        bloom = self.scalable_backend.implement_bloom_filter()
        for e in known_events:
            bloom.add(str(e))
        return bloom.contains(str(new_event))

    def adaptive_chain_sampling_run(self, event):
        if not self.tune_conf.enable_scalable_backend or not self.scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        sampler = self.scalable_backend.implement_adaptive_sampling()
        priority_score = calc_event_priority(event)  # グローバル関数を使用
        return sampler.should_sample(priority_score)

    def get_genesis_block(self, state_params=None, metadata=None):
        state_params = state_params or {
            "severity_level": 0.0,
            "action_magnitude": 0.0,
            "threat_context": 0.0,
            "trust_score": 1.0,
            "security_mode": 0.0
        }
        metadata = metadata or {
            "department": self.department,
            "created_at": datetime.now().isoformat()
        }
        return SecurityBlock(
            index=0,
            data="GENESIS",
            previous_hash="arbitrary",
            state_params=state_params,
            divergence=0.0,
            metadata=metadata
        )

    # === 基準値設計 ===
    def set_normal_model(self, state_vector_list, n_clusters=3, min_cluster_size=10):
        """
        個人または部署の正常モデルを学習（スケーリング・クラスタ数自動調整付き）
        """
        # 1. 入力をnp配列へ
        if isinstance(state_vector_list, np.ndarray):
            vectors = state_vector_list
        elif state_vector_list and isinstance(state_vector_list[0], dict):
            vectors = np.array([vectorize_state(s) for s in state_vector_list])
        else:
            # 念のためsafe_values()通してもOK
            vectors = np.array([safe_values(v) for v in state_vector_list])

        self.feature_dim = vectors.shape[1]
        vectors_scaled = self.scaler.fit_transform(vectors)
        self.is_fitted = True

        # 2. 情報テンソルの計算
        self.info_tensor = compute_information_tensor(vectors_scaled)

        # 3. クラスタ数をデータ量で自動調整（最低1、最大n_clusters、推奨: 10件/クラスタ目安）
        n_data = len(vectors_scaled)
        actual_clusters = min(n_clusters, max(1, n_data // min_cluster_size))
        self.cluster_model = fit_normal_clusters(vectors_scaled, n_clusters=actual_clusters)

    # === 追加の正常パターンで既存モデルを更新 ===
    def update_normal_model(self, additional_states):
        if self.cluster_model is None:
            # 初回学習
            self.set_normal_model(additional_states)
        else:
            # 既存データと結合して再学習
            existing_states = []
            for block in self.blocks[1:]:
                if block.metadata.get("security_mode") == "normal":
                    existing_states.append(block.state_params)

            all_states = existing_states + additional_states
            self.set_normal_model(all_states, n_clusters=3)

    def update_anomaly_model(self, abnormal_states):
        """
        異常クラスタ（anomaly cluster）データをモデルに追加して再学習するメソッド。
        - abnormal_states: 異常な状態のリスト（dict形式の場合テンソル化が必要）
        """
        # --- dict型ならvectorize_stateでテンソル化 ---
        if abnormal_states and isinstance(abnormal_states[0], dict):
            vectors = np.array([vectorize_state(s) for s in abnormal_states])
        else:
            vectors = np.array(abnormal_states)

        if not hasattr(self, "anomaly_model") or self.anomaly_model is None:
            # 初回なら新規クラスタ学習
            self.anomaly_model = fit_normal_clusters(vectors, n_clusters=2)
        else:
            existing_states = getattr(self, "anomaly_states", [])
            if existing_states and isinstance(existing_states[0], dict):
                existing_states = [vectorize_state(s) for s in existing_states]
            all_states = np.vstack([existing_states, vectors]) if len(existing_states) > 0 else vectors
            self.anomaly_model = fit_normal_clusters(all_states, n_clusters=2)
            self.anomaly_states = all_states

        self.anomaly_states = vectors  # 最新状態も上書き

    def evaluate_by_cluster(self, new_state, weights_l1=None, weights_l2=None, config=None):
        """外部関数を使用してクラスタ評価（config対応）"""
        config = config or getattr(self, "tune_conf", None)
        if self.info_tensor is None or self.cluster_model is None:
            raise ValueError("モデルが未設定です")
        if config is not None:
            if weights_l1 is None:
                weights_l1 = config.weights_l1.copy()
            if weights_l2 is None:
                weights_l2 = config.weights_l2.copy()

        return evaluate_security_event_by_cluster(
            new_state, self.cluster_model, self.info_tensor, weights_l1, weights_l2, config=config
        )

    def evaluate_by_multi_cluster(self, new_state, weights_l1=None, weights_l2=None, config=None):
        """
        スケーリングを考慮したマルチクラスタ評価
        """
        config = config or getattr(self, "tune_conf", None)
        vec = vectorize_state(new_state)
        if getattr(self, "is_fitted", False):  # self.is_fitted がTrueなら
            vec_scaled = self.scaler.transform(vec.reshape(1, -1))[0]
        else:
            vec_scaled = vec

        return classify_event_by_multi_clusters(
            vec_scaled,
            self.cluster_model,
            self.exceptional_model,
            self.anomaly_model,
            self.info_tensor,
            config,
            weights_l1,
            weights_l2
        )

    # ===== 共通のadd_block_by_cluster_evalメソッド =====
    def add_block_by_cluster_eval(
        self, data, state_params, raw_event=None, step=None,
        system="SecurityMonitor", experiment="DivergenceCluster"
    ):
        """クラスタ評価によるブロック追加（基底実装）"""
        user_id = raw_event.get("user_id") if raw_event else "unknown"
        user_history = self.get_user_history(user_id)

        # 1. クラスタ＆divergence（子クラスでオーバーライド可能）
        score, cluster, verdict = self._evaluate_cluster(state_params, user_history, raw_event)

        # 2. セキュリティ判定
        mode, weighted_score, reason = self._apply_security_rules(
            score, verdict, state_params, raw_event, user_history
        )

        # 3. メタデータ作成（子クラスで拡張可能）
        metadata = self._create_metadata(
            mode, weighted_score, step, system, experiment,
            cluster, score, verdict, reason, raw_event
        )

        # 4. ブロック追加
        add_block(self, data, state_params, divergence=score, metadata=metadata)

        # 5. 後処理（履歴更新など）
        self._post_process(user_id, raw_event, mode, weighted_score)

        # 6. 結果返却（子クラスで拡張可能）
        return self._create_result(mode, score, cluster, verdict, reason)

    def _evaluate_cluster(self, state_params, user_history, raw_event):
        """クラスタ評価（子クラスでオーバーライド）"""
        result = self.evaluate_by_multi_cluster(state_params)
        score = result["best_divergence"]
        cluster = result["cluster"]
        verdict = result["verdict"]

        score = self.calculate_context_aware_divergence(score, user_history, raw_event)
        return score, cluster, verdict

    def _apply_security_rules(self, score, verdict, state_params, raw_event, user_history):
        """セキュリティルール適用"""
        if not hasattr(self, "chain_context"):
            self.chain_context = {"pattern_history": {}, "anomaly_score": 0.0}

        # 初期値
        mode = verdict
        reason = ""

        if raw_event:
            trust_score = user_history.get("trust_score", 0.75)
            verdict2, reason = unified_security_judge(
                raw_event, user_history, self.chain_context,
                normal_div=score, trust_score=trust_score
            )
            if verdict2 != "normal":
                verdict = verdict2
                if verdict == "latent_suspicious":
                    score = max(score, 8.0)

        # 時刻系
        timestamp = raw_event.get("timestamp", raw_event.get("event_time")) if raw_event else datetime.now()
        if isinstance(timestamp, str):
            event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            event_time = timestamp

        time_of_day = self.get_time_of_day(event_time.hour)
        dynamic_threshold = self.get_adaptive_threshold(time_of_day)

        # モード分類
        mode, weighted_score = classify_security_mode_auto(
            score, state_params,
            previous_ema=self.ema_filter.value,
            config=self._get_config_dict(),
            threshold_override=dynamic_threshold,
            operation=raw_event.get("operation") if raw_event else None
        )

        # 再度ルール判定（優先度による上書き）
        if raw_event:
            trust_score = user_history.get("trust_score", 0.75)
            verdict2, reason = unified_security_judge(
                raw_event, user_history, self.chain_context,
                normal_div=score, trust_score=trust_score
            )
            priority = {
                "normal": 0,
                "investigating": 1,
                "latent_suspicious": 2,
                "suspicious": 3,
                "critical": 4
            }
            if priority.get(verdict2, 0) > priority.get(mode, 0):
                print(f"[OVERRIDE↑] {mode} → {verdict2} by unified_security_judge: {reason}")
                mode = verdict2

        return mode, weighted_score, reason

    def _get_config_dict(self):
        """設定辞書取得（子クラスでオーバーライド）"""
        return self.tune_conf.to_org_dict()

    def _create_metadata(self, mode, weighted_score, step, system, experiment,
                        cluster, score, verdict, reason, raw_event):
        """メタデータ作成"""
        time_of_day = None
        dynamic_threshold = None

        if raw_event:
            timestamp = raw_event.get("timestamp", raw_event.get("event_time"))
            if isinstance(timestamp, str):
                event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            else:
                event_time = timestamp or datetime.now()

            time_of_day = self.get_time_of_day(event_time.hour)
            dynamic_threshold = self.get_adaptive_threshold(time_of_day)

        metadata = {
            "security_mode": mode,
            "weighted_score": float(weighted_score),
            "step": step,
            "system": system,
            "experiment": experiment,
            "cluster_id": cluster,
            "divergence_score": score,
            "verdict": verdict,
            "department": self.department,
            "context_adjusted": True,
            "dynamic_threshold": dynamic_threshold
        }

        if raw_event:
            for k in [
                "event_time", "user_id", "target_resource", "event_type",
                "source_ip", "event_score", "action", "department",
                "timestamp", "operation", "file_path", "file_size_kb",
                "process_name", "destination_ip", "status"
            ]:
                if k in raw_event:
                    metadata[k] = raw_event[k]

        return metadata

    def _post_process(self, user_id, raw_event, mode, weighted_score):
        """後処理（履歴更新など）"""
        self.ema_filter.update(weighted_score)

        if raw_event:
            timestamp = raw_event.get("timestamp", raw_event.get("event_time"))
            if isinstance(timestamp, str):
                event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            else:
                event_time = timestamp or datetime.now()

            self.update_user_history(user_id, event_time, mode)

            if "file_path" in raw_event:
                self.update_resource_pattern(user_id, raw_event["file_path"], event_time)

    def _create_result(self, mode, score, cluster, verdict, reason):
        """結果辞書作成"""
        return {
            "status": mode,
            "divergence": score,
            "added": True,
            "cluster_id": cluster,
            "verdict": verdict,
            "alert_level": "HIGH" if mode in ["critical", "suspicious", "latent_suspicious"] else "LOW",
            "reason": reason if verdict != "normal" else "",
        }

    def get_time_of_day(self, hour):
        """時間帯を判定"""
        if BUSINESS_HOURS["start"] <= hour < 12:
            return "morning"
        elif 12 <= hour < BUSINESS_HOURS["end"]:
            return "afternoon"
        elif BUSINESS_HOURS["end"] <= hour < 22:
            return "evening"
        else:
            return "night"

    def get_adaptive_threshold(self, time_of_day):
        """時間帯に応じた適応的閾値を取得"""
        dept_tbl = self.tune_conf.dept_threshold_table
        dept_thresholds = dept_tbl.get(self.department, {})
        # デフォルト値も定数化
        return dept_thresholds.get(time_of_day, THRESHOLDS["default_adaptive"])

    def calculate_context_aware_divergence(self, base_divergence, user_history, raw_event=None):
        """コンテキストを考慮したDivergence計算"""
        adjusted_divergence = base_divergence

        # 最後の正常アクセスからの経過時間による調整
        if user_history.get("last_normal_access"):
            hours_since_normal = (datetime.now() - user_history["last_normal_access"]).total_seconds() / 3600
            if hours_since_normal < NORMAL_ACCESS_GRACE_PERIOD:
                adjusted_divergence *= DIVERGENCE_MULTIPLIERS["recent_normal_access"]

        # アクセス回数による調整
        if user_history.get("access_count", 0) > MIN_ACCESS_COUNT:
            adjusted_divergence *= DIVERGENCE_MULTIPLIERS["frequent_user"]

        if raw_event:
            # ディレクトリアクセス頻度による調整
            if "file_path" in raw_event:
                directory = self._extract_directory(raw_event["file_path"])
                if self.is_frequent_directory(user_history.get("user_id"), directory):
                    adjusted_divergence *= DIVERGENCE_MULTIPLIERS["frequent_directory"]

            # 安全なプロセスによる調整
            process = raw_event.get("process_name", "").lower()
            if process in SAFE_PROCESSES:
                adjusted_divergence *= DIVERGENCE_MULTIPLIERS["safe_process"]

            # ビジネスコンテキストによる調整
            if "_business_context" in raw_event:
                contexts = raw_event["_business_context"]
                for context, multiplier in BUSINESS_CONTEXT_MULTIPLIERS.items():
                    if context in contexts:
                        adjusted_divergence *= multiplier

        return adjusted_divergence

    def update_resource_pattern(self, user_id, file_path, access_time):
        """リソースアクセスパターンを記録"""
        directory = self._extract_directory(file_path)

        if user_id not in self.resource_access_patterns:
            self.resource_access_patterns[user_id] = {}

        if directory not in self.resource_access_patterns[user_id]:
            self.resource_access_patterns[user_id][directory] = {
                "count": 0,
                "first_access": access_time,
                "last_access": None
            }

        pattern = self.resource_access_patterns[user_id][directory]
        pattern["count"] += 1
        pattern["last_access"] = access_time

    def is_frequent_directory(self, user_id, directory):
        """頻繁にアクセスするディレクトリかチェック"""
        if user_id in self.resource_access_patterns:
            if directory in self.resource_access_patterns[user_id]:
                access_count = self.resource_access_patterns[user_id][directory]["count"]
                return access_count > FREQUENT_ACCESS_THRESHOLD
        return False

    def update_user_history(self, user_id, event_time, status):
        """ユーザーのアクセス履歴を更新"""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = self._create_empty_user_history()

        history = self.user_histories[user_id]
        history["access_count"] += 1
        history["last_access"] = event_time

        if status == "normal":
            history["last_normal_access"] = event_time
            history["failed_attempts"] = 0
        elif status in CRITICAL_STATUSES:
            history["failed_attempts"] += 1
            history["last_failed"] = event_time

    def get_user_history(self, user_id):
        """ユーザー履歴を取得"""
        return self.user_histories.get(user_id, self._create_empty_user_history())

    def get_recent_alerts(self, n=10, min_severity="suspicious"):
        """最近のアラートを取得"""
        min_level = SEVERITY_ORDER.get(min_severity, 1)
        alerts = []

        # 検索範囲を効率化
        search_range = min(len(self.blocks), n * 2)

        for block in reversed(self.blocks[-search_range:]):
            if self._should_include_alert(block, min_level):
                alert = self._create_alert_from_block(block)
                alerts.append(alert)

                if len(alerts) >= n:
                    break

        return alerts

    def verify(self, verbose=True):
        """チェーンの整合性を検証"""
        if len(self.blocks) < 2:
            if verbose:
                print("✅ Chain has only genesis block")
            return True

        for i in range(1, len(self.blocks)):
            # 前のブロックのハッシュチェック
            if self.blocks[i].previous_hash != self.blocks[i-1].hash:
                if verbose:
                    print(f"❌ Block {i} has invalid previous hash")
                    print(f"  Expected: {self.blocks[i-1].hash}")
                    print(f"  Got: {self.blocks[i].previous_hash}")
                return False

            # ブロック自体の検証
            if not self.blocks[i].verify():
                if verbose:
                    print(f"❌ Block {i} failed self-verification")
                return False

        if verbose:
            print(f"✅ Chain is valid ({len(self.blocks)} blocks)")
        return True

    @classmethod
    def from_json(cls, filepath):
        """JSONファイルからチェーンを読み込み"""
        with open(filepath, "r", encoding="utf-8") as f:
            blocks_data = json.load(f)

        if not isinstance(blocks_data, list):
            raise ValueError("JSON file must contain a list of blocks")

        if not blocks_data:
            raise ValueError("JSON file contains no blocks")

        # データフォーマットの判定
        if all(isinstance(item, dict) for item in blocks_data):
            # LanScope Cat形式
            if self._is_lanscope_format(blocks_data):
                print("🧪 LanScope Cat format log detected. Building initial chain...")
                return build_initial_security_chain(blocks_data, SecurityLogData())

            # レガシー形式
            elif self._is_legacy_format(blocks_data):
                print("🧪 Legacy security event log detected. Building initial chain...")
                return build_initial_security_chain(blocks_data, SecurityLogData())

            # ブロックチェーン形式
            elif self._is_blockchain_format(blocks_data):
                return cls._from_blockchain_format(blocks_data)

        raise ValueError("Unknown data format in JSON file")

    @staticmethod
    def _serialize_anything(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: SecurityEventChain._serialize_anything(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [SecurityEventChain._serialize_anything(x) for x in obj]
        else:
            return obj

    def export_to_json(self, filepath):
        blocks_data = []
        for block in self.blocks:
            block_data = {
                "index": block.index,
                "data": SecurityEventChain._serialize_anything(block.data),
                "previous_hash": block.previous_hash,
                "state_params": SecurityEventChain._serialize_anything(block.state_params),
                "divergence": block.divergence,
                "metadata": SecurityEventChain._serialize_anything(block.metadata),
            }
            blocks_data.append(block_data)

        with open(filepath, "w", encoding="utf-8") as f:
            import json
            json.dump(blocks_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Exported {len(blocks_data)} blocks to {filepath}")

    #===== プライベートヘルパーメソッド =====
    def _serialize_state_params(self, state_params):
        """
        state_params（dict）をJSONで保存可能な形に再帰的に変換（例：np.ndarray→list, np.float→float など）
        """
        def serialize(v):
            # ndarray → list
            if isinstance(v, np.ndarray):
                return v.tolist()
            # NumPyスカラー型 → Python型
            elif isinstance(v, np.floating):
                return float(v)
            elif isinstance(v, np.integer):
                return int(v)
            # dict → 再帰
            elif isinstance(v, dict):
                return {kk: serialize(vv) for kk, vv in v.items()}
            # list/tuple → 再帰
            elif isinstance(v, (list, tuple)):
                return [serialize(x) for x in v]
            else:
                return v

        return {k: serialize(v) for k, v in state_params.items()}

    def _extract_directory(self, file_path):
        """ファイルパスからディレクトリを抽出"""
        return "\\".join(file_path.split("\\")[:-1])

    def _create_empty_user_history(self):
        """空のユーザー履歴を作成"""
        return {
            "access_count": 0,
            "last_normal_access": None,
            "last_access": None,
            "failed_attempts": 0,
            "last_failed": None
        }

    def _should_include_alert(self, block, min_level):
        """アラートに含めるべきかチェック"""
        mode = block.metadata.get("security_mode")
        if mode in SEVERITY_ORDER:
            return SEVERITY_ORDER[mode] >= min_level
        return False

    def _create_alert_from_block(self, block):
        """ブロックからアラートを作成"""
        alert = {
            "block_index": block.index,
            "time": block.metadata.get("timestamp", block.metadata.get("event_time", "Unknown")),
            "user": block.metadata.get("user_id", "Unknown"),
            "mode": block.metadata.get("security_mode"),
            "score": block.divergence,
            "department": block.metadata.get("department", "Unknown")
        }

        # ターゲット情報の追加
        if "file_path" in block.metadata:
            alert["target"] = block.metadata["file_path"]
        elif "destination_ip" in block.metadata:
            alert["target"] = f"Network: {block.metadata['destination_ip']}"
        else:
            alert["target"] = block.metadata.get("target_resource", "Unknown")

        # 操作情報の追加
        if "operation" in block.metadata:
            alert["operation"] = block.metadata["operation"]

        return alert

# -------------------------------
# 👤 UserSecurityChain クラス（リファクタリング版）
# -------------------------------
class UserSecurityChain(SecurityEventChain):
    def __init__(self, user_id, department, tune_conf: ChainTuneConfig = ChainTuneConfig()):
        super().__init__(metadata={"user_id": user_id, "department": department}, tune_conf=tune_conf)
        self.user_id = user_id
        self.department = department
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.ema_filter = EMAFilter(alpha=tune_conf.ema_alpha_user)
        self.fork_alerts = []
        self.baseline_behavior = {
            "typical_hours": [],
            "common_directories": [],
            "common_processes": [],
            "typical_locations": [],
            "operation_frequency": {},
            "file_size_patterns": {},
            "file_access_patterns": {}
        }

        # 🟩 個人用chain_context（部署・グローバルと独立可）
        self.chain_context = { "pattern_history": {}, "anomaly_score": 0.0 }

        # 🟩 スケーラブル裏方チェーン（個人用） configでON時のみ生成
        self.conf = tune_conf
        self.scalable_backend = ScalableSecurityChain(tune_conf) if tune_conf.enable_scalable_backend else None

    # ===== スケーラブル個人API（改良版） =====
    def compress_user_history(self):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        compressor = self.scalable_backend.implement_time_series_compression()
        # 直近100回分のアクセスを圧縮（空ならNoneを返す）
        if len(self.blocks) == 0:
            print("⚠️ ユーザー履歴データが空です")
            return None
        data = np.array([block.state_params for block in self.blocks[-100:]])
        if data.size == 0:
            print("⚠️ 圧縮データが空です")
            return None
        return compressor.compress_block(data)

    def aggregate_user_behavior(self):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        if len(self.blocks) == 0:
            return {"mean": 0, "std": 0}
        data = np.array([block.state_params for block in self.blocks[-24:]])
        if data.size == 0:
            return {"mean": 0, "std": 0}
        return {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0)
        }

    def run_user_mini_batch_clustering(self, batch_size=30):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        clusters = self.scalable_backend.hierarchical_clustering()
        if len(self.blocks) < 2:
            print("⚠️ 個人クラスタリングに十分なデータがありません")
            return None
        data = [block.state_params for block in self.blocks[-100:]]
        return clusters.fit_incremental(data, batch_size=batch_size)

    def check_bloom_duplicate(self, known_events, new_event):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        bloom = self.scalable_backend.implement_bloom_filter()
        for e in known_events:
            bloom.add(str(e))
        return bloom.contains(str(new_event))

    def adaptive_sampling_run(self, event):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        sampler = self.scalable_backend.implement_adaptive_sampling()
        priority_score = calc_event_priority(event)  # グローバル関数を使用
        return sampler.should_sample(priority_score)

    def set_normal_model(self, state_vector_list, n_clusters=3, min_cluster_size=10):
        """
        個人または部署の正常モデルを学習（スケーリング・クラスタ数自動調整付き）
        """
        # 1. 入力をnp配列へ
        if isinstance(state_vector_list, np.ndarray):
            vectors = state_vector_list
        elif state_vector_list and isinstance(state_vector_list[0], dict):
            vectors = np.array([vectorize_state(s) for s in state_vector_list])
        else:
            # 念のためsafe_values()通してもOK
            vectors = np.array([safe_values(v) for v in state_vector_list])

        self.feature_dim = vectors.shape[1]
        vectors_scaled = self.scaler.fit_transform(vectors)
        self.is_fitted = True

        # 2. 情報テンソルの計算
        self.info_tensor = compute_information_tensor(vectors_scaled)

        # 3. クラスタ数をデータ量で自動調整（最低1、最大n_clusters、推奨: 10件/クラスタ目安）
        n_data = len(vectors_scaled)
        actual_clusters = min(n_clusters, max(1, n_data // min_cluster_size))
        self.cluster_model = fit_normal_clusters(vectors_scaled, n_clusters=actual_clusters)

    def set_abnormal_model(self, abnormal_states, n_clusters=2):
        """外部関数を使用して逸脱モデルを設定"""
        vectors_scaled = self.scaler.transform(vectors) if self.is_fitted else vectors
        self.abnormal_model = fit_normal_clusters(vectors_scaled, n_clusters=n_clusters)

    def set_exceptional_model(self, exceptional_states, n_clusters=2):
        """外部関数を使用して例外モデルを設定"""
        vectors = np.array([vectorize_state(s) for s in exceptional_states])
        vectors_scaled = self.scaler.transform(vectors) if self.is_fitted else vectors
        self.exceptional_model = fit_normal_clusters(vectors_scaled, n_clusters=n_clusters)

    def evaluate_by_multi_cluster(self, new_state, weights_l1=None, weights_l2=None):
        """スケーリングを考慮したマルチクラスタ評価"""
        vec = vectorize_state(new_state)
        if self.is_fitted:
            vec_scaled = self.scaler.transform(vec.reshape(1, -1))[0]
        else:
            vec_scaled = vec

        result = classify_event_by_multi_clusters(
            vec_scaled,
            self.cluster_model,
            self.exceptional_model,
            self.abnormal_model if hasattr(self, 'abnormal_model') else None,
            self.info_tensor,
            config=getattr(self, 'tune_conf', None),
            weights_l1=weights_l1,
            weights_l2=weights_l2
        )

        return result["verdict"], result["best_divergence"], result["cluster"]

    # ===== 個人用のオーバーライド =====
    def _evaluate_cluster(self, state_params, user_history, raw_event):
        """個人用のクラスタ評価"""
        verdict, score, cluster = self.evaluate_by_multi_cluster(state_params)
        score = self.calculate_context_aware_divergence(score, user_history, raw_event)
        return score, cluster, verdict

    def _get_config_dict(self):
        """個人用の設定辞書"""
        return self.tune_conf.to_user_dict()

    def _create_result(self, mode, score, cluster, verdict, reason):
        """個人用の結果辞書（deviation_scoreを追加）"""
        result = super()._create_result(mode, score, cluster, verdict, reason)

        # 個人特有の情報を追加
        if hasattr(self, '_last_raw_event'):
            result["deviation_score"] = self.calculate_behavior_deviation(self._last_raw_event)
        else:
            result["deviation_score"] = 0.0

        return result

    def _post_process(self, user_id, raw_event, mode, weighted_score):
        """個人用の後処理（ファイルアクセスパターン更新を追加）"""
        super()._post_process(user_id, raw_event, mode, weighted_score)

        # raw_eventを保存（deviation_score計算用）
        self._last_raw_event = raw_event

        # 個人特有の処理
        if raw_event and "file_path" in raw_event:
            self.update_file_access_pattern(raw_event)

    # ===== 個人特有のメソッド（既存のまま） =====

    def update_normal_model(self, additional_states):
        """
        追加の正常パターンで既存モデルを更新
        """
        if self.cluster_model is None:
            # 初回学習
            self.set_normal_model(additional_states)
        else:
            # 既存データと結合して再学習
            existing_states = []
            for block in self.blocks[1:]:
                if block.metadata.get("security_mode") == "normal":
                    existing_states.append(block.state_params)

            all_states = existing_states + additional_states
            self.set_normal_model(all_states, n_clusters=3)

    def calculate_behavior_deviation(self, event):
        """行動逸脱度の計算"""
        deviation_score = 0.0

        # タイムスタンプの処理
        event_time = self._parse_event_time(event)
        hour = event_time.hour

        # 1. 時間帯の逸脱チェック
        if self.baseline_behavior["typical_hours"]:
            if hour not in self.baseline_behavior["typical_hours"]:
                min_diff = min([abs(hour - h) for h in self.baseline_behavior["typical_hours"]])
                # 最大12時間差を0-1に正規化し、重み付け
                deviation_score += (min_diff / DEVIATION_SCORES["hour_diff_divisor"]) * DEVIATION_SCORES["hour_weight"]

        # 2. ディレクトリアクセスの逸脱チェック
        file_path = event.get("file_path", "")
        if file_path and self.baseline_behavior["common_directories"]:
            directory = self._extract_directory(file_path)
            if directory not in self.baseline_behavior["common_directories"]:
                deviation_score += DEVIATION_SCORES["unknown_directory"]

                # 高リスクディレクトリへのアクセス
                if self._is_high_risk_directory(directory):
                    deviation_score += DEVIATION_SCORES["high_risk_directory"]

        # 3. ファイルサイズの異常チェック
        if file_path and "file_size_kb" in event:
            extension = self._extract_extension(file_path)
            file_size = event["file_size_kb"]

            if extension in self.baseline_behavior["file_size_patterns"]:
                sizes = self.baseline_behavior["file_size_patterns"][extension]
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    if file_size > avg_size * FILE_SIZE_ANOMALY_MULTIPLIER:
                        deviation_score += DEVIATION_SCORES["abnormal_file_size"]

        # 4. プロセスの逸脱チェック
        process_name = event.get("process_name", "")
        if process_name and self.baseline_behavior["common_processes"]:
            if process_name not in self.baseline_behavior["common_processes"]:
                deviation_score += DEVIATION_SCORES["unknown_process"]

                # 危険なプロセスのチェック
                if self._is_dangerous_process(process_name):
                    deviation_score += DEVIATION_SCORES["dangerous_process"]

        # 5. 操作頻度の逸脱チェック
        operation = event.get("operation", "")
        if operation and self.baseline_behavior["operation_frequency"]:
            total_ops = sum(self.baseline_behavior["operation_frequency"].values())
            if total_ops > 0:
                op_freq = self.baseline_behavior["operation_frequency"].get(operation, 0)

                if op_freq == 0:
                    deviation_score += DEVIATION_SCORES["new_operation"]
                elif op_freq / total_ops < OPERATION_FREQUENCY_THRESHOLD:
                    deviation_score += DEVIATION_SCORES["rare_operation"]

        # 6. IPアドレスの逸脱チェック
        source_ip = event.get("source_ip", "")
        if source_ip and self.baseline_behavior["typical_locations"]:
            if source_ip not in self.baseline_behavior["typical_locations"]:
                if self._is_internal_ip(source_ip):
                    deviation_score += DEVIATION_SCORES["unknown_internal_ip"]
                else:
                    deviation_score += DEVIATION_SCORES["unknown_external_ip"]

        # スコアを0-1の範囲に制限
        return min(deviation_score, DEVIATION_SCORES["max_score"])

    def learn_baseline(self, historical_events):
        """ベースライン行動を学習"""
        for event in historical_events:
            # タイムスタンプ処理
            event_time = self._parse_event_time(event)
            hour = event_time.hour

            # 1. 典型的な活動時間の学習
            if hour not in self.baseline_behavior["typical_hours"]:
                self.baseline_behavior["typical_hours"].append(hour)

            # 2. ファイル関連パターンの学習
            file_path = event.get("file_path", "")
            if file_path:
                self._learn_file_patterns(file_path, event)

            # 3. プロセスパターンの学習
            process_name = event.get("process_name", "")
            if process_name and process_name not in self.baseline_behavior["common_processes"]:
                self.baseline_behavior["common_processes"].append(process_name)

            # 4. 操作頻度の学習
            operation = event.get("operation", event.get("event_type", ""))
            if operation:
                self.baseline_behavior["operation_frequency"][operation] = \
                    self.baseline_behavior["operation_frequency"].get(operation, 0) + 1

            # 5. アクセス元IPの学習
            source_ip = event.get("source_ip", "")
            if source_ip and source_ip not in self.baseline_behavior["typical_locations"]:
                self.baseline_behavior["typical_locations"].append(source_ip)

        # 学習後の統計情報を出力（デバッグ用）
        self._print_baseline_summary()

    def update_file_access_pattern(self, event):
        """ファイルアクセスパターンを更新"""
        file_path = event.get("file_path", "")
        if not file_path:
            return

        directory = self._extract_directory(file_path)
        extension = self._extract_extension(file_path)

        # パターンの初期化
        if directory not in self.baseline_behavior["file_access_patterns"]:
            self.baseline_behavior["file_access_patterns"][directory] = {
                "count": 0,
                "extensions": {},
                "avg_size": 0,
                "last_access": None,
                "first_access": datetime.now()
            }

        pattern = self.baseline_behavior["file_access_patterns"][directory]

        # アクセス情報の更新
        pattern["count"] += 1
        pattern["last_access"] = datetime.now()

        # 拡張子別カウント
        if extension not in pattern["extensions"]:
            pattern["extensions"][extension] = 0
        pattern["extensions"][extension] += 1

        # 平均ファイルサイズの更新（インクリメンタル計算）
        if "file_size_kb" in event:
            current_avg = pattern["avg_size"]
            new_size = event["file_size_kb"]
            # 新しい平均 = (現在の平均 * (n-1) + 新しい値) / n
            pattern["avg_size"] = (current_avg * (pattern["count"] - 1) + new_size) / pattern["count"]

        # アクセス頻度が高いディレクトリを特定
        if pattern["count"] >= FREQUENT_DIRECTORY_ACCESS_THRESHOLD:
            pattern["is_frequent"] = True

    def calculate_context_aware_divergence(self, base_divergence, user_history, raw_event=None):
        """個人の文脈を考慮したDivergence計算"""
        # 親クラスのメソッドを呼び出し
        adjusted_divergence = super().calculate_context_aware_divergence(
            base_divergence, user_history, raw_event
        )

        if raw_event:
            # 1. 頻繁にアクセスするディレクトリかチェック
            file_path = raw_event.get("file_path", "")
            if file_path:
                directory = self._extract_directory(file_path)
                if directory in self.baseline_behavior["file_access_patterns"]:
                    access_pattern = self.baseline_behavior["file_access_patterns"][directory]
                    access_count = access_pattern["count"]

                    # 頻繁なアクセスディレクトリは信頼度が高い
                    if access_count > FREQUENT_DIRECTORY_ACCESS_THRESHOLD:
                        adjusted_divergence *= DIVERGENCE_MULTIPLIERS["personal_frequent_directory"]

                    # 最近アクセスしたディレクトリも考慮
                    if access_pattern.get("last_access"):
                        hours_since_access = (datetime.now() - access_pattern["last_access"]).total_seconds() / 3600
                        if hours_since_access < RECENT_ACCESS_HOURS:
                            adjusted_divergence *= DIVERGENCE_MULTIPLIERS["recent_access"]

            # 2. よく使うプロセスかチェック
            process = raw_event.get("process_name", "")
            if process in self.baseline_behavior["common_processes"]:
                op_freq = self.baseline_behavior["operation_frequency"]
                total_operations = sum(op_freq.values())

                # 十分な操作履歴がある場合は信頼度を上げる
                if total_operations > THRESHOLDS["abnormal_access_count"]:
                    adjusted_divergence *= DIVERGENCE_MULTIPLIERS["trusted_user_process"]

            # 3. 個人の作業パターンを考慮
            operation = raw_event.get("operation", "")
            if operation in self.baseline_behavior["operation_frequency"]:
                op_count = self.baseline_behavior["operation_frequency"][operation]
                total_ops = sum(self.baseline_behavior["operation_frequency"].values())

                if total_ops > 0 and op_count / total_ops > HIGH_FREQUENCY_OPERATION_RATIO:
                    # 頻繁に行う操作は信頼度が高い
                    adjusted_divergence *= DIVERGENCE_MULTIPLIERS["frequent_operation"]

        return adjusted_divergence

    # ===== プライベートヘルパーメソッド =====
    def _parse_event_time(self, event):
        """イベントからタイムスタンプを抽出して解析"""
        timestamp = event.get("timestamp", event.get("event_time"))
        if isinstance(timestamp, str):
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return timestamp or datetime.now()

    def _extract_directory(self, file_path):
        """ファイルパスからディレクトリを抽出"""
        return "\\".join(file_path.split("\\")[:-1])

    def _extract_extension(self, file_path):
        """ファイルパスから拡張子を抽出"""
        return file_path.split(".")[-1].lower() if "." in file_path else "unknown"

    def _is_high_risk_directory(self, directory):
        """高リスクディレクトリかチェック"""
        directory_lower = directory.lower()
        return any(risk_path in directory_lower for risk_path in HIGH_RISK_PATHS)

    def _is_dangerous_process(self, process_name):
        """危険なプロセスかチェック"""
        process_lower = process_name.lower()
        return any(danger in process_lower for danger in DANGEROUS_PROCESSES)

    def _is_internal_ip(self, ip_address):
        """内部IPアドレスかチェック"""
        return ip_address.startswith(PRIVATE_IP_PREFIXES)

    def _learn_file_patterns(self, file_path, event):
        """ファイル関連のパターンを学習"""
        directory = self._extract_directory(file_path)

        # ディレクトリパターンの学習
        if directory and directory not in self.baseline_behavior["common_directories"]:
            self.baseline_behavior["common_directories"].append(directory)

        # ファイルサイズパターンの学習
        file_size = event.get("file_size_kb", 0)
        extension = self._extract_extension(file_path)

        if extension not in self.baseline_behavior["file_size_patterns"]:
            self.baseline_behavior["file_size_patterns"][extension] = []
        self.baseline_behavior["file_size_patterns"][extension].append(file_size)

    def _print_baseline_summary(self):
        """ベースライン学習結果のサマリーを出力（デバッグ用）"""
        print(f"👤 ユーザー {self.user_id} のベースライン学習完了:")
        print(f"  - 活動時間帯: {len(self.baseline_behavior['typical_hours'])}パターン")
        print(f"  - 共通ディレクトリ: {len(self.baseline_behavior['common_directories'])}箇所")
        print(f"  - 使用プロセス: {len(self.baseline_behavior['common_processes'])}種類")
        print(f"  - 操作種別: {len(self.baseline_behavior['operation_frequency'])}種類")
        print(f"  - アクセス元IP: {len(self.baseline_behavior['typical_locations'])}箇所")

# -----------------
# 🔥 組織横断・全体管理クラス
# -----------------
class DepartmentSecurityChainManager:
    def __init__(self, tune_conf: ChainTuneConfig = ChainTuneConfig()):
        self.conf = tune_conf
        self.department_chains = {}
        self.user_department_map = {}
        self.security_log = SecurityLogData()
        self.cross_department_access = {}
        self.exceptional_patterns = {}
        self.user_chains = {}
        self.file_path_to_dept_map = {}
        # ★スケーラビリティ裏方チェーン導入（バッチ統計や圧縮/集約/高速判定で利用可！）
        self.scalable_backend = ScalableSecurityChain(tune_conf)

    def initialize_department_chain(self, dept_name, normals, exc=None, ano=None):
        print(f"🏢 {dept_name}部門のチェーンを初期化中...")
        chain = SecurityEventChain(metadata={"department": dept_name}, tune_conf=self.conf)
        chain.department = dept_name

        # --- config渡しを徹底 ---
        my_conf = self.conf  # 明示（読みやすさ＆保守性のため）

        # ステート変換＆学習
        normal_states = [security_event_to_state(e, self.security_log, config=my_conf) for e in normals]
        chain.set_normal_model(normal_states, n_clusters=3)
        print(f"  ✅ {len(normal_states)}件の正常パターンを学習")

        if exc:
            exceptional_states = [security_event_to_state(e, self.security_log, config=my_conf) for e in exc]
            chain.exceptional_model = fit_normal_clusters(exceptional_states, n_clusters=2)
            print(f"  📌 {len(exceptional_states)}件の例外パターンを学習")

        if ano:
            anomaly_states = [security_event_to_state(e, self.security_log, config=my_conf) for e in ano]
            chain.anomaly_model = fit_normal_clusters(anomaly_states, n_clusters=2)
            print(f"  🚨 {len(anomaly_states)}件の異常パターンを学習")

        self.department_chains[dept_name] = chain
        self.scalable_backend.minute_aggregates.extend(normal_states)

    def process_event(self, event):
        """イベント処理のメインメソッド"""
        user_id = event.get("user_id", "unknown")
        dept_name = event.get("department") or self.user_department_map.get(user_id, "unknown")

        # 1. 例外パターンチェック
        if self._should_skip_as_exception(event, user_id, dept_name):
            return self._create_exception_result()

        # 2. 未知のユーザー/部署チェック
        if dept_name == "unknown" or dept_name not in self.department_chains:
            return self._create_unknown_result(user_id, dept_name)

        # 3. イベント状態生成
        event_state = security_event_to_state(event, self.security_log, config=self.conf)
        self.scalable_backend.add_event_optimized(event_state)

        # 4. 部署チェーン処理
        result_dept = self._process_department_chain(event, event_state, user_id, dept_name)

        # 5. 個人チェーン処理
        result_user = self._process_user_chain(event, event_state, user_id)

        # 6. 結果統合
        return self._merge_results(result_dept, result_user, user_id, dept_name)

    def _should_skip_as_exception(self, event, user_id, dept_name):
        """例外パターンかどうかの判定"""
        if self.is_exceptional_case(event) and user_id != "unknown" and dept_name != "unknown":
            source_ip = event.get("source_ip", "")
            return source_ip.startswith(PRIVATE_IP_PREFIXES)
        return False

    def _create_exception_result(self):
        """例外パターンの結果作成"""
        return {
            "status": "normal",
            "divergence": 5.0,
            "alert_level": "LOW",
            "reason": "例外パターン吸収"
        }

    def _create_unknown_result(self, user_id, dept_name):
        """未知のユーザー/部署の結果作成"""
        print(f"⚠️ 未知のユーザー/部署: {user_id} / {dept_name}")
        return {
            "status": "suspicious",
            "divergence": 30.0,
            "alert_level": "HIGH",
            "reason": "Unknown user or department"
        }

    def _process_department_chain(self, event, event_state, user_id, dept_name):
        """部署チェーンの処理"""
        dept_chain = self.department_chains[dept_name]
        dept_state = dict(event_state)

        # 部署間アクセスチェック
        is_cross_dept, accessed_dept = self.check_cross_department_access(dept_name, event)
        if is_cross_dept:
            dept_state["threat_context"] = min(dept_state["threat_context"] + 0.3, 1.0)
            dept_state["trust_score"] = max(dept_state["trust_score"] - 0.2, 0.0)

        result_dept = dept_chain.add_block_by_cluster_eval(
            data=f"Event_{user_id}_{event.get('timestamp', 'now')}",
            state_params=dept_state,
            raw_event=event
        )

        if is_cross_dept:
            self.record_cross_department_access(user_id, dept_name, accessed_dept, event)
            result_dept["cross_dept_warning"] = True
            result_dept["accessed_dept"] = accessed_dept

        return result_dept

    def _process_user_chain(self, event, event_state, user_id):
        """個人チェーンの処理"""
        if user_id in self.user_chains:
            user_chain = self.user_chains[user_id]
            user_state = dict(event_state)
            return user_chain.add_block_by_cluster_eval(
                data=f"UserEvent_{user_id}_{event.get('timestamp', 'now')}",
                state_params=user_state,
                raw_event=event
            )
        else:
            deviation_score = self.estimate_user_deviation(user_id, event)
            return {
                "status": "normal",
                "divergence": 0.0,
                "alert_level": "LOW",
                "deviation_score": deviation_score
            }

    def _merge_results(self, result_dept, result_user, user_id, dept_name):
        """部署と個人の結果を統合"""
        merged_status = self.merge_status(
            result_dept["status"],
            result_user["status"],
            result_dept["divergence"],
            result_user["divergence"],
            result_user.get("deviation_score", 0.0),
            user_id in self.user_chains
        )

        merged_divergence = max(result_dept["divergence"], result_user["divergence"])
        merged_alert = "HIGH" if merged_status in ["suspicious", "critical"] else "LOW"

        return {
            "status": merged_status,
            "divergence": merged_divergence,
            "alert_level": merged_alert,
            "dept_result": result_dept,
            "user_result": result_user,
            "has_user_chain": user_id in self.user_chains,
            "cross_dept_access": result_dept.get("cross_dept_warning", False)
        }

    def check_cross_department_access(self, user_dept, event):
        """部署間アクセスを判定（定数を使用）"""
        file_path = event.get("file_path", "")
        destination_ip = event.get("destination_ip", "")

        if file_path:
            path_lower = file_path.lower()

            # 高リスクパスチェック
            for risk_path in HIGH_RISK_PATHS:
                if risk_path in path_lower:
                    return True, "system"

            # 部署パターンチェック（DEPT_PATH_RULESから取得）
            for dept, dept_info in DEPT_PATH_RULES.items():
                if dept != user_dept and "patterns" in dept_info:
                    for pattern in dept_info["patterns"]:
                        if pattern in path_lower:
                            return True, dept

        if destination_ip:
            # IPレンジチェック
            for dept, ip_range in DEPT_IP_RANGES.items():
                if dept != user_dept and destination_ip.startswith(ip_range):
                    return True, dept

        return False, None

    def record_cross_department_access(self, user_id, from_dept, to_dept, event):
        """部署間アクセスを記録"""
        key = f"{user_id}_{from_dept}_{to_dept}"

        if key not in self.cross_department_access:
            self.cross_department_access[key] = {
                "count": 0,
                "first_seen": datetime.now(),
                "last_seen": None,
                "file_paths": [],
                "operations": []
            }

        access_info = self.cross_department_access[key]
        access_info["count"] += 1
        access_info["last_seen"] = datetime.now()

        self._update_access_info(access_info, event)

    def _update_access_info(self, access_info, event):
        """アクセス情報の更新"""
        if "file_path" in event:
            file_path = event["file_path"]
            if file_path not in access_info["file_paths"]:
                access_info["file_paths"].append(file_path)

        if "operation" in event:
            operation = event["operation"]
            if operation not in access_info["operations"]:
                access_info["operations"].append(operation)

    def detect_lateral_movement(self, time_window_minutes=30):
        """横展開攻撃の検知"""
        now = datetime.now()
        user_accesses = self._aggregate_recent_accesses(now, time_window_minutes)
        return self._identify_suspicious_users(user_accesses)

    def _aggregate_recent_accesses(self, now, time_window_minutes):
        """直近のアクセスを集計"""
        user_accesses = {}

        for key, access_info in self.cross_department_access.items():
            # キーの安全な分割
            user_id, from_dept, to_dept = self._parse_access_key(key)
            if not user_id:
                continue

            # 時間窓内のアクセスのみ
            if not self._is_within_time_window(access_info, now, time_window_minutes):
                continue

            # ユーザーアクセス情報の更新
            if user_id not in user_accesses:
                user_accesses[user_id] = {
                    "departments": set(),
                    "file_count": 0,
                    "operations": set()
                }

            self._update_user_access_stats(user_accesses[user_id], from_dept, to_dept, access_info)

        return user_accesses

    def _parse_access_key(self, key):
        """アクセスキーを安全に解析"""
        parts = key.split("_")
        if len(parts) < 3:
            return None, None, None

        user_id = "_".join(parts[:-2])
        from_dept = parts[-2]
        to_dept = parts[-1]
        return user_id, from_dept, to_dept

    def _is_within_time_window(self, access_info, now, window_minutes):
        """アクセスが時間窓内かチェック"""
        if not access_info["last_seen"]:
            return False

        time_diff = (now - access_info["last_seen"]).total_seconds()
        return time_diff < window_minutes * 60

    def _update_user_access_stats(self, user_stats, from_dept, to_dept, access_info):
        """ユーザーアクセス統計の更新"""
        user_stats["departments"].add(from_dept)
        user_stats["departments"].add(to_dept)
        user_stats["file_count"] += len(access_info["file_paths"])
        user_stats["operations"].update(access_info["operations"])

    def _identify_suspicious_users(self, user_accesses):
        """疑わしいユーザーの特定"""
        suspicious_users = []

        for user_id, access_data in user_accesses.items():
            unique_depts = access_data["departments"]
            if len(unique_depts) > 2:
                risk_score = self._calculate_risk_score(unique_depts, access_data["operations"])

                suspicious_users.append({
                    "user_id": user_id,
                    "departments_accessed": list(unique_depts),
                    "file_access_count": access_data["file_count"],
                    "operations": list(access_data["operations"]),
                    "risk_score": risk_score
                })

        return suspicious_users

    def _calculate_risk_score(self, departments, operations):
        """リスクスコアの計算"""
        # 部署数に基づく基本スコア
        base_score = min(len(departments) * 0.3, 1.0)

        # 危険な操作のチェック
        dangerous_ops = ["FileDelete", "FileMove", "FileCopy"]
        if any(op in operations for op in dangerous_ops):
            base_score = min(base_score + 0.2, 1.0)

        return base_score

    def merge_status(self, status_dept, status_user, dept_divergence, user_divergence, deviation_score, has_user_chain):
        """部署とユーザーのステータスを統合"""
        order = ["normal", "investigating", "latent_suspicious", "suspicious", "critical"]

        # 個人チェーンがある場合の補正
        if has_user_chain and deviation_score > 0.7:
            if user_divergence > 15.0:
                return "suspicious"
            elif user_divergence > 10.0:
                return "investigating"

        # 両方suspicious以上なら critical
        dept_idx = order.index(status_dept)
        user_idx = order.index(status_user)

        if dept_idx >= order.index("suspicious") and user_idx >= order.index("suspicious"):
            return "critical"

        # 最も重い判定を採用
        return order[max(dept_idx, user_idx)]

    def is_exceptional_case(self, event):
        """例外パターンの判定"""
        patterns = self.exceptional_patterns.get(event.get("department"), [])

        for pattern in patterns:
            if self._matches_pattern(event, pattern):
                return True

        return False

    def _matches_pattern(self, event, pattern):
        """イベントがパターンに一致するかチェック"""
        match_count = 0
        required_matches = 0

        # 各フィールドのマッチングチェック
        fields_to_check = [
            ("user_id", lambda e, p: e.get("user_id") == p["user_id"]),
            ("file_path", lambda e, p: "file_path" in p and "file_path" in e and p["file_path"] in e["file_path"]),
            ("operation", lambda e, p: e.get("operation") == p["operation"]),
            ("_business_context", lambda e, p: "_business_context" in p and "_business_context" in e and
                                              set(p["_business_context"]).intersection(set(e["_business_context"])))
        ]

        for field, check_func in fields_to_check:
            if field in pattern:
                required_matches += 1
                if check_func(event, pattern):
                    match_count += 1

        # 70%以上マッチしたら該当
        return required_matches > 0 and match_count / required_matches >= 0.7

    def estimate_user_deviation(self, user_id, event):
        """ユーザーの逸脱度を推定"""
        if user_id in self.user_chains:
            return self.user_chains[user_id].calculate_behavior_deviation(event)
        return 0.0

    def get_department_summary(self):
        """部署別サマリーを取得"""
        summary = {}
        for dept_name, chain in self.department_chains.items():
            summary[dept_name] = self._create_dept_summary(chain)
        return summary

    def _create_dept_summary(self, chain):
        """個別部署のサマリー作成"""
        recent_alerts = chain.get_recent_alerts(n=5)
        operation_stats = self._calculate_operation_stats(chain)

        return {
            "total_events": len(chain.blocks) - 1,
            "recent_alerts": len(recent_alerts),
            "alert_details": recent_alerts,
            "chain_valid": chain.verify(verbose=False),
            "operation_stats": operation_stats
        }

    def _calculate_operation_stats(self, chain):
        """操作統計の計算"""
        operation_stats = {}
        for block in chain.blocks[-100:]:
            if "operation" in block.metadata:
                op = block.metadata["operation"]
                operation_stats[op] = operation_stats.get(op, 0) + 1
        return operation_stats

    # ===== スケーラビリティ機能（既存のまま） =====
    def adaptive_sampling_run(self, event):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        sampler = self.scalable_backend.implement_adaptive_sampling()
        priority_score = calc_event_priority(event)  # グローバル関数を使用
        return sampler.should_sample(priority_score)

    def compress_monthly_data(self):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        compressor = self.scalable_backend.implement_time_series_compression()
        data = np.array(self.scalable_backend.day_aggregates)
        if data.size == 0:
            print("⚠️ 圧縮データが空だよ！")
            return None
        return compressor.compress_block(data)

    def aggregate_daily_stats(self):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        data = np.array(self.scalable_backend.hour_aggregates)
        if data.size == 0:
            return {"mean": 0, "std": 0}
        return {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}

    def run_mini_batch_clustering(self, batch_size=500):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        clusters = self.scalable_backend.hierarchical_clustering()
        # --- dict型ならvectorize_stateでベクトル化 ---
        data = list(self.scalable_backend.minute_aggregates)
        if data and isinstance(data[0], dict):
            vectors = np.array([vectorize_state(s) for s in data])
        else:
            vectors = np.array(data)
        if vectors.size == 0:
            raise ValueError("クラスタリング対象データが空です！")
        return clusters.fit_incremental(vectors, batch_size=batch_size)

    def check_bloom_duplicate(self, known_events, new_event):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        bloom = self.scalable_backend.implement_bloom_filter()
        for e in known_events:
            bloom.add(str(e))
        return bloom.contains(str(new_event))

    def parallel_reaggregate(self):
        if not self.conf.enable_scalable_backend:
            raise RuntimeError("ScalableBackend機能はconfigでOFFです！")
        events_by_dept = {
            dept: [block.state_params for block in chain.blocks]
            for dept, chain in self.department_chains.items()
        }
        return self.scalable_backend.parallel_department_processing(events_by_dept)

    def initialize_user_chain(self, user_id, department, historical_events=None):
        """ユーザーチェーンの初期化"""
        print(f"👤 {user_id}のユーザーチェーンを初期化中...")

        # ユーザーチェーンの作成
        user_chain = UserSecurityChain(user_id, department, tune_conf=self.conf)

        # 履歴イベントがある場合はベースライン学習
        if historical_events:
            # 正常イベントのみ抽出してベースライン学習
            normal_events = [e for e in historical_events if e.get("expected", "normal") == "normal"]
            if normal_events:
                user_chain.learn_baseline(normal_events)

                # 正常モデルの学習
                normal_states = [
                    security_event_to_state(e, self.security_log, config=self.conf)
                    for e in normal_events
                ]
                user_chain.set_normal_model(normal_states, n_clusters=3)
                print(f"  ✅ {len(normal_events)}件の正常パターンを学習")

        # ユーザーチェーンを登録
        self.user_chains[user_id] = user_chain
        self.user_department_map[user_id] = department

        return user_chain
# -----------------
# 🔥 レポーティング
# -----------------
    def get_cross_department_report(self, time_window_minutes=30):
        """部署間アクセスレポートの生成"""
        report = {
            "summary": {
                "total_cross_dept_accesses": len(self.cross_department_access),
                "suspicious_users": [],
                "department_matrix": {}
            },
            "details": []
        }

        # 横展開攻撃の検知
        suspicious_users = self.detect_lateral_movement(time_window_minutes)
        report["summary"]["suspicious_users"] = suspicious_users

        # 部署間アクセスマトリックスの作成
        dept_matrix = {}
        for key, access_info in self.cross_department_access.items():
            _, from_dept, to_dept = self._parse_access_key(key)
            if from_dept and to_dept:
                if from_dept not in dept_matrix:
                    dept_matrix[from_dept] = {}
                if to_dept not in dept_matrix[from_dept]:
                    dept_matrix[from_dept][to_dept] = 0
                dept_matrix[from_dept][to_dept] += access_info["count"]

        report["summary"]["department_matrix"] = dept_matrix

        # 詳細情報の追加
        for key, access_info in self.cross_department_access.items():
            user_id, from_dept, to_dept = self._parse_access_key(key)
            if user_id:
                detail = {
                    "user_id": user_id,
                    "from_department": from_dept,
                    "to_department": to_dept,
                    "access_count": access_info["count"],
                    "first_seen": access_info["first_seen"].isoformat() if access_info["first_seen"] else None,
                    "last_seen": access_info["last_seen"].isoformat() if access_info["last_seen"] else None,
                    "unique_files": len(access_info["file_paths"]),
                    "operations": list(set(access_info["operations"]))
                }
                report["details"].append(detail)

        return report

    def analyze_department_patterns(self, department, time_range_hours=24):
        """部署の行動パターン分析"""
        if department not in self.department_chains:
            return {"error": f"Department {department} not found"}

        chain = self.department_chains[department]
        now = datetime.now()
        cutoff_time = now - timedelta(hours=time_range_hours)

        analysis = {
            "department": department,
            "time_range": f"Last {time_range_hours} hours",
            "patterns": {
                "operations": {},
                "file_access": {},
                "time_distribution": {},
                "alert_summary": {
                    "critical": 0,
                    "suspicious": 0,
                    "investigating": 0,
                    "normal": 0
                }
            }
        }

        # 指定時間範囲内のブロックを分析
        for block in chain.blocks:
            if "timestamp" in block.metadata:
                block_time = datetime.strptime(block.metadata["timestamp"], "%Y-%m-%d %H:%M:%S")
                if block_time < cutoff_time:
                    continue

                # 操作統計
                operation = block.metadata.get("operation", "unknown")
                analysis["patterns"]["operations"][operation] = \
                    analysis["patterns"]["operations"].get(operation, 0) + 1

                # ファイルアクセス統計
                if "file_path" in block.metadata:
                    file_path = block.metadata["file_path"]
                    directory = "\\".join(file_path.split("\\")[:-1])
                    analysis["patterns"]["file_access"][directory] = \
                        analysis["patterns"]["file_access"].get(directory, 0) + 1

                # 時間分布
                hour = block_time.hour
                hour_key = f"{hour:02d}:00-{hour:02d}:59"
                analysis["patterns"]["time_distribution"][hour_key] = \
                    analysis["patterns"]["time_distribution"].get(hour_key, 0) + 1

                # アラートサマリー
                mode = block.metadata.get("security_mode", "normal")
                if mode in analysis["patterns"]["alert_summary"]:
                    analysis["patterns"]["alert_summary"][mode] += 1

        return analysis

# ===== 1. build_initial_security_chain関数 =====
def build_initial_security_chain(events_data, security_log, config=None):
    """
    生のセキュリティイベントから初期チェーンを構築
    SecurityEventChain（組織チェーン）のインスタンスを返す
    """
    if config is None:
        config = ChainTuneConfig()

    # 組織チェーンを作成
    chain = SecurityEventChain(tune_conf=config)

    print(f"📊 初期チェーン構築開始: {len(events_data)}件のイベント")

    for i, event in enumerate(events_data):
        # ステート変換
        state = security_event_to_state(event, security_log, config=config)

        # ✨ ヒューリスティック補正
        operation = event.get("operation", "")
        file_path = event.get("file_path", "").lower()
        file_size = event.get("file_size_kb", 0)
        status = event.get("status", "")

        if "FAILED" in status:
            state["normal_div"] *= 1.25
        if "confidential" in file_path:
            state["ano_div"] *= 1.20
        if operation == "FileCopy" and file_size > 100000:
            state["ano_div"] *= 1.3

        # 初期学習フェーズ
        if i == INITIAL_LEARNING_THRESHOLD:
            states = [block.state_params for block in chain.blocks[1:]]
            if len(states) > INITIAL_CLUSTER_COUNT:
                print(f"  🎯 {i}件目で正常モデルの学習開始...")
                chain.set_normal_model(states, n_clusters=3)

        # クラスタモデルが利用可能な場合は add_block_by_cluster_eval を使用
        if chain.cluster_model is not None:
            chain.add_block_by_cluster_eval(
                data=f"Event_{i}",
                state_params=state,
                raw_event=event,
                step=i
            )
        else:
            # モデル学習前はシンプルなブロック追加
            add_block(
                chain,
                data=f"Event_{i}",
                state_params=state,
                divergence=0.0,
                metadata=_create_initial_metadata(event, i)
            )

    return chain

    def _create_initial_metadata(event, index):
        """初期メタデータの作成"""
        metadata = {
            "index": index,
            "timestamp": event.get("timestamp", datetime.now().isoformat()),
            "initial_load": True
        }

        # イベントから主要な情報をコピー
        for key in ["user_id", "department", "operation", "file_path", "source_ip"]:
            if key in event:
                metadata[key] = event[key]

        return metadata

    def add_block_by_cluster_eval(
        self,
        data,
        state_params,
        raw_event=None,
        step=None,
        system="PersonalSecurityMonitor",
        experiment="UserDivergenceCluster"
    ):
        """個人用のクラスタ評価によるブロック追加（特例判定含む）"""
        user_history = self.get_user_history(self.user_id)

        # 1. クラスタ＆divergence
        verdict, score, cluster = self.evaluate_by_multi_cluster(state_params)
        score = self.calculate_context_aware_divergence(score, user_history, raw_event)

        # 2. unified_security_judge で特例判定
        if not hasattr(self, "chain_context"):
            self.chain_context = {"pattern_history": {}, "anomaly_score": 0.0}

        if raw_event:
            # 必要に応じてnormal_div/scoreやtrust_score（予測・既存値）も引数で渡せる
            normal_div = score
            trust_score = user_history.get("trust_score", 0.75)
            verdict2, reason = unified_security_judge(
                raw_event, user_history, self.chain_context, normal_div, trust_score
            )
            if verdict2 != "normal":
                verdict = verdict2
                if verdict == "latent_suspicious":
                    score = max(score, 8.0)

        # 3. 時刻系
        timestamp = raw_event.get("timestamp", raw_event.get("event_time")) if raw_event else datetime.now()
        if isinstance(timestamp, str):
            event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            event_time = timestamp

        time_of_day = self.get_time_of_day(event_time.hour)
        dynamic_threshold = self.get_adaptive_threshold(time_of_day)

        # 4. モード分類
        mode, weighted_score = classify_security_mode_auto(
            divergence=score,
            state_params=state_params,
            previous_ema=self.ema_filter.value,
            config=self.tune_conf.to_user_dict(),
            threshold_override=dynamic_threshold,
            operation=operation
        )

        # 2. unified_security_judge で構造ルール判定
        if raw_event:
            trust_score = user_history.get("trust_score", 0.75)
            verdict2, reason = unified_security_judge(
                raw_event, user_history, self.chain_context,
                normal_div=score,
                trust_score=trust_score
            )
            priority = {
                "normal": 0,
                "investigating": 1,
                "latent_suspicious": 2,
                "suspicious": 3,
                "critical": 4
            }
            if priority.get(verdict2, 0) > priority.get(mode, 0):
                print(f"[OVERRIDE↑] {mode} → {verdict2} by unified_security_judge: {reason}")
                mode = verdict2

        self.ema_filter.update(weighted_score)
        self.update_user_history(self.user_id, event_time, mode)

        if raw_event and "file_path" in raw_event:
            self.update_file_access_pattern(raw_event)

        metadata = {
            "security_mode": mode,
            "weighted_score": float(weighted_score),
            "step": step,
            "system": system,
            "experiment": experiment,
            "cluster_id": cluster,
            "divergence_score": score,
            "verdict": verdict,
            "department": self.department,
            "context_adjusted": True,
            "dynamic_threshold": dynamic_threshold
        }
        if raw_event:
            for k in [
                "event_time", "user_id", "target_resource", "event_type",
                "source_ip", "event_score", "action", "department",
                "timestamp", "operation", "file_path", "file_size_kb",
                "process_name", "destination_ip", "status"
            ]:
                if k in raw_event:
                    metadata[k] = raw_event[k]

        add_block(self, data, state_params, divergence=score, metadata=metadata)

        return {
            "status": mode,
            "divergence": score,
            "added": True,
            "cluster_id": cluster,
            "deviation_score": self.calculate_behavior_deviation(raw_event) if raw_event else 0.0,
            "alert_level": "HIGH" if mode in ["critical", "suspicious", "latent_suspicious"] else "LOW",
            "verdict": verdict,
            "reason": reason if verdict != "normal" else "",
        }

# -----------------
# 🔥 スケーラビリティチェーン
# -----------------
class ScalableSecurityChain:
    """
    スケーラビリティ最適化済みセキュリティチェーン
    - コンフィグ駆動
    - 階層集約/圧縮/並列/Bloom/適応サンプリング内蔵
    """

    def __init__(self, tune_conf):
        self.conf = tune_conf
        self.max_memory_mb = tune_conf.max_memory_mb
        self.cache_size = tune_conf.cache_size

        # 階層的集約
        self.minute_aggregates = deque(maxlen=tune_conf.block_history_min)
        self.hour_aggregates   = deque(maxlen=tune_conf.block_history_hr)
        self.day_aggregates    = deque(maxlen=tune_conf.block_history_day)

        # LRUキャッシュ
        self.divergence_cache = OrderedDict()

        # インデックス
        self.user_index = {}
        self.time_index = {}
        self.anomaly_index = []

        # 圧縮ストレージ
        self.compressed_blocks = []
        self.active_blocks = deque(maxlen=1000)

    def serialize_event(event):
        """event辞書中のnp.ndarrayをlistに変換して返す"""
        def convert(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            elif isinstance(x, dict):
                return {k: convert(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [convert(i) for i in x]
            else:
                return x
        return convert(event)

    def add_event_optimized(self, event):
        """キャッシュ+並列化+バッチ投入"""
        event_str = str(event)
        event_hash = self.compute_event_hash(event)
        if event_hash in self.divergence_cache:
            return self.divergence_cache[event_hash]

        # 非同期バッチ処理を起動
        asyncio.create_task(self._async_tensor_conversion(event))
        asyncio.create_task(self._async_rule_check(event))
        self._add_to_batch_queue(event)
        return event_hash

    @staticmethod
    def compute_event_hash(event):
        """イベントオブジェクトをハッシュ化してユニークID化（ndarray対応）"""
        def convert(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            elif isinstance(x, dict):
                return {k: convert(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [convert(i) for i in x]
            else:
                return x
        event_serializable = convert(event)
        event_str = json.dumps(event_serializable, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(event_str.encode('utf-8')).hexdigest()

    async def _async_tensor_conversion(self, event):
        action_vector = operation_type_vector(event.get("operation", "FileRead"))
        """非同期テンソル変換（例:ベクトル化, PCA等）"""
        tensor = np.empty(6, dtype=np.float32)
        tensor[0] = event.get('event_score', 0) / 100.0
        tensor[1] = self._calculate_severity_vectorized(event)
        tensor[2] = float(event.get("file_size_kb", 0)) / 1e6  # MB単位で正規化
        tensor[3] = 1.0 if event.get("status", "") == "FAILED" else 0.0
        tensor[4] = self._operation_type_vector(event.get("operation", ""))
        tensor[5] = self._risk_path_flag(event.get("file_path", ""))
        return tensor

    def _calculate_severity_vectorized(self, event):
        # 重大度計算ロジックを必要に応じて差し替え
        return event.get("severity_level", 0.0)

    def _add_to_batch_queue(self, event):
        if not hasattr(self, "batch_queue"):
            self.batch_queue = []
        self.batch_queue.append(event)
        # もしバッファがN個溜まったらバッチ処理起動、とかも可能

    # ======================
    # 下層サブ構造もconfig駆動でリファクタ
    # ======================

    def implement_sliding_window(self, window_size=None):
        """統計的スライディングウィンドウ"""
        win_size = window_size or self.conf.cache_size
        class SlidingWindowChain:
            def __init__(self, size):
                self.window = deque(maxlen=size)
                self.mean = np.zeros(6)
                self.std = np.ones(6)
                self.count = 0

            def update(self, new_event):
                old_mean = self.mean.copy()
                self.count += 1
                delta = new_event - old_mean
                self.mean += delta / self.count
                self.std = np.sqrt(
                    (self.std**2 * (self.count-1) + delta * (new_event - self.mean)) / self.count
                )
                self.window.append(new_event)
        return SlidingWindowChain(win_size)

    def hierarchical_clustering(self):
        """MiniBatch階層クラスタリング（スケールアウト用）"""
        class HierarchicalClusters:
            def __init__(self, levels=3, clusters_per_level=10):
                self.levels = levels
                self.clusters_per_level = clusters_per_level
                self.models = {}
            def fit_incremental(self, data_stream, batch_size=1000):
                mbk = MiniBatchKMeans(n_clusters=self.clusters_per_level, batch_size=batch_size, n_init=3)
                for batch in self._batch_generator(data_stream, batch_size):
                    mbk.partial_fit(batch)
                return mbk
            def _batch_generator(self, data, batch_size):
                batch = []
                for item in data:
                    batch.append(item)
                    if len(batch) >= batch_size:
                        yield np.array(batch)
                        batch = []
                if batch:
                    yield np.array(batch)
        return HierarchicalClusters()

    def implement_bloom_filter(self, expected_items=1_000_000, fp_rate=0.01):
        """ブルームフィルタ（config連動）"""
        size = int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2))
        hash_count = int((size / expected_items) * math.log(2))
        class BloomFilter:
            def __init__(self, size, hash_count):
                self.size = size
                self.hash_count = hash_count
                self.bit_array = np.zeros(size, dtype=bool)
            def add(self, item):
                for seed in range(self.hash_count):
                    index = self._hash(item, seed) % self.size
                    self.bit_array[index] = True
            def contains(self, item):
                for seed in range(self.hash_count):
                    index = self._hash(item, seed) % self.size
                    if not self.bit_array[index]:
                        return False
                return True
            def _hash(self, item, seed):
                h = hashlib.md5(f"{item}{seed}".encode()).digest()
                return int.from_bytes(h[:4], 'big')
        return BloomFilter(size, hash_count)

    def implement_time_series_compression(self):
        """Δエンコーディング+gzip時系列圧縮"""
        class TimeSeriesCompressor:
            def __init__(self):
                self.compression_ratio = 0.0
            def compress_block(self, block_data):
                if len(block_data) > 1:
                    deltas = np.diff(block_data, axis=0)
                    base = block_data[0]
                else:
                    deltas = np.array([])
                    base = block_data[0] if len(block_data) > 0 else None
                quantized = (deltas * 1000).astype(np.int16) if len(deltas) > 0 else np.array([], dtype=np.int16)
                compressed = gzip.compress(pickle.dumps({'base': base, 'deltas': quantized, 'shape': block_data.shape}))
                original_size = block_data.nbytes
                compressed_size = len(compressed)
                self.compression_ratio = 1 - (compressed_size / original_size)
                return compressed
            def decompress_block(self, compressed_data):
                data = pickle.loads(gzip.decompress(compressed_data))
                if len(data['deltas']) > 0:
                    deltas = data['deltas'].astype(np.float64) / 1000
                    reconstructed = np.cumsum(np.vstack([data['base'], deltas]), axis=0)
                else:
                    reconstructed = np.array([data['base']])
                return reconstructed
        return TimeSeriesCompressor()

    def implement_adaptive_sampling(self):
        """高負荷時の適応サンプリング（パラメータconfig連動）"""
        tr = self.conf.target_sample_rate
        hl = self.conf.high_load_thresh
        class AdaptiveSampler:
            def __init__(self):
                self.target_rate   = tr
                self.high_load_th  = hl
                self.current_load = 0.0
                self.sample_probability = 1.0
            def should_sample(self, priority_score):
                if priority_score > 0.8:
                    return True
                if self.current_load > self.high_load_th:
                    self.sample_probability = self.target_rate
                else:
                    self.sample_probability = min(1.0, self.target_rate + (1 - self.current_load))
                return np.random.random() < self.sample_probability
            def update_load(self, processing_time, threshold=0.001):
                self.current_load = min(1.0, processing_time / threshold)
        return AdaptiveSampler()

    # 仮コード本番での希望にあわせて
    def parallel_department_processing(self, events_by_dept):
        def aggregate(events):
            arr = np.array([vectorize_state(ev) if isinstance(ev, dict) else ev for ev in events])
            return {"mean": arr.mean(axis=0), "std": arr.std(axis=0)} if arr.size > 0 else {"mean": 0, "std": 0}
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {dept: executor.submit(aggregate, events) for dept, events in events_by_dept.items()}
            for dept, future in futures.items():
                results[dept] = future.result()
        return results

    async def _async_rule_check(self, event):
        # ---------------------------------------------------------------
        # 🚦【ルールベース判定ロジック】
        # ---------------------------------------------------------------
        #
        # 目的：
        #   ・現場の「こうなったら絶対アウト/警戒！」という明示的な基準をシステム化する。
        #   ・AIクラスタや機械学習モデルでは気づきにくい、超具体的な業務ルールや例外パターン、
        #     法規制、ブラックリスト、突発的な運用ルールなどの「絶対条件」を吸収する。
        #
        # 代表例：
        #   - 夜間のファイル削除・管理フォルダ操作は常にアラート
        #   - ブラックリストIP・ユーザー・ファイルパスへのアクセス
        #   - 年度末・決算時期など特定時期のみ許される操作（例外吸収）
        #   - 急なパスワード連続失敗やログ削除など“誰が見ても危険”な動作
        #
        # 【AI判定との違い】
        #   - AIクラスタ判定:「普段の正常パターンからの逸脱」を“数値的な距離”でスコア化し、最も近いクラスタ（normal/exceptional/anomaly）を分類する。
        #   - ルールベース:「絶対アウト」「現場の常識」など、“1発判定”で決める。AIでは拾えない「人間ならではの運用ノウハウ」を吸収！
        #   - 判定競合時は「危険側（criticalなど）」を優先してmodeを上書きするのが多い
        #
        # 【実装目的まとめ】
        #   - 運用現場のブラックボックス的な知見・暫定運用ルールをすぐ反映できる
        #   - AI判定を補助し、誤検知や見逃しのリスクを減らす
        #   - 法令遵守や現場独自の強制ルールに柔軟対応
        # ---------------------------------------------------------------
        await asyncio.sleep(0)  # 何もしないで即return
        return None

# -----------------
# 🔥 実行関数
# -----------------
if __name__ == "__main__":
    print("=== 🛡️ Security Divergence Chain System (LanScope Cat Edition) ===\n")

    # 設定クラス（ChainTuneConfig）をロード（必要に応じて外部YAML/ENVからもOK）
    tune_conf = ChainTuneConfig()

    # 2. 部署別チェーンマネージャー初期化
    manager = DepartmentSecurityChainManager(tune_conf=tune_conf)

    # 3. 正常ログの読み込み
    with open(tune_conf.security_chain_filepath, "r", encoding="utf-8") as f:
        normal_events = json.load(f)
    print(f"📁 {len(normal_events)}件の学習用ログを読み込みました")

    # 部署ごとにイベントを分類＆操作統計
    dept_events, operation_stats = {}, {}
    for event in normal_events:
        dept = event.get("department", "unknown")
        dept_events.setdefault(dept, []).append(event)
        operation = event.get("operation", "unknown")
        operation_stats[operation] = operation_stats.get(operation, 0) + 1

    print("\n📊 操作タイプ別統計:")
    for op, count in sorted(operation_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {count}件")

    # 各部署のチェーンを初期化
    for dept_name, events in dept_events.items():
        if dept_name != "unknown":
            manager.initialize_department_chain(dept_name, events)

    # 時間帯別追加学習
    print("\n📚 時間帯別正常パターンを追加学習中...")
    for dept_name, chain in manager.department_chains.items():
        my_chain_conf = manager.conf
        patterns = []
        for gen_func in (generate_morning_patterns, generate_noon_patterns, generate_evening_patterns):
            patterns += gen_func(dept_name)
        additional_states = [
            security_event_to_state(p, manager.security_log, config=my_chain_conf) for p in patterns
        ]
        if additional_states:
            chain.update_normal_model(additional_states)
            print(f"  ✅ {dept_name}: {len(additional_states)}件の時間帯別パターンを追加")

    # ---★ (3) operation × process_name の組み合わせで分割で正常モデル再学習 ---
    print("\n🔁 操作・プロセス別で正常モデル再学習...")
    for dept_name, events in dept_events.items():
        op_proc_states = defaultdict(list)
        for e in events:
            # 通常の正常イベントのみ
            if e.get("expected", "normal") == "normal":
                op = e.get('operation', '').lower()
                proc = e.get('process_name', '').lower()
                key = (op, proc)
                state = security_event_to_state(e, manager.security_log, config=manager.conf)
                op_proc_states[key].append(state)
        # 全クラスタを合体
        all_states = []
        for states in op_proc_states.values():
            all_states.extend(states)
        if all_states:
            manager.department_chains[dept_name].set_normal_model(all_states, n_clusters=3)
            print(f"  ✅ {dept_name}: {len(all_states)}件の全正常操作（operation×process_name単位で統合）で再学習！")

    print(f"\n✅ {len(manager.department_chains)}個の部署チェーンを初期化完了")

    # === 例外パターンの登録 ===
    print("\n🎯 特殊ケースを例外パターンとして登録中...")
    exceptional_patterns = {}
    for dept_name in manager.department_chains:
        year_end_patterns = generate_year_end_patterns(dept_name)
        if year_end_patterns:
            exceptional_patterns[dept_name] = year_end_patterns
            print(f"  ✅ {dept_name}: {len(year_end_patterns)}件の年度末協力パターンを例外リストに追加")
    manager.exceptional_patterns = exceptional_patterns
    print(f"\n  合計 {sum(len(v) for v in exceptional_patterns.values())}件の部署間協力パターンを例外管理！")

    print("\n🚨 異常クラスタ（anomaly_cluster.json）を読み込み中...")
    try:
        with open(tune_conf.error_chain_filepath, "r", encoding="utf-8") as f:
            anomaly_clusters = json.load(f)
        print(f"  📁 {len(anomaly_clusters)}件の異常クラスタデータを検出")

        dept_event_map = {}

        for item in anomaly_clusters:
            # パターンA: {"department": "xxx", "events": [...]}
            if isinstance(item, dict) and "department" in item and "events" in item:
                dept = item["department"]
                dept_event_map.setdefault(dept, []).extend(item["events"])
            # パターンB: 直接イベント(dict)の場合
            elif isinstance(item, dict) and "department" in item:
                dept = item["department"]
                dept_event_map.setdefault(dept, []).append(item)
            # パターンC: イベントリスト（departmentキーなし or unknown）
            else:
                dept_event_map.setdefault("unknown", []).append(item)

        # 部署ごとに学習モデル投入
        for dept, events in dept_event_map.items():
            if dept in manager.department_chains:
                abnormal_states = [
                    security_event_to_state(e, manager.security_log, config=my_chain_conf)
                    for e in events
                ]
                if abnormal_states:
                    manager.department_chains[dept].update_anomaly_model(abnormal_states)
                    print(f"  🚨 {dept}: {len(abnormal_states)}件の異常クラスタを追加")
            else:
                print(f"  ⚠️ 部署未登録: {dept}（スキップ）")
    except Exception as e:
        print(f"  ⚠️ 異常クラスタの読込失敗: {e}")

    print(f"\n✅ すべての正常・例外・異常クラスタの初期化が完了しました！")

    # === 🌟【スケーラビリティAPI呼び出し】 ===
    if manager.conf.enable_scalable_backend:
        print("\n🚀 ScalableBackendバッチAPI呼び出しデモ")
        try:
            manager.compress_monthly_data()
            print("  ✅ 圧縮完了")
            stats = manager.aggregate_daily_stats()
            print(f"  ✅ 日次統計: {stats}")
            labels = manager.run_mini_batch_clustering().labels_
            print(f"  ✅ ミニバッチクラスタリングlabels: {labels[:5]} ...")
            is_dup = manager.check_bloom_duplicate(normal_events[:100], normal_events[0])
            print(f"  ✅ Bloom重複判定: {is_dup}")
            par_results = manager.parallel_reaggregate()
            print(f"  ✅ 並列再集計: 部署数={len(par_results)}")
            sample_event = normal_events[0]
            if manager.adaptive_sampling_run(sample_event):
                print("  ✅ 適応サンプリング本処理可！")
            else:
                print("  ℹ️ サンプリングでスキップ")
        except Exception as ex:
            print(f"  ⚠️ ScalableBackend API呼び出しエラー: {ex}")
    else:
        print("\n（ScalableBackendはconfigでOFFです）")

    # 4. 正常アクセスの検証（誤検知チェック）
    print("\n=== ✅ 正常アクセスパターン検証（誤検知チェック） ===")
    with open(tune_conf.normal_test_events_file, "r", encoding="utf-8") as f:
        normal_test_events = json.load(f)

    false_positive_count = 0
    false_positive_details = []

    for i, event in enumerate(normal_test_events):
        result = manager.process_event(event)

        # 結果の表示
        icon = "🔴" if result["alert_level"] == "HIGH" else "🟡" if result["status"] == "investigating" else "🟢"

        is_false_positive = False
        if result["alert_level"] == "HIGH" or result["status"] in ["suspicious", "critical"]:
            false_positive_count += 1
            is_false_positive = True
            icon = "❌ FALSE POSITIVE"

            false_positive_details.append({
                "event_id": i,
                "user": event['user_id'],
                "operation": event.get('operation', 'unknown'),
                "file_path": event.get('file_path', 'N/A'),
                "dept_status": result.get('dept_result', {}).get('status', 'N/A'),
                "dept_divergence": result.get('dept_result', {}).get('divergence', 0),
                "user_status": result.get('user_result', {}).get('status', 'N/A'),
                "user_divergence": result.get('user_result', {}).get('divergence', 0),
                "has_user_chain": result.get('has_user_chain', False)
            })

        # 基本情報の表示
        print(f"{icon} Normal {i}: {event.get('operation', 'unknown')}")
        print(f"   User: {event['user_id']} | Dept: {event['department']}")
        if 'file_path' in event:
            print(f"   File: {event['file_path']}")
        print(f"   Status: {result['status']} | Divergence: {result['divergence']:.4f}")

        if is_false_positive:
            print("   📊 詳細分析:")
            print(f"      部署判定: {result['dept_result']['status']} (Div: {result['dept_result']['divergence']:.2f})")
            if result.get('has_user_chain', False):
                print(f"      個人判定: {result['user_result']['status']} (Div: {result['user_result']['divergence']:.2f})")
        print()

    # 5. 混合シナリオテスト（最もリアル）
    with open(tune_conf.suspicious_events_file, "r", encoding="utf-8") as f:
        mixed_events = json.load(f)

    event_results = []
    correct_detections = 0

    for i, event in enumerate(mixed_events):
        result = manager.process_event(event)

        expected = event.get("expected", "unknown")
        actual = result["status"]

        def is_equivalent(expected, actual):
            normal_equiv = ["normal", "investigating"]
            suspicious_equiv = ["suspicious", "critical"]
            if expected in normal_equiv and actual in normal_equiv:
                return True
            if expected in suspicious_equiv and actual in suspicious_equiv:
                return True
            return expected == actual

        is_correct = is_equivalent(expected, actual)
        if is_correct:
            correct_detections += 1

        # 個人チェーンの逸脱度（無い場合None）
        user_chain = manager.user_chains.get(event.get("user_id", "unknown"))
        deviation_score = user_chain.calculate_behavior_deviation(event) if user_chain else None

        event_results.append({
            "No": i + 1,
            "Time": event.get('timestamp', '').split()[1] if "timestamp" in event else event.get('event_time', '').split()[1],
            "User": event['user_id'],
            "Dept": event.get('department', ""),
            "Operation": event.get('operation', event.get('action', '')),
            "Expected": expected,
            "Actual": actual,
            "Correct": is_correct,
            "Divergence": result["divergence"],
            "AlertLevel": result["alert_level"],
            "DeviationScore": result.get('user_result', {}).get('deviation_score', None),
            "UserStatus": result.get('user_result', {}).get('status', 'N/A'),
            "UserDivergence": result.get('user_result', {}).get('divergence', 0.0),
            "DeptStatus": result.get('dept_result', {}).get('status', 'N/A'),
            "DeptDivergence": result.get('dept_result', {}).get('divergence', 0.0),
            "CrossDeptWarning": result.get("cross_dept_warning", False)
        })

        # --- ログ出力をセーブ ---
        if i < 70:  # 先頭10件だけサマリ出力
            print(f"{'✅' if is_correct else '❌'} Event {i+1}: {event.get('operation', '')} | Exp={expected} | Act={actual}")

    # DataFrameで集計
    df_results = pd.DataFrame(event_results)
    print(f"\n=== 🎭 混合シナリオテスト結果（先頭60件のみ表示/全{len(mixed_events)}件） ===")
    print(df_results.head(60).to_string(index=False))  # 先頭10件だけ

    # 総合判定精度
    accuracy = correct_detections / len(mixed_events) * 100
    print(f"\n🎯 総合シナリオ判定精度: {accuracy:.1f}% ({correct_detections}/{len(mixed_events)})")

    # 必要ならCSV保存
    df_results.to_csv("scaling_security_logs.csv", index=False)

    # 6. ゼロデイ攻撃検証とサマリー
    print("\n=== 🚨 ゼロデイ攻撃シミュレーション ===")
    with open(tune_conf.zero_day_events_file, "r", encoding="utf-8") as f:
        zero_day_events = json.load(f)

    detected_attacks = 0
    zero_day_results = []
    for i, event in enumerate(zero_day_events):
        result = manager.process_event(event)
        zero_day_results.append({**event, **result})
        detected = result["alert_level"] == "HIGH" or result["status"] in ["suspicious", "critical"]
        if detected:
            detected_attacks += 1
        icon = (
            "🔴" if detected else
            "🟡" if result["status"] == "investigating" else
            "❌ MISSED ATTACK"
        )
        cross_dept = "⚠️ CROSS-DEPT" if result.get("cross_dept_warning") else ""
        print(f"{icon} Event {i+1}: {event.get('operation', event.get('action', 'N/A'))}")
        print(f"   User: {event.get('user_id', 'unknown')} | Target: {event.get('target_resource')}")
        print(f"   Status: {result['status']} | Divergence: {result['divergence']:.4f} {cross_dept}")

        # 追加：クラスタ距離や詳細
        if "dept_result" in result:
            dept = result["dept_result"]
            print(f"     [DEBUG] Dept: status={dept.get('status')}, divergence={dept.get('divergence')}, cluster={dept.get('cluster_id')}")
        if "user_result" in result:
            user = result["user_result"]
            print(f"     [DEBUG] User: status={user.get('status')}, divergence={user.get('divergence')}, cluster={user.get('cluster_id')}")
        print("")  # 空行で見やすく

    # 7. 攻撃パターン検出 & 時系列
    with open(tune_conf.attack_patterns_file, "r", encoding="utf-8") as f:
        attack_patterns = json.load(f)
    with open(tune_conf.suspicious_events_file, "r", encoding="utf-8") as f:
        suspicious_events = json.load(f)

    attack_pattern_matches = {k: [] for k in attack_patterns}

    # 重要なイベントだけ先頭300件程度サンプリング可
    for i, event in enumerate(suspicious_events[:300]):
        result = manager.process_event(event)
        event["divergence"] = result.get("divergence", 0)
        status = result["status"]
        action = event.get("operation", event.get("action", ""))

        for p_type, p_info in attack_patterns.items():
            if any(ind in action for ind in p_info["indicators"]) and status in ["suspicious", "critical"]:
                attack_pattern_matches[p_type].append(i + 1)

    # パターンごとのサマリ出力
    for pattern_type, indices in attack_pattern_matches.items():
        if indices:
            p = attack_patterns[pattern_type]
            print(f"\n【{p['name']}】{p['description']}")
            print(f"  検出イベント番号: {', '.join(map(str, indices[:10]))} ... (全{len(indices)}件)")
            print(f"  指標: {', '.join(p['indicators'])}")

    # 攻撃進行タイムライン（最大10件だけ）
    attack_timeline = [
        {
            "time": e.get("timestamp", e.get("event_time", "")).split()[1] if "timestamp" in e or "event_time" in e else "",
            "user": e.get("user_id", "unknown"),
            "operation": e.get("operation", e.get("action", "")),
            "severity": e.get("expected", "")
        }
        for e in suspicious_events[:300] if e.get("expected") in ["suspicious", "critical"]
    ]

    if attack_timeline:
        print("\n⏰ 攻撃進行タイムライン（最大10件表示）:")
        print("  時刻     | ユーザー    | アクション                    | 深刻度")
        print("  " + "-" * 65)
        for entry in attack_timeline[:20]:
            print(f"  {entry['time']} | {entry['user']:<10} | {entry['operation']:<28} | {entry['severity']}")

    # 8. 異常行動ユーザー分析（Divergence降順Top10）
    print("\n👤 異常行動ユーザー分析（Divergenceスコアベース）:")
    top_anomalies = sorted(
        suspicious_events,
        key=lambda e: e.get("divergence", 0),
        reverse=True
    )[:10]
    for i, e in enumerate(top_anomalies):
        # 時刻とアクションの安全な取得
        t = e.get("timestamp", e.get("event_time", "")).split()[1] if "timestamp" in e or "event_time" in e else ""
        act = e.get("scenario", e.get("operation", e.get("action", "")))
        print(f"  {i+1}. {t} | User: {e.get('user_id', 'unknown')} | operation: {act} | Divergence: {e.get('divergence', 0):.1f}")
    if len(suspicious_events) > 10:
        print(f"  ... 他{len(suspicious_events)-10}件の異常行動あり")

    # 9. 精度サマリー
    print("\n=== 📈 システム検知精度サマリー ===")
    print(f"正常アクセス判定: {len(normal_test_events)}件")
    print(f"  └─ 正常判定: {len(normal_test_events) - false_positive_count}件")
    print(f"  └─ 誤検知: {false_positive_count}件")
    print(f"攻撃イベント: {len(zero_day_events)}件")
    print(f"  └─ 検知成功: {detected_attacks}件")
    print(f"  └─ 見逃し: {len(zero_day_events) - detected_attacks}件")
    precision = detected_attacks / (detected_attacks + false_positive_count) if (detected_attacks + false_positive_count) > 0 else 0
    recall = detected_attacks / len(zero_day_events) if len(zero_day_events) > 0 else 0
    print(f"\n  精度（Precision）: {precision:.2%}")
    print(f"  再現率（Recall）: {recall:.2%}")

    # 10. 横展開攻撃検知
    print("\n=== 🔄 横展開攻撃の検知 ===")
    lateral_movements = manager.detect_lateral_movement(time_window_minutes=180)
    if lateral_movements:
        print("⚠️ 複数部署のリソースに同時アクセスしているユーザー:")
        for user_info in lateral_movements:
            print(f"   User: {user_info['user_id']}")
            print(f"   Departments: {', '.join(user_info['departments_accessed'])}")
            print(f"   Risk Score: {user_info['risk_score']:.2f}\n")
    else:
        print("✅ 横展開攻撃の兆候なし")

    # 11. ⚡ 部署横断 同期アラート
    print("\n=== ⚡ 部署横断 同期アラート ===")
    cross_dept_alert_sync(manager, window_minutes=180)

    # 12. チェーンエクスポート
    print("\n=== 💾 チェーンエクスポート ===")
    for dept_name, chain in manager.department_chains.items():
        filename = f"security_chain_{dept_name}.json"
        chain.export_to_json(filename)
        print(f"✅ {dept_name}部門のチェーンを {filename} に保存")

    print("\n✅ セキュリティ分析 完了！")

    # ダッシュボードで可視化
    report = manager.get_cross_department_report(time_window_minutes=60*24)
    with open("cross_dept_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 部署ごとの行動パターンをダッシュボードで可視化
    for dept in manager.department_chains:
        analysis = manager.analyze_department_patterns(dept, time_range_hours=72)
        print(json.dumps(analysis, ensure_ascii=False, indent=2))

    # --- ここで混合ログや攻撃パターンなどを指定 ---
    with open(tune_conf.suspicious_events_file, "r", encoding="utf-8") as f:
        all_bench_events = json.load(f)

    # 必要に応じてサンプリング or 全件投入
    BENCHMARK_SIZE = 60  # 必要ならlen(all_bench_events)とかでも
    benchmark_events = random.sample(all_bench_events, BENCHMARK_SIZE) if len(all_bench_events) > BENCHMARK_SIZE else all_bench_events

    start = time.perf_counter()

    for event in benchmark_events:
        result = manager.process_event(event)

    elapsed = time.perf_counter() - start
    throughput = len(benchmark_events) / elapsed if elapsed > 0 else 0

    print(f"\n=== ⏱️ 処理性能ベンチマーク ===")
    print(f"イベント数: {len(benchmark_events)}")
    print(f"処理時間: {elapsed:.3f} 秒")
    print(f"スループット: {throughput:.1f} events/sec")
