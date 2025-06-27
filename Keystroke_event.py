# =====================================
# キーストロークダイナミクス認証モジュール
# =====================================

import numpy as np
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import asyncio
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

# Numba JITインポート（高速化用）
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Numbaが使えない場合のダミー関数
    def njit(func):
        return func
    prange = range

# =====================================
# データ構造定義
# =====================================

@dataclass
class KeystrokeEvent:
    """キーストロークイベント"""
    timestamp: float  # ミリ秒精度のタイムスタンプ
    key_code: str     # キーコード（ハッシュ化可能）
    event_type: str   # 'down' or 'up'
    session_id: str
    user_id: str
    
@dataclass
class KeystrokeDynamics:
    """キーストロークダイナミクス特徴量"""
    dwell_times: List[float] = field(default_factory=list)      # キー押下時間
    flight_times: List[float] = field(default_factory=list)     # キー間移動時間
    inter_key_intervals: List[float] = field(default_factory=list)  # キー間隔
    typing_speed: float = 0.0                                   # タイピング速度
    rhythm_consistency: float = 0.0                             # リズムの一貫性
    pressure_variance: float = 0.0                              # 圧力の分散（利用可能な場合）
    
@dataclass
class UserKeystrokeProfile:
    """ユーザーのキーストロークプロファイル"""
    user_id: str
    sample_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # 統計的特徴
    mean_dwell_time: float = 0.0
    std_dwell_time: float = 0.0
    mean_flight_time: float = 0.0
    std_flight_time: float = 0.0
    mean_typing_speed: float = 0.0
    std_typing_speed: float = 0.0
    
    # 詳細な特徴分布
    dwell_time_distribution: Dict[str, float] = field(default_factory=dict)
    flight_time_distribution: Dict[str, float] = field(default_factory=dict)
    common_digraphs: Dict[str, float] = field(default_factory=dict)  # よく使う2文字組
    common_trigraphs: Dict[str, float] = field(default_factory=dict)  # よく使う3文字組
    
    # 機械学習モデル
    ml_model: Any = None
    feature_scaler: Any = None

# =====================================
# 高速化された特徴量計算関数
# =====================================

@njit
def calculate_dwell_times_fast(timestamps: np.ndarray, event_types: np.ndarray) -> np.ndarray:
    """高速なDwell Time計算"""
    dwell_times = []
    i = 0
    while i < len(timestamps) - 1:
        if event_types[i] == 0 and event_types[i+1] == 1:  # down -> up
            dwell_time = timestamps[i+1] - timestamps[i]
            if 0 < dwell_time < 1000:  # 妥当な範囲（0-1秒）
                dwell_times.append(dwell_time)
            i += 2
        else:
            i += 1
    return np.array(dwell_times)

@njit
def calculate_flight_times_fast(timestamps: np.ndarray, event_types: np.ndarray) -> np.ndarray:
    """高速なFlight Time計算"""
    flight_times = []
    last_up = -1
    
    for i in range(len(timestamps)):
        if event_types[i] == 1:  # up event
            last_up = i
        elif event_types[i] == 0 and last_up >= 0:  # down event
            flight_time = timestamps[i] - timestamps[last_up]
            if 0 < flight_time < 5000:  # 妥当な範囲（0-5秒）
                flight_times.append(flight_time)
    
    return np.array(flight_times)

# =====================================
# キーストローク収集システム
# =====================================

class KeystrokeCollector:
    """キーストローク収集システム"""
    
    def __init__(self, privacy_mode: bool = True):
        self.privacy_mode = privacy_mode
        self.event_buffer = deque(maxlen=10000)
        self.session_data = defaultdict(list)
        self.is_collecting = False
        self.collection_thread = None
        
        # プライバシー保護用のハッシュソルト
        if privacy_mode:
            self.hash_salt = hashlib.sha256(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()
    
    def start_collection(self, user_id: str, session_id: str):
        """収集開始"""
        self.is_collecting = True
        self.current_user = user_id
        self.current_session = session_id
        
        # 実際の実装では、OSレベルのキーボードフックを使用
        # ここではシミュレーション
        self.collection_thread = threading.Thread(
            target=self._simulate_keystroke_collection
        )
        self.collection_thread.start()
    
    def stop_collection(self):
        """収集停止"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
    
    def _simulate_keystroke_collection(self):
        """キーストローク収集のシミュレーション"""
        # 実際の実装では、pynput、keyboard、またはOS固有のAPIを使用
        import random
        
        while self.is_collecting:
            # ランダムなキーストロークイベントを生成
            key_code = f"key_{random.randint(65, 90)}"  # A-Z
            
            # Downイベント
            down_event = KeystrokeEvent(
                timestamp=datetime.now().timestamp() * 1000,
                key_code=self._hash_key(key_code) if self.privacy_mode else key_code,
                event_type='down',
                session_id=self.current_session,
                user_id=self.current_user
            )
            self.event_buffer.append(down_event)
            
            # Dwell time (キー押下時間)
            dwell = random.gauss(100, 30)  # 平均100ms、標準偏差30ms
            if dwell > 0:
                threading.Event().wait(dwell / 1000)
            
            # Upイベント
            up_event = KeystrokeEvent(
                timestamp=datetime.now().timestamp() * 1000,
                key_code=self._hash_key(key_code) if self.privacy_mode else key_code,
                event_type='up',
                session_id=self.current_session,
                user_id=self.current_user
            )
            self.event_buffer.append(up_event)
            
            # Flight time (次のキーまでの時間)
            flight = random.gauss(150, 50)  # 平均150ms、標準偏差50ms
            if flight > 0:
                threading.Event().wait(flight / 1000)
    
    def _hash_key(self, key_code: str) -> str:
        """キーコードのハッシュ化（プライバシー保護）"""
        return hashlib.sha256(
            f"{key_code}{self.hash_salt}".encode()
        ).hexdigest()[:8]
    
    def get_session_events(self, session_id: str) -> List[KeystrokeEvent]:
        """セッションのイベント取得"""
        return [e for e in self.event_buffer if e.session_id == session_id]

# =====================================
# 特徴量抽出器
# =====================================

class KeystrokeFeatureExtractor:
    """キーストローク特徴量抽出器"""
    
    def __init__(self):
        self.min_events = 20  # 最小イベント数
    
    def extract_features(self, events: List[KeystrokeEvent]) -> Optional[KeystrokeDynamics]:
        """イベントリストから特徴量を抽出"""
        if len(events) < self.min_events:
            return None
        
        # イベントをソート
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # NumPy配列に変換
        timestamps = np.array([e.timestamp for e in sorted_events])
        event_types = np.array([0 if e.event_type == 'down' else 1 for e in sorted_events])
        
        # 特徴量計算
        if NUMBA_AVAILABLE:
            dwell_times = calculate_dwell_times_fast(timestamps, event_types)
            flight_times = calculate_flight_times_fast(timestamps, event_types)
        else:
            dwell_times = self._calculate_dwell_times(sorted_events)
            flight_times = self._calculate_flight_times(sorted_events)
        
        # その他の特徴量
        inter_key_intervals = self._calculate_inter_key_intervals(sorted_events)
        typing_speed = self._calculate_typing_speed(sorted_events)
        rhythm_consistency = self._calculate_rhythm_consistency(inter_key_intervals)
        
        return KeystrokeDynamics(
            dwell_times=dwell_times.tolist() if isinstance(dwell_times, np.ndarray) else dwell_times,
            flight_times=flight_times.tolist() if isinstance(flight_times, np.ndarray) else flight_times,
            inter_key_intervals=inter_key_intervals,
            typing_speed=typing_speed,
            rhythm_consistency=rhythm_consistency
        )
    
    def _calculate_dwell_times(self, events: List[KeystrokeEvent]) -> List[float]:
        """Dwell Time計算（Numba無し版）"""
        dwell_times = []
        i = 0
        while i < len(events) - 1:
            if events[i].event_type == 'down' and events[i+1].event_type == 'up':
                if events[i].key_code == events[i+1].key_code:
                    dwell_time = events[i+1].timestamp - events[i].timestamp
                    if 0 < dwell_time < 1000:  # 妥当な範囲
                        dwell_times.append(dwell_time)
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return dwell_times
    
    def _calculate_flight_times(self, events: List[KeystrokeEvent]) -> List[float]:
        """Flight Time計算（Numba無し版）"""
        flight_times = []
        last_up_event = None
        
        for event in events:
            if event.event_type == 'up':
                last_up_event = event
            elif event.event_type == 'down' and last_up_event:
                flight_time = event.timestamp - last_up_event.timestamp
                if 0 < flight_time < 5000:  # 妥当な範囲
                    flight_times.append(flight_time)
        
        return flight_times
    
    def _calculate_inter_key_intervals(self, events: List[KeystrokeEvent]) -> List[float]:
        """キー間隔の計算"""
        intervals = []
        down_events = [e for e in events if e.event_type == 'down']
        
        for i in range(1, len(down_events)):
            interval = down_events[i].timestamp - down_events[i-1].timestamp
            if 0 < interval < 5000:
                intervals.append(interval)
        
        return intervals
    
    def _calculate_typing_speed(self, events: List[KeystrokeEvent]) -> float:
        """タイピング速度の計算（キー/分）"""
        if len(events) < 2:
            return 0.0
        
        down_events = [e for e in events if e.event_type == 'down']
        if len(down_events) < 2:
            return 0.0
        
        time_span = (down_events[-1].timestamp - down_events[0].timestamp) / 1000 / 60  # 分
        if time_span > 0:
            return len(down_events) / time_span
        return 0.0
    
    def _calculate_rhythm_consistency(self, intervals: List[float]) -> float:
        """リズムの一貫性を計算（低いほど一貫性が高い）"""
        if len(intervals) < 2:
            return 0.0
        
        return np.std(intervals) / (np.mean(intervals) + 1e-6)
    
    def extract_advanced_features(self, dynamics: KeystrokeDynamics) -> np.ndarray:
        """機械学習用の高度な特徴量を抽出"""
        features = []
        
        # 基本統計量
        for times in [dynamics.dwell_times, dynamics.flight_times, dynamics.inter_key_intervals]:
            if times:
                features.extend([
                    np.mean(times),
                    np.std(times),
                    np.median(times),
                    np.percentile(times, 25),
                    np.percentile(times, 75),
                    stats.skew(times),
                    stats.kurtosis(times)
                ])
            else:
                features.extend([0.0] * 7)
        
        # その他の特徴
        features.append(dynamics.typing_speed)
        features.append(dynamics.rhythm_consistency)
        
        # n-gram時間パターン（連続するキーの時間間隔の統計）
        if len(dynamics.inter_key_intervals) >= 3:
            bigram_intervals = []
            for i in range(len(dynamics.inter_key_intervals) - 1):
                bigram_intervals.append(
                    dynamics.inter_key_intervals[i+1] / (dynamics.inter_key_intervals[i] + 1e-6)
                )
            features.extend([np.mean(bigram_intervals), np.std(bigram_intervals)])
        else:
            features.extend([1.0, 0.0])
        
        return np.array(features)

# =====================================
# プロファイル管理システム
# =====================================

class KeystrokeProfileManager:
    """ユーザープロファイル管理システム"""
    
    def __init__(self, storage_path: str = "./keystroke_profiles"):
        self.storage_path = storage_path
        self.profiles = {}
        self.feature_extractor = KeystrokeFeatureExtractor()
        
        # ストレージディレクトリの作成
        import os
        os.makedirs(storage_path, exist_ok=True)
    
    def create_profile(self, user_id: str, training_events: List[KeystrokeEvent]) -> UserKeystrokeProfile:
        """新規プロファイルの作成"""
        profile = UserKeystrokeProfile(user_id=user_id)
        
        # 特徴量抽出
        dynamics = self.feature_extractor.extract_features(training_events)
        if not dynamics:
            raise ValueError("Not enough training data")
        
        # 統計的特徴の計算
        profile.mean_dwell_time = np.mean(dynamics.dwell_times)
        profile.std_dwell_time = np.std(dynamics.dwell_times)
        profile.mean_flight_time = np.mean(dynamics.flight_times)
        profile.std_flight_time = np.std(dynamics.flight_times)
        profile.mean_typing_speed = dynamics.typing_speed
        profile.std_typing_speed = dynamics.typing_speed * 0.1  # 仮の値
        
        # 分布の計算
        profile.dwell_time_distribution = self._calculate_distribution(dynamics.dwell_times)
        profile.flight_time_distribution = self._calculate_distribution(dynamics.flight_times)
        
        # 機械学習モデルの訓練
        profile = self._train_ml_model(profile, training_events)
        
        # プロファイルの保存
        self.profiles[user_id] = profile
        self.save_profile(profile)
        
        return profile
    
    def update_profile(self, user_id: str, new_events: List[KeystrokeEvent]):
        """プロファイルの更新"""
        if user_id not in self.profiles:
            self.load_profile(user_id)
        
        profile = self.profiles.get(user_id)
        if not profile:
            raise ValueError(f"Profile not found for user {user_id}")
        
        # 新しい特徴量を抽出
        dynamics = self.feature_extractor.extract_features(new_events)
        if not dynamics:
            return
        
        # 統計量の更新（指数移動平均）
        alpha = 0.1  # 学習率
        profile.mean_dwell_time = (1 - alpha) * profile.mean_dwell_time + alpha * np.mean(dynamics.dwell_times)
        profile.std_dwell_time = (1 - alpha) * profile.std_dwell_time + alpha * np.std(dynamics.dwell_times)
        profile.mean_flight_time = (1 - alpha) * profile.mean_flight_time + alpha * np.mean(dynamics.flight_times)
        profile.std_flight_time = (1 - alpha) * profile.std_flight_time + alpha * np.std(dynamics.flight_times)
        
        profile.sample_count += len(new_events)
        profile.last_updated = datetime.now()
        
        # モデルの再訓練（定期的に）
        if profile.sample_count % 1000 == 0:
            profile = self._train_ml_model(profile, new_events)
        
        self.save_profile(profile)
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """値の分布を計算"""
        if not values:
            return {}
        
        # ヒストグラムのビン
        bins = np.percentile(values, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        hist, _ = np.histogram(values, bins=bins)
        
        distribution = {}
        for i, count in enumerate(hist):
            key = f"bin_{i}"
            distribution[key] = count / len(values)
        
        return distribution
    
    def _train_ml_model(self, profile: UserKeystrokeProfile, events: List[KeystrokeEvent]) -> UserKeystrokeProfile:
        """機械学習モデルの訓練"""
        # 複数セッションから特徴量を抽出
        session_features = []
        
        # セッションごとにグループ化
        sessions = defaultdict(list)
        for event in events:
            sessions[event.session_id].append(event)
        
        # 各セッションから特徴量抽出
        for session_events in sessions.values():
            dynamics = self.feature_extractor.extract_features(session_events)
            if dynamics:
                features = self.feature_extractor.extract_advanced_features(dynamics)
                session_features.append(features)
        
        if len(session_features) < 5:
            return profile
        
        # 特徴量の正規化
        X = np.array(session_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # One-Class SVMの訓練
        model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        model.fit(X_scaled)
        
        profile.ml_model = model
        profile.feature_scaler = scaler
        
        return profile
    
    def save_profile(self, profile: UserKeystrokeProfile):
        """プロファイルの保存"""
        filepath = f"{self.storage_path}/{profile.user_id}_profile.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(profile, f)
    
    def load_profile(self, user_id: str) -> Optional[UserKeystrokeProfile]:
        """プロファイルの読み込み"""
        filepath = f"{self.storage_path}/{user_id}_profile.pkl"
        try:
            with open(filepath, 'rb') as f:
                profile = pickle.load(f)
                self.profiles[user_id] = profile
                return profile
        except FileNotFoundError:
            return None

# =====================================
# 認証システム
# =====================================

class KeystrokeAuthenticator:
    """キーストローク認証システム"""
    
    def __init__(self, profile_manager: KeystrokeProfileManager):
        self.profile_manager = profile_manager
        self.feature_extractor = KeystrokeFeatureExtractor()
        
        # 認証閾値
        self.statistical_threshold = 3.0  # 標準偏差の倍数
        self.ml_threshold = 0.0  # One-Class SVMの決定境界
        self.combined_threshold = 0.7  # 統合スコアの閾値
    
    def authenticate(self, user_id: str, keystroke_events: List[KeystrokeEvent]) -> Dict[str, Any]:
        """ユーザー認証の実行"""
        # プロファイルの取得
        profile = self.profile_manager.profiles.get(user_id)
        if not profile:
            profile = self.profile_manager.load_profile(user_id)
        
        if not profile:
            return {
                "authenticated": False,
                "confidence": 0.0,
                "reason": "Profile not found"
            }
        
        # 特徴量抽出
        dynamics = self.feature_extractor.extract_features(keystroke_events)
        if not dynamics:
            return {
                "authenticated": False,
                "confidence": 0.0,
                "reason": "Insufficient keystroke data"
            }
        
        # 統計的検証
        stat_score = self._statistical_verification(dynamics, profile)
        
        # 機械学習による検証
        ml_score = self._ml_verification(dynamics, profile)
        
        # スコアの統合
        combined_score = 0.7 * stat_score + 0.3 * ml_score
        
        # 認証判定
        authenticated = combined_score >= self.combined_threshold
        
        # 詳細な分析
        anomalies = self._detect_anomalies(dynamics, profile)
        
        return {
            "authenticated": authenticated,
            "confidence": combined_score,
            "statistical_score": stat_score,
            "ml_score": ml_score,
            "anomalies": anomalies,
            "reason": self._get_authentication_reason(authenticated, combined_score, anomalies)
        }
    
    def _statistical_verification(self, dynamics: KeystrokeDynamics, profile: UserKeystrokeProfile) -> float:
        """統計的検証"""
        scores = []
        
        # Dwell Timeの検証
        if dynamics.dwell_times:
            mean_dwell = np.mean(dynamics.dwell_times)
            z_score_dwell = abs(mean_dwell - profile.mean_dwell_time) / (profile.std_dwell_time + 1e-6)
            scores.append(1.0 - min(z_score_dwell / self.statistical_threshold, 1.0))
        
        # Flight Timeの検証
        if dynamics.flight_times:
            mean_flight = np.mean(dynamics.flight_times)
            z_score_flight = abs(mean_flight - profile.mean_flight_time) / (profile.std_flight_time + 1e-6)
            scores.append(1.0 - min(z_score_flight / self.statistical_threshold, 1.0))
        
        # タイピング速度の検証
        z_score_speed = abs(dynamics.typing_speed - profile.mean_typing_speed) / (profile.std_typing_speed + 1e-6)
        scores.append(1.0 - min(z_score_speed / self.statistical_threshold, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def _ml_verification(self, dynamics: KeystrokeDynamics, profile: UserKeystrokeProfile) -> float:
        """機械学習による検証"""
        if not profile.ml_model or not profile.feature_scaler:
            return 0.5  # モデルがない場合は中立的なスコア
        
        # 特徴量抽出
        features = self.feature_extractor.extract_advanced_features(dynamics)
        features_scaled = profile.feature_scaler.transform([features])
        
        # 予測
        decision = profile.ml_model.decision_function(features_scaled)[0]
        
        # スコアに変換（シグモイド関数）
        score = 1 / (1 + np.exp(-decision))
        
        return score
    
    def _detect_anomalies(self, dynamics: KeystrokeDynamics, profile: UserKeystrokeProfile) -> List[str]:
        """異常の検出"""
        anomalies = []
        
        # Dwell Timeの異常
        if dynamics.dwell_times:
            mean_dwell = np.mean(dynamics.dwell_times)
            if abs(mean_dwell - profile.mean_dwell_time) > 3 * profile.std_dwell_time:
                anomalies.append(f"Abnormal dwell time: {mean_dwell:.1f}ms (expected: {profile.mean_dwell_time:.1f}ms)")
        
        # Flight Timeの異常
        if dynamics.flight_times:
            mean_flight = np.mean(dynamics.flight_times)
            if abs(mean_flight - profile.mean_flight_time) > 3 * profile.std_flight_time:
                anomalies.append(f"Abnormal flight time: {mean_flight:.1f}ms (expected: {profile.mean_flight_time:.1f}ms)")
        
        # リズムの異常
        if dynamics.rhythm_consistency > 1.5:
            anomalies.append(f"Inconsistent typing rhythm: {dynamics.rhythm_consistency:.2f}")
        
        # タイピング速度の異常
        speed_diff = abs(dynamics.typing_speed - profile.mean_typing_speed)
        if speed_diff > 50:  # 50 keys/min difference
            anomalies.append(f"Unusual typing speed: {dynamics.typing_speed:.1f} keys/min")
        
        return anomalies
    
    def _get_authentication_reason(self, authenticated: bool, score: float, anomalies: List[str]) -> str:
        """認証理由の生成"""
        if authenticated:
            if score > 0.9:
                return "Strong match with user profile"
            else:
                return "Acceptable match with user profile"
        else:
            if anomalies:
                return f"Authentication failed: {'; '.join(anomalies[:2])}"
            else:
                return f"Low confidence score: {score:.2f}"

# =====================================
# 継続的認証システム
# =====================================

class ContinuousAuthenticationSystem:
    """継続的認証システム"""
    
    def __init__(self, authenticator: KeystrokeAuthenticator, window_size: int = 50):
        self.authenticator = authenticator
        self.window_size = window_size  # 認証に使用するキーストローク数
        self.session_events = defaultdict(deque)
        self.authentication_history = defaultdict(list)
        self.alert_threshold = 0.6  # アラート閾値
        
    def add_event(self, event: KeystrokeEvent):
        """イベントの追加と継続的認証"""
        # セッションごとのイベント管理
        session_key = f"{event.user_id}_{event.session_id}"
        self.session_events[session_key].append(event)
        
        # ウィンドウサイズを超えたら古いイベントを削除
        if len(self.session_events[session_key]) > self.window_size * 2:
            self.session_events[session_key].popleft()
        
        # 十分なイベントが集まったら認証実行
        if len(self.session_events[session_key]) >= self.window_size:
            recent_events = list(self.session_events[session_key])[-self.window_size:]
            auth_result = self.authenticator.authenticate(event.user_id, recent_events)
            
            # 履歴に追加
            self.authentication_history[session_key].append({
                "timestamp": datetime.now(),
                "result": auth_result
            })
            
            # アラートチェック
            if auth_result["confidence"] < self.alert_threshold:
                self._trigger_alert(event.user_id, event.session_id, auth_result)
            
            return auth_result
        
        return None
    
    def get_session_status(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """セッションの認証状態を取得"""
        session_key = f"{user_id}_{session_id}"
        history = self.authentication_history.get(session_key, [])
        
        if not history:
            return {
                "status": "no_data",
                "average_confidence": 0.0,
                "alert_count": 0
            }
        
        # 最近の認証結果を分析
        recent_results = history[-10:]  # 最新10件
        avg_confidence = np.mean([r["result"]["confidence"] for r in recent_results])
        alert_count = sum(1 for r in recent_results if r["result"]["confidence"] < self.alert_threshold)
        
        # ステータス判定
        if avg_confidence >= 0.8:
            status = "authenticated"
        elif avg_confidence >= 0.6:
            status = "warning"
        else:
            status = "suspicious"
        
        return {
            "status": status,
            "average_confidence": avg_confidence,
            "alert_count": alert_count,
            "last_check": recent_results[-1]["timestamp"],
            "trend": self._calculate_trend(recent_results)
        }
    
    def _trigger_alert(self, user_id: str, session_id: str, auth_result: Dict[str, Any]):
        """認証アラートの発行"""
        alert = {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "session_id": session_id,
            "confidence": auth_result["confidence"],
            "anomalies": auth_result.get("anomalies", []),
            "severity": self._calculate_severity(auth_result)
        }
        
        # 実際の実装では、ここでセキュリティチームに通知
        print(f"🚨 KEYSTROKE ALERT: User {user_id} - Confidence: {auth_result['confidence']:.2f}")
        print(f"   Anomalies: {', '.join(auth_result.get('anomalies', []))}")
        
        return alert
    
    def _calculate_trend(self, results: List[Dict]) -> str:
        """信頼度のトレンドを計算"""
        if len(results) < 3:
            return "stable"
        
        confidences = [r["result"]["confidence"] for r in results]
        recent_avg = np.mean(confidences[-3:])
        older_avg = np.mean(confidences[-6:-3]) if len(confidences) >= 6 else np.mean(confidences[:-3])
        
        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_severity(self, auth_result: Dict[str, Any]) -> str:
        """アラートの深刻度を計算"""
        confidence = auth_result["confidence"]
        
        if confidence < 0.3:
            return "critical"
        elif confidence < 0.5:
            return "high"
        elif confidence < 0.7:
            return "medium"
        else:
            return "low"

# =====================================
# セキュリティチェーンとの統合
# =====================================

class KeystrokeEnhancedSecurityChain:
    """キーストローク認証を統合したセキュリティチェーン"""
    
    def __init__(self, chain_manager, keystroke_auth_system: ContinuousAuthenticationSystem):
        self.chain_manager = chain_manager
        self.keystroke_auth = keystroke_auth_system
        
    def process_event_with_keystroke(self, event: Dict, keystroke_events: List[KeystrokeEvent]) -> Dict:
        """キーストローク認証を含むイベント処理"""
        
        # キーストローク認証の実行
        keystroke_result = None
        if keystroke_events:
            # イベントをシステムに追加
            for ke in keystroke_events:
                auth_result = self.keystroke_auth.add_event(ke)
                if auth_result:
                    keystroke_result = auth_result
        
        # 通常のセキュリティチェーン処理
        chain_result = self.chain_manager.process_event(event)
        
        # キーストローク認証結果の統合
        if keystroke_result:
            # 信頼スコアの調整
            original_trust = event.get("trust_score", 1.0)
            keystroke_confidence = keystroke_result["confidence"]
            
            # 統合信頼スコア
            integrated_trust = 0.7 * original_trust + 0.3 * keystroke_confidence
            
            # 異常検知の強化
            if keystroke_confidence < 0.5:
                if chain_result["status"] == "normal":
                    chain_result["status"] = "investigating"
                    chain_result["reason"] = f"Keystroke anomaly detected: {keystroke_result['reason']}"
                elif chain_result["status"] == "investigating":
                    chain_result["status"] = "suspicious"
            
            # 結果に追加情報を含める
            chain_result["keystroke_auth"] = {
                "confidence": keystroke_confidence,
                "anomalies": keystroke_result.get("anomalies", []),
                "integrated_trust": integrated_trust
            }
        
        return chain_result

# =====================================
# 使用例とテスト
# =====================================

def example_usage():
    """キーストロークダイナミクスシステムの使用例"""
    
    # 1. システムの初期化
    print("=== キーストロークダイナミクス認証システム ===\n")
    
    # コレクターの初期化
    collector = KeystrokeCollector(privacy_mode=True)
    
    # プロファイルマネージャーの初期化
    profile_manager = KeystrokeProfileManager()
    
    # 認証システムの初期化
    authenticator = KeystrokeAuthenticator(profile_manager)
    
    # 継続認証システムの初期化
    continuous_auth = ContinuousAuthenticationSystem(authenticator)
    
    # 2. トレーニングフェーズ
    print("📝 トレーニングフェーズ開始...")
    user_id = "test_user"
    session_id = "training_session"
    
    # キーストローク収集（シミュレーション）
    collector.start_collection(user_id, session_id)
    import time
    time.sleep(2)  # 2秒間収集
    collector.stop_collection()
    
    # トレーニングイベントの取得
    training_events = collector.get_session_events(session_id)
    print(f"  収集したイベント数: {len(training_events)}")
    
    # プロファイル作成
    if len(training_events) >= 20:
        profile = profile_manager.create_profile(user_id, training_events)
        print(f"  ✅ プロファイル作成完了")
        print(f"     平均Dwell Time: {profile.mean_dwell_time:.1f}ms")
        print(f"     平均Flight Time: {profile.mean_flight_time:.1f}ms")
        print(f"     タイピング速度: {profile.mean_typing_speed:.1f} keys/min")
    
    # 3. 認証フェーズ
    print("\n🔐 認証フェーズ開始...")
    
    # 新しいセッションでキーストローク収集
    test_session_id = "test_session"
    collector.start_collection(user_id, test_session_id)
    time.sleep(1)  # 1秒間収集
    collector.stop_collection()
    
    # 認証実行
    test_events = collector.get_session_events(test_session_id)
    auth_result = authenticator.authenticate(user_id, test_events)
    
    print(f"\n認証結果:")
    print(f"  認証成功: {auth_result['authenticated']}")
    print(f"  信頼度: {auth_result['confidence']:.2f}")
    print(f"  統計スコア: {auth_result['statistical_score']:.2f}")
    print(f"  MLスコア: {auth_result['ml_score']:.2f}")
    
    if auth_result['anomalies']:
        print(f"  検出された異常:")
        for anomaly in auth_result['anomalies']:
            print(f"    - {anomaly}")
    
    # 4. 継続認証のデモ
    print("\n🔄 継続認証デモ...")
    
    # 複数のイベントを追加
    for event in test_events[:30]:  # 最初の30イベント
        result = continuous_auth.add_event(event)
        if result:
            print(f"  継続認証実行 - 信頼度: {result['confidence']:.2f}")
    
    # セッション状態の確認
    session_status = continuous_auth.get_session_status(user_id, test_session_id)
    print(f"\nセッション状態:")
    print(f"  ステータス: {session_status['status']}")
    print(f"  平均信頼度: {session_status['average_confidence']:.2f}")
    print(f"  アラート数: {session_status['alert_count']}")
    print(f"  トレンド: {session_status['trend']}")

if __name__ == "__main__":
    example_usage()


""""
継続的認証
pythoncontinuous_auth = ContinuousAuthenticationSystem(authenticator)
# リアルタイムでキーストロークを監視
result = continuous_auth.add_event(keystroke_event)

セキュリティチェーンとの統合
pythonenhanced_chain = KeystrokeEnhancedSecurityChain(chain_manager, keystroke_auth)
result = enhanced_chain.process_event_with_keystroke(event, keystroke_events)

#　🛡️**“通常時は緩やか、怪しい時だけ厳密認証”**
#　普段は警戒度低く、警告フラグ時に「真面目判定モード」へ！
#　２段階でのチェックレベル引き上げも可能
result = enhanced_chain.process_event_with_keystroke(event, keystroke_events)
if result['status'] in ['suspicious', 'critical']:
    keystroke_result = authenticator.authenticate(user_id, recent_keystroke_events)
    if keystroke_result['confidence'] < 0.5:
        result['status'] = 'reject'
        # ここで自動的に一時ブロック、アラート送信なども
"""
