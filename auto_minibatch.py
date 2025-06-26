"""
AdaptiveTensorFieldSecurity - MiniBatchKMeans自動増分学習実装
=====================================================
セキュリティイベントの増分学習を行うMiniBatchKMeansの実装
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import queue
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import hashlib
from dataclasses import dataclass, field
import asyncio
import psutil

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """学習設定"""
    n_clusters_normal: int = 3
    n_clusters_exceptional: int = 2
    n_clusters_anomaly: int = 2
    batch_size_default: int = 500
    batch_size_min: int = 100
    batch_size_max: int = 1000
    update_interval_normal: int = 3600  # 1時間（秒）
    update_interval_peak: int = 1800    # 30分（秒）
    update_interval_night: int = 7200   # 2時間（秒）
    update_interval_emergency: int = 600 # 10分（秒）
    max_queue_size: int = 10000
    reassignment_ratio: float = 0.01
    random_state: int = 42
    silhouette_threshold: float = 0.3
    anomaly_rate_threshold: float = 2.0
    drift_threshold: float = 0.3
    n_init: int = 3
    max_iter: int = 100


@dataclass
class EventFeatures:
    """セキュリティイベントの特徴量（11次元）"""
    severity_level: float = 0.0
    action_magnitude: float = 0.0
    threat_context: float = 0.0
    trust_score: float = 1.0
    security_mode: float = 0.0
    event_score: float = 0.0
    is_confidential_access: float = 0.0
    is_cross_dept_access: float = 0.0
    is_external_network: float = 0.0
    permission_level_diff: float = 0.0
    operation_success_flag: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """特徴量をベクトル化"""
        return np.array([
            self.severity_level,
            self.action_magnitude,
            self.threat_context,
            self.trust_score,
            self.security_mode,
            self.event_score,
            self.is_confidential_access,
            self.is_cross_dept_access,
            self.is_external_network,
            self.permission_level_diff,
            self.operation_success_flag
        ])


class AdaptiveMiniBatchKMeans:
    """適応的MiniBatchKMeans実装"""
    
    def __init__(self, config: LearningConfig = None):
        self.config = config or LearningConfig()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # モデル群
        self.models = {
            'normal': self._create_model(self.config.n_clusters_normal),
            'exceptional': self._create_model(self.config.n_clusters_exceptional),
            'anomaly': self._create_model(self.config.n_clusters_anomaly)
        }
        
        # データバッファ
        self.event_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.event_buffer = deque(maxlen=self.config.batch_size_max * 10)
        
        # 統計情報
        self.stats = {
            'total_events': 0,
            'anomaly_count': 0,
            'last_update': datetime.now(),
            'baseline_anomaly_rate': 0.05,
            'current_anomaly_rate': 0.0,
            'cluster_quality': {}
        }
        
        # 学習履歴
        self.learning_history = deque(maxlen=1000)
        
        # スレッド管理
        self.learning_thread = None
        self.is_running = False
        
    def _create_model(self, n_clusters: int) -> MiniBatchKMeans:
        """MiniBatchKMeansモデルを作成"""
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            batch_size=self.config.batch_size_default,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            reassignment_ratio=self.config.reassignment_ratio,
            random_state=self.config.random_state
        )
    
    def add_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """新しいイベントを追加して即座に評価"""
        # 特徴量抽出
        features = self._extract_features(event)
        vector = features.to_vector()
        
        # 統計更新
        self.stats['total_events'] += 1
        
        # 即座に異常評価
        result = self.evaluate_event(vector, event)
        
        # バッファに追加（学習用）
        self.event_buffer.append({
            'vector': vector,
            'features': features,
            'timestamp': datetime.now(),
            'result': result,
            'raw_event': event
        })
        
        # 学習トリガー判定
        if self._should_trigger_learning():
            self._trigger_incremental_learning()
        
        return result
    
    def evaluate_event(self, vector: np.ndarray, raw_event: Dict = None) -> Dict[str, Any]:
        """イベントを評価"""
        if not self.is_fitted:
            return {
                'verdict': 'unknown',
                'model_type': None,
                'cluster_id': -1,
                'distance': float('inf'),
                'confidence': 0.0,
                'raw_event': raw_event
            }
        
        # 正規化
        vector_scaled = self.scaler.transform(vector.reshape(1, -1))[0]
        
        # 各モデルでの評価
        results = {}
        for model_type, model in self.models.items():
            if hasattr(model, 'cluster_centers_') and model.cluster_centers_ is not None:
                # 最近傍クラスタを特定
                distances = np.linalg.norm(model.cluster_centers_ - vector_scaled, axis=1)
                cluster_id = np.argmin(distances)
                min_distance = distances[cluster_id]
                
                results[model_type] = {
                    'cluster_id': cluster_id,
                    'distance': min_distance,
                    'confidence': 1.0 / (1.0 + min_distance)
                }
        
        # 最適なモデルを選択
        if results:
            best_model = min(results.items(), key=lambda x: x[1]['distance'])
            model_type = best_model[0]
            
            verdict = self._determine_verdict(
                model_type, 
                best_model[1]['distance'],
                best_model[1]['confidence']
            )
            
            return {
                'verdict': verdict,
                'model_type': model_type,
                'cluster_id': best_model[1]['cluster_id'],
                'distance': best_model[1]['distance'],
                'confidence': best_model[1]['confidence'],
                'all_distances': results,
                'raw_event': raw_event
            }
        
        return {
            'verdict': 'unknown',
            'model_type': None,
            'cluster_id': -1,
            'distance': float('inf'),
            'confidence': 0.0,
            'raw_event': raw_event
        }
    
    def _extract_features(self, event: Dict[str, Any]) -> EventFeatures:
        """イベントから特徴量を抽出"""
        # ここは実際のsecurity_event_to_state関数の簡略版
        features = EventFeatures()
        
        # イベントスコアの計算（簡略版）
        operation = event.get('operation', '')
        operation_scores = {
            'FileRead': 15, 'FileWrite': 30, 'FileCopy': 60,
            'FileDelete': 50, 'ProcessCreate': 40, 'NetworkConnect': 35,
            'Login': 10, 'LoginFailed': 40, 'Logout': 2
        }
        features.event_score = operation_scores.get(operation, 10) / 100.0
        
        # 重大度レベル
        if features.event_score > 0.5:
            features.severity_level = 0.8
        elif features.event_score > 0.3:
            features.severity_level = 0.5
        else:
            features.severity_level = 0.2
        
        # アクション強度（仮）
        features.action_magnitude = np.random.rand() * 0.5 + features.event_score * 0.5
        
        # 脅威コンテキスト
        if event.get('destination_ip', '').startswith(('198.', '203.')):
            features.threat_context = 0.8
            features.is_external_network = 1.0
        
        # 信頼スコア（過去の履歴から計算するが、ここでは簡略化）
        if event.get('status') == 'FAILED':
            features.trust_score = 0.5
            features.operation_success_flag = 0.0
        
        # 部署間アクセス
        if event.get('cross_dept_access'):
            features.is_cross_dept_access = 1.0
            features.permission_level_diff = 0.5
        
        # 機密アクセス
        file_path = event.get('file_path', '').lower()
        if 'confidential' in file_path or 'secret' in file_path:
            features.is_confidential_access = 1.0
        
        return features
    
    def _should_trigger_learning(self) -> bool:
        """学習をトリガーすべきか判定"""
        triggers = {
            # イベント数トリガー
            'event_count': len(self.event_buffer) >= self._get_adaptive_batch_size(),
            
            # 異常率トリガー
            'anomaly_rate': self._calculate_current_anomaly_rate() > 
                          self.stats['baseline_anomaly_rate'] * self.config.anomaly_rate_threshold,
            
            # ドリフト検出
            'concept_drift': self._detect_concept_drift() > self.config.drift_threshold,
            
            # 時間経過
            'time_elapsed': (datetime.now() - self.stats['last_update']).seconds > 
                          self._get_update_interval()
        }
        
        triggered = any(triggers.values())
        
        if triggered:
            logger.info(f"学習トリガー発動: {[k for k, v in triggers.items() if v]}")
        
        return triggered
    
    def _get_adaptive_batch_size(self) -> int:
        """システム負荷に応じてバッチサイズを動的調整"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80 or memory_percent > 85:
            return self.config.batch_size_min
        elif cpu_percent < 30 and memory_percent < 40:
            return self.config.batch_size_max
        else:
            return self.config.batch_size_default
    
    def _get_update_interval(self) -> int:
        """現在の状況に応じた更新間隔を取得"""
        hour = datetime.now().hour
        
        # 緊急モード判定
        if self._calculate_current_anomaly_rate() > 0.2:
            return self.config.update_interval_emergency
        
        # 時間帯別
        if 9 <= hour <= 18:  # 業務時間
            return self.config.update_interval_peak
        elif 22 <= hour or hour <= 6:  # 深夜
            return self.config.update_interval_night
        else:
            return self.config.update_interval_normal
    
    def _calculate_current_anomaly_rate(self) -> float:
        """現在の異常率を計算"""
        recent_events = list(self.event_buffer)[-100:]
        if not recent_events:
            return 0.0
        
        anomaly_count = sum(1 for e in recent_events 
                          if e['result'].get('verdict') in ['suspicious', 'critical'])
        return anomaly_count / len(recent_events)
    
    def _detect_concept_drift(self) -> float:
        """概念ドリフトを検出"""
        if len(self.event_buffer) < 100:
            return 0.0
        
        # 直近のデータと過去のデータの分布を比較
        recent_vectors = np.array([e['vector'] for e in list(self.event_buffer)[-50:]])
        older_vectors = np.array([e['vector'] for e in list(self.event_buffer)[-100:-50]])
        
        # 各特徴量の平均値の変化を計算
        recent_mean = np.mean(recent_vectors, axis=0)
        older_mean = np.mean(older_vectors, axis=0)
        
        # 正規化された距離
        drift_score = np.linalg.norm(recent_mean - older_mean) / np.sqrt(len(recent_mean))
        
        return drift_score
    
    def _determine_verdict(self, model_type: str, distance: float, confidence: float) -> str:
        """モデルタイプと距離から判定を決定"""
        if model_type == 'normal':
            if distance < 5.0:
                return 'normal'
            elif distance < 10.0:
                return 'investigating'
            elif distance < 15.0:
                return 'suspicious'
            else:
                return 'critical'
        elif model_type == 'exceptional':
            if distance < 8.0:
                return 'normal'  # 例外として許容
            else:
                return 'investigating'
        else:  # anomaly
            if distance < 10.0:
                return 'critical'
            else:
                return 'suspicious'
    
    def _trigger_incremental_learning(self):
        """増分学習をトリガー"""
        if not self.is_running:
            self.is_running = True
            self.learning_thread = threading.Thread(target=self._incremental_learning_worker)
            self.learning_thread.daemon = True
            self.learning_thread.start()
    
    def _incremental_learning_worker(self):
        """増分学習ワーカースレッド"""
        try:
            logger.info("増分学習開始")
            start_time = datetime.now()
            
            # バッファからデータを取得
            batch_size = self._get_adaptive_batch_size()
            batch_data = list(self.event_buffer)[-batch_size:]
            
            if len(batch_data) < 10:
                logger.warning("学習データが不足")
                return
            
            # ベクトル抽出
            vectors = np.array([e['vector'] for e in batch_data])
            
            # 初回学習または再学習
            if not self.is_fitted:
                self._initial_fit(vectors)
            else:
                self._partial_fit(vectors, batch_data)
            
            # 品質評価
            quality_metrics = self._evaluate_cluster_quality(vectors)
            self.stats['cluster_quality'] = quality_metrics
            
            # 品質が閾値を下回ったら再学習
            if quality_metrics.get('silhouette', 0) < self.config.silhouette_threshold:
                logger.warning(f"クラスタ品質低下: {quality_metrics['silhouette']:.3f}")
                self._refit_models(vectors)
            
            # 統計更新
            self.stats['last_update'] = datetime.now()
            self.stats['current_anomaly_rate'] = self._calculate_current_anomaly_rate()
            
            # 学習履歴に記録
            self.learning_history.append({
                'timestamp': datetime.now(),
                'batch_size': len(batch_data),
                'duration': (datetime.now() - start_time).total_seconds(),
                'quality_metrics': quality_metrics,
                'anomaly_rate': self.stats['current_anomaly_rate']
            })
            
            logger.info(f"増分学習完了: {len(batch_data)}件, "
                       f"{(datetime.now() - start_time).total_seconds():.2f}秒")
            
        except Exception as e:
            logger.error(f"増分学習エラー: {e}")
        finally:
            self.is_running = False
    
    def _initial_fit(self, vectors: np.ndarray):
        """初回学習"""
        logger.info("初回学習実行")
        
        # スケーラーのフィット
        self.scaler.fit(vectors)
        vectors_scaled = self.scaler.transform(vectors)
        
        # 正常モデルの学習（全データを正常と仮定）
        self.models['normal'].fit(vectors_scaled)
        self.is_fitted = True
        
        logger.info(f"初回学習完了: {len(vectors)}件")
    
    def _partial_fit(self, vectors: np.ndarray, batch_data: List[Dict]):
        """部分学習"""
        vectors_scaled = self.scaler.transform(vectors)
        
        # 各モデルごとにデータを分類
        model_data = defaultdict(list)
        
        for i, data in enumerate(batch_data):
            verdict = data['result'].get('verdict', 'unknown')
            
            if verdict in ['normal', 'investigating']:
                model_data['normal'].append(vectors_scaled[i])
            elif verdict in ['suspicious', 'critical'] and data['result'].get('model_type') == 'anomaly':
                model_data['anomaly'].append(vectors_scaled[i])
            elif verdict == 'normal' and data['result'].get('model_type') == 'exceptional':
                model_data['exceptional'].append(vectors_scaled[i])
        
        # 各モデルを更新
        for model_type, data in model_data.items():
            if len(data) >= 10:  # 最小サンプル数
                data_array = np.array(data)
                self.models[model_type].partial_fit(data_array)
                logger.info(f"{model_type}モデル更新: {len(data)}件")
    
    def _refit_models(self, vectors: np.ndarray):
        """モデルの再学習"""
        logger.warning("モデル再学習開始")
        
        # スケーラーも更新
        self.scaler.fit(vectors)
        vectors_scaled = self.scaler.transform(vectors)
        
        # K-meansで初期分類
        initial_kmeans = MiniBatchKMeans(
            n_clusters=self.config.n_clusters_normal + 
                      self.config.n_clusters_exceptional + 
                      self.config.n_clusters_anomaly,
            random_state=self.config.random_state
        )
        labels = initial_kmeans.fit_predict(vectors_scaled)
        
        # クラスタを分析して各モデルに割り当て
        cluster_sizes = np.bincount(labels)
        large_clusters = np.argsort(cluster_sizes)[-self.config.n_clusters_normal:]
        small_clusters = np.argsort(cluster_sizes)[:self.config.n_clusters_anomaly]
        medium_clusters = [c for c in range(len(cluster_sizes)) 
                          if c not in large_clusters and c not in small_clusters]
        
        # 各モデルを再学習
        for cluster_ids, model_type in [
            (large_clusters, 'normal'),
            (medium_clusters[:self.config.n_clusters_exceptional], 'exceptional'),
            (small_clusters, 'anomaly')
        ]:
            if len(cluster_ids) > 0:
                mask = np.isin(labels, cluster_ids)
                if np.sum(mask) > 0:
                    self.models[model_type].fit(vectors_scaled[mask])
        
        logger.info("モデル再学習完了")
    
    def _evaluate_cluster_quality(self, vectors: np.ndarray) -> Dict[str, float]:
        """クラスタ品質を評価"""
        if not self.is_fitted:
            return {}
        
        vectors_scaled = self.scaler.transform(vectors)
        metrics = {}
        
        try:
            # 正常モデルの品質評価
            if hasattr(self.models['normal'], 'labels_'):
                labels = self.models['normal'].predict(vectors_scaled)
                if len(np.unique(labels)) > 1:
                    metrics['silhouette'] = silhouette_score(vectors_scaled, labels)
                    metrics['davies_bouldin'] = davies_bouldin_score(vectors_scaled, labels)
        except Exception as e:
            logger.warning(f"品質評価エラー: {e}")
        
        return metrics
    
    def save_models(self, filepath: str):
        """モデルを保存"""
        state = {
            'models': {k: pickle.dumps(v) for k, v in self.models.items()},
            'scaler': pickle.dumps(self.scaler),
            'stats': self.stats,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"モデル保存完了: {filepath}")
    
    def load_models(self, filepath: str):
        """モデルを読み込み"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.models = {k: pickle.loads(v) for k, v in state['models'].items()}
        self.scaler = pickle.loads(state['scaler'])
        self.stats = state['stats']
        self.config = state['config']
        self.is_fitted = state['is_fitted']
        
        logger.info(f"モデル読み込み完了: {filepath}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """クラスタ情報を取得"""
        info = {}
        
        for model_type, model in self.models.items():
            if hasattr(model, 'cluster_centers_') and model.cluster_centers_ is not None:
                info[model_type] = {
                    'n_clusters': len(model.cluster_centers_),
                    'cluster_sizes': self._get_cluster_sizes(model),
                    'center_distances': self._get_center_distances(model)
                }
        
        return info
    
    def _get_cluster_sizes(self, model: MiniBatchKMeans) -> List[int]:
        """各クラスタのサイズを取得"""
        # 最近のデータでクラスタサイズを推定
        recent_vectors = [e['vector'] for e in list(self.event_buffer)[-1000:]]
        if not recent_vectors:
            return []
        
        vectors_scaled = self.scaler.transform(np.array(recent_vectors))
        labels = model.predict(vectors_scaled)
        
        return np.bincount(labels).tolist()
    
    def _get_center_distances(self, model: MiniBatchKMeans) -> np.ndarray:
        """クラスタ中心間の距離を計算"""
        centers = model.cluster_centers_
        n_clusters = len(centers)
        distances = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances


# 使用例とテストコード
if __name__ == "__main__":
    # 設定の作成
    config = LearningConfig(
        n_clusters_normal=3,
        batch_size_default=50,  # テスト用に小さく
        update_interval_normal=60  # 1分
    )
    
    # インスタンス作成
    learner = AdaptiveMiniBatchKMeans(config)
    
    # テストイベントの生成
    test_events = []
    
    # 正常イベント
    for i in range(100):
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': f'user_{i % 10}',
            'operation': np.random.choice(['FileRead', 'Login', 'Logout']),
            'status': 'SUCCESS',
            'department': np.random.choice(['sales', 'hr', 'engineering'])
        }
        test_events.append(event)
    
    # 異常イベント
    for i in range(20):
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': f'attacker_{i}',
            'operation': np.random.choice(['FileDelete', 'NetworkConnect']),
            'status': 'FAILED',
            'destination_ip': '198.51.100.50',
            'file_path': '/confidential/secret.docx',
            'cross_dept_access': True
        }
        test_events.append(event)
    
    # イベントを処理
    results = []
    for event in test_events:
        result = learner.add_event(event)
        results.append(result)
        
        # 結果のサンプル表示
        if len(results) % 20 == 0:
            print(f"\n処理済みイベント数: {len(results)}")
            print(f"最新の判定: {result['verdict']}")
            print(f"信頼度: {result['confidence']:.3f}")
            print(f"異常率: {learner.stats['current_anomaly_rate']:.3f}")
    
    # 最終的なクラスタ情報
    print("\n=== クラスタ情報 ===")
    cluster_info = learner.get_cluster_info()
    for model_type, info in cluster_info.items():
        print(f"\n{model_type}モデル:")
        print(f"  クラスタ数: {info['n_clusters']}")
        print(f"  各クラスタサイズ: {info['cluster_sizes']}")
    
    # 学習履歴
    print("\n=== 学習履歴 ===")
    for history in list(learner.learning_history)[-5:]:
        print(f"時刻: {history['timestamp']}")
        print(f"  バッチサイズ: {history['batch_size']}")
        print(f"  処理時間: {history['duration']:.2f}秒")
        print(f"  品質指標: {history.get('quality_metrics', {})}")
    
    # モデルの保存
    learner.save_models('adaptive_kmeans_model.pkl')
    print("\nモデルを保存しました")

"""
# =======================================
# 基本的な使い方
# =======================================

learner = AdaptiveMiniBatchKMeans()

# イベントを追加（自動的に学習もトリガー）
result = learner.add_event({
    'operation': 'FileRead',
    'user_id': 'yamada_t',
    'file_path': '/confidential/data.xlsx'
})

# 判定結果
print(result['verdict'])  # 'suspicious'
print(result['confidence'])  # 0.85
"""
