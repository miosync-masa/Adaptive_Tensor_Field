# =====================================
# 完全統合版 MLOps + Lambda³ベイズ推論パイプライン
# =====================================

import asyncio
import json
import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
import aiohttp
from kafka import KafkaConsumer, KafkaProducer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import mlflow
import kubernetes
from kubernetes import client, config
import schedule
import threading

# =====================================
# module呼び出し
# =====================================
from auto_minibatch import AdaptiveMiniBatchKMeans
from policy_manager import PolicyManager
from diversity_adversarial_defense 
import AdversarialDefense, DiversityAwareCluster, PolicyManager as AdvPolicyManager

# グローバルインスタンス
auto_minibatch = AdaptiveMiniBatchKMeans()
policy_manager = PolicyManager()

# =====================================
# 設定クラス
# =====================================

@dataclass
class OptimizedL3Config:
    """最適化されたLambda³設定"""
    # Lambda³パラメータ
    delta_percentile: float = 90.0
    local_window: int = 10
    local_jump_percentile: float = 95.0
    window: int = 20
    
    # MCMC最適化パラメータ
    draws: int = 500
    tune: int = 200
    chains: int = 2
    target_accept: float = 0.8
    
    # 高速化オプション
    use_jax: bool = True
    use_gpu: bool = False
    save_warmup: bool = True
    
    # インクリメンタル学習
    incremental_update: bool = True
    warmup_cache_path: str = "./warmup_cache"
    
    # MLOpsパラメータ
    kafka_servers: List[str] = None
    threat_intel_apis: List[str] = None
    min_batch_size: int = 100
    
    def __post_init__(self):
        if self.kafka_servers is None:
            self.kafka_servers = ['localhost:9092']
        if self.threat_intel_apis is None:
            self.threat_intel_apis = []

# =====================================
# 0. 多様性＆敵対的防御モジュール（AdversarialDefense等）
# =====================================

class SimpleAdversarialDefense:
    def __init__(self, normal_samples=None, event_buffer=None):
        self.normal_samples = normal_samples or []
        self.event_buffer = event_buffer or []

    def get_recent_anomalies(self):
        """
        (参考実装) メモリ上のイベントバッファから抽出
        本番運用はDB,ログAPI等で好きに書き換えOK
        """
        return [ev for ev in self.event_buffer if ev.get("is_anomaly", False)]

    # 他にもgenerate_attack_variations, train_against_variants...など追加

# =====================================
# 1. データ収集モジュール（Lambda³統合）
# =====================================

class EnhancedDataCollector:
    """
    Lambda³特徴量を含む強化データ収集クラス
    ・即時免疫反応（異常時は自動でローカル学習・ブロードキャスト・EDR連携）
    ・系列ごとバッファリング＋バッチ学習（Lambda³特徴量抽出）
    ・柔軟な前処理パイプライン連携（send_to_preprocessing）
    """
    def __init__(
        self,
        config: OptimizedL3Config,
        auto_minibatch=None,
        kafka_producer=None,
        policy_manager=None,
        preprocessor=None
    ):
        self.config = config
        self.kafka_consumer = KafkaConsumer(
            'security-events',
            bootstrap_servers=config.kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.data_buffer = []
        self.series_data = {}  # 系列ごとのイベントバッファ

        # 外部連携系
        self.auto_minibatch = auto_minibatch or globals().get("auto_minibatch")
        self.kafka_producer = kafka_producer
        self.policy_manager = policy_manager or globals().get("policy_manager")

        # 前処理パイプラインの受け口（オプション）
        self.preprocessor = preprocessor

    def set_preprocessor(self, preprocessor):
        """前処理パイプライン連携の後付け登録も可"""
        self.preprocessor = preprocessor

    async def collect_security_events(self):
        """セキュリティイベントの非同期収集＋即時免疫反応"""
        async for message in self.kafka_consumer:
            event = message.value

            # --- 1. 即時免疫反応 ---
            if self._is_anomaly(event):
                self._handle_immunity(event)

            # --- 2. 系列ごとのバッファリング ---
            series_name = event.get('series', 'default')
            self.series_data.setdefault(series_name, []).append(event)
            self.data_buffer.append(event)

            # --- 3. バッチ学習の発火 ---
            if len(self.data_buffer) >= self.config.min_batch_size:
                await self.process_batch()

    def _is_anomaly(self, event):
        """イベントが異常かどうか判定（汎用：dict型・object型両対応）"""
        if hasattr(event, "is_anomaly") and callable(event.is_anomaly):
            return event.is_anomaly()
        if isinstance(event, dict) and "anomaly" in event:
            return event["anomaly"]
        return False

    def _handle_immunity(self, event):
        """異常イベント時の即時免疫処理（並列発火OK）"""
        # 1. ローカルミニバッチ学習
        if self.auto_minibatch:
            self.auto_minibatch.add_event(event)
        # 2. 即時Kafkaブロードキャスト
        if self.kafka_producer:
            self.kafka_producer.send('anomaly-topic', json.dumps(event).encode())
        # 3. ポリシー／EDRアップデート
        if self.policy_manager:
            self.policy_manager.update_defense(event)

    async def process_batch(self):
        """系列バッファのバッチ処理・Lambda³特徴量抽出・前処理送信"""
        if not self.data_buffer:
            return

        features_by_series = {}
        for series_name, series_events in self.series_data.items():
            if len(series_events) > 10:
                values = np.array([e['value'] for e in series_events])
                features = self.calc_lambda3_features(values)
                features_by_series[series_name] = features

        # 前処理パイプラインへ渡す（非同期/バッファ式）
        await self.send_to_preprocessing(features_by_series)

        # バッファクリア
        self.data_buffer.clear()
        # series_data側は必要に応じて都度クリアor伸ばす
        # self.series_data[series_name] = []  # 必要なら各系列で切ってもよい

    async def send_to_preprocessing(self, features_by_series):
        """Lambda³特徴量バッチを前処理パイプラインへ非同期送信"""
        if self.preprocessor:
            # 受け口がasync/await型なら
            if hasattr(self.preprocessor, 'receive_features'):
                await self.preprocessor.receive_features(features_by_series)
            else:
                # 非同期でない場合（例：直列パイプラインの場合は同期呼び出しでもOK）
                self.preprocessor.receive_features(features_by_series)
        else:
            print("[WARN] Preprocessor未登録: features未送信")

    def calc_lambda3_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Lambda³特徴量の計算（系列単位で呼び出し）"""
        n = len(data)
        diff = np.diff(data, prepend=data[0])
        threshold = np.percentile(np.abs(diff), self.config.delta_percentile)

        # ジャンプ検出
        delta_LambdaC_pos = (diff > threshold).astype(np.float32)
        delta_LambdaC_neg = (diff < -threshold).astype(np.float32)

        # ローカル標準偏差
        window = self.config.local_window
        padded_data = np.pad(data, (window, window), mode='edge')
        local_std = np.array([
            np.std(padded_data[i:i + 2 * window + 1])
            for i in range(n)
        ])

        # ローカルジャンプ
        score = np.abs(diff) / (local_std + 1e-8)
        local_threshold = np.percentile(score, self.config.local_jump_percentile)
        local_jump_detect = (score > local_threshold).astype(np.float32)

        # 時間窓標準偏差
        rho_T = np.array([
            np.std(data[max(0, i - self.config.window):i + 1])
            for i in range(n)
        ])

        # 時間トレンド
        time_trend = np.arange(n, dtype=np.float32) / n

        return {
            'delta_LambdaC_pos': delta_LambdaC_pos,
            'delta_LambdaC_neg': delta_LambdaC_neg,
            'rho_T': rho_T,
            'time_trend': time_trend,
            'local_jump_detect': local_jump_detect,
            'raw_data': data
        }

# =====================================
# 2. 前処理モジュール（Lambda³対応）
# =====================================

class Lambda3DataPreprocessor:
    """Lambda³特化型前処理"""
    
    def __init__(self, config: OptimizedL3Config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_history = []
        self.lambda3_memory = []  # Lambda³イベントメモリ
        
        # 1分ごとの定期実行設定
        schedule.every(1).minutes.do(self.run_preprocessing)
        threading.Thread(target=self._run_scheduler, daemon=True).start()
    
    def _run_scheduler(self):
        """スケジューラーの実行"""
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    def run_preprocessing(self):
        """前処理のメインループ"""
        batch_features = self.get_batch_from_queue()
        
        if batch_features:
            # Lambda³特徴量の統合
            integrated_features = self.integrate_lambda3_features(batch_features)
            
            # イベントメモリの更新
            self.update_lambda3_memory(integrated_features)
            
            # 因果性分析
            causality_features = self.analyze_causality(integrated_features)
            
            # 最終特徴量の構築
            final_features = self.build_final_features(
                integrated_features, 
                causality_features
            )
            
            # 学習パイプラインに送信
            self.send_to_training_pipeline(final_features)
    
    def integrate_lambda3_features(self, batch_features: Dict) -> Dict:
        """複数系列のLambda³特徴量を統合"""
        integrated = {}
        
        # 各系列の特徴量を結合
        for series_name, features in batch_features.items():
            for feature_name, values in features.items():
                key = f"{series_name}_{feature_name}"
                integrated[key] = values
        
        # 系列間相互作用特徴量
        if len(batch_features) > 1:
            series_names = list(batch_features.keys())
            for i in range(len(series_names)):
                for j in range(i+1, len(series_names)):
                    # 相互相関
                    series_a = batch_features[series_names[i]]['raw_data']
                    series_b = batch_features[series_names[j]]['raw_data']
                    
                    if len(series_a) == len(series_b):
                        correlation = np.corrcoef(series_a, series_b)[0, 1]
                        integrated[f"corr_{series_names[i]}_{series_names[j]}"] = correlation
        
        return integrated
    
    def update_lambda3_memory(self, features: Dict):
        """Lambda³イベントメモリの更新"""
        event_dict = {}
        
        for key, values in features.items():
            if 'delta_LambdaC_pos' in key:
                series_name = key.split('_')[0]
                if series_name not in event_dict:
                    event_dict[series_name] = {}
                event_dict[series_name]['pos'] = np.sum(values) > 0
            elif 'delta_LambdaC_neg' in key:
                series_name = key.split('_')[0]
                if series_name not in event_dict:
                    event_dict[series_name] = {}
                event_dict[series_name]['neg'] = np.sum(values) > 0
        
        self.lambda3_memory.append({
            'timestamp': datetime.now(),
            'events': event_dict
        })
        
        # メモリサイズ制限
        if len(self.lambda3_memory) > 1000:
            self.lambda3_memory = self.lambda3_memory[-1000:]
    
    def analyze_causality(self, features: Dict) -> Dict:
        """因果性分析"""
        causality_features = {}
        
        if len(self.lambda3_memory) < 10:
            return causality_features
        
        # 単一系列内の因果性
        series_names = set([k.split('_')[0] for k in features.keys()])
        
        for series in series_names:
            # positive → negative パターンの検出
            pos_neg_count = 0
            for i in range(len(self.lambda3_memory) - 1):
                curr_event = self.lambda3_memory[i]['events'].get(series, {})
                next_event = self.lambda3_memory[i + 1]['events'].get(series, {})
                
                if curr_event.get('pos', False) and next_event.get('neg', False):
                    pos_neg_count += 1
            
            causality_features[f"{series}_pos_neg_causality"] = pos_neg_count / max(len(self.lambda3_memory) - 1, 1)
        
        return causality_features

# =====================================
# 3. 階層的ベイズ学習モジュール
# =====================================

class HierarchicalBayesianTrainer:
    """階層的ベイズ学習システム"""
    
    def __init__(self, config: OptimizedL3Config):
        self.config = config
        self.models = {
            'layer1_immediate': None,    # 1分: 高速ベイズ
            'layer2_fast': None,         # 1時間: 標準ベイズ
            'layer3_deep': None,         # 日次: 階層ベイズ
            'layer4_evolution': None     # 週次: モデル選択
        }
        self.last_traces = {}
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # JAXバックエンドの設定
        if config.use_jax:
            import jax
            import numpyro
            pm.sampling_jax.sample_numpyro_nuts()
    
    def train_immediate_layer(self, features: Dict, data: np.ndarray):
        """Layer 1: 即時ベイズ適応（1分）"""
        with mlflow.start_run(run_name="immediate_bayesian"):
            start_time = time.time()
            
            # 最小限のサンプル数で高速推論
            with pm.Model() as model:
                # 情報的事前分布（前回の結果を使用）
                if 'layer1' in self.last_traces:
                    trace = self.last_traces['layer1']
                    beta_0 = pm.Normal('beta_0', 
                                     mu=float(trace.posterior['beta_0'].mean()),
                                     sigma=float(trace.posterior['beta_0'].std()))
                else:
                    beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
                
                # シンプルな線形モデル
                mu = beta_0
                for feature_name, values in features.items():
                    if 'delta_LambdaC' in feature_name or 'rho_T' in feature_name:
                        beta = pm.Normal(f'beta_{feature_name}', mu=0, sigma=1)
                        mu += beta * values
                
                sigma = pm.HalfNormal('sigma', sigma=1)
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)
                
                # 高速サンプリング
                trace = pm.sample(
                    draws=200,
                    tune=100,
                    chains=2,
                    nuts_sampler="numpyro" if self.config.use_jax else None,
                    progressbar=False
                )
            
            self.last_traces['layer1'] = trace
            self.models['layer1_immediate'] = (model, trace)
            
            mlflow.log_metric("layer1_training_time", time.time() - start_time)
            mlflow.log_metric("layer1_rhat_max", float(az.rhat(trace).max()))
    
    def train_fast_layer(self, features: Dict, data: np.ndarray):
        """Layer 2: 高速ベイズ学習（1時間）"""
        with mlflow.start_run(run_name="fast_bayesian"):
            # 複数系列の相互作用を含むモデル
            with pm.Model() as model:
                # 基本パラメータ
                beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
                
                # 各系列のパラメータ
                mu = beta_0
                series_effects = {}
                
                for feature_name, values in features.items():
                    if any(key in feature_name for key in ['delta_LambdaC', 'rho_T', 'time_trend']):
                        series_name = feature_name.split('_')[0]
                        
                        # 系列固有効果
                        if series_name not in series_effects:
                            series_effects[series_name] = pm.Normal(
                                f'series_effect_{series_name}', mu=0, sigma=1
                            )
                        
                        beta = pm.Normal(f'beta_{feature_name}', mu=0, sigma=3)
                        mu += beta * values + series_effects[series_name]
                
                # 相互作用項
                if 'corr_' in str(features.keys()):
                    for key, value in features.items():
                        if key.startswith('corr_'):
                            beta_interaction = pm.Normal(f'beta_{key}', mu=0, sigma=2)
                            mu += beta_interaction * value
                
                sigma = pm.HalfNormal('sigma', sigma=1)
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)
                
                # 標準サンプリング
                trace = pm.sample(
                    draws=self.config.draws,
                    tune=self.config.tune,
                    chains=self.config.chains,
                    target_accept=self.config.target_accept,
                    nuts_sampler="numpyro" if self.config.use_jax else None,
                    progressbar=False
                )
            
            self.models['layer2_fast'] = (model, trace)
            mlflow.sklearn.log_model(model, "layer2_model")
    
    def train_deep_layer(self, features: Dict, data: pd.DataFrame):
        """Layer 3: 階層ベイズ（日次）"""
        with mlflow.start_run(run_name="hierarchical_bayesian"):
            # データの階層構造を抽出
            if 'department' in data.columns and 'user_id' in data.columns:
                dept_idx = pd.Categorical(data['department']).codes
                user_idx = pd.Categorical(data['user_id']).codes
                n_depts = len(data['department'].unique())
                n_users = len(data['user_id'].unique())
                
                with pm.Model() as model:
                    # ハイパープライヤー
                    mu_global = pm.Normal('mu_global', mu=0, sigma=5)
                    sigma_dept = pm.HalfNormal('sigma_dept', sigma=2)
                    sigma_user = pm.HalfNormal('sigma_user', sigma=1)
                    
                    # 階層構造
                    dept_effect = pm.Normal('dept_effect', mu=mu_global, 
                                          sigma=sigma_dept, shape=n_depts)
                    user_effect = pm.Normal('user_effect', mu=0, 
                                          sigma=sigma_user, shape=n_users)
                    
                    # Lambda³特徴量の効果
                    beta_lambda = {}
                    for feature_name in features:
                        if 'delta_LambdaC' in feature_name:
                            beta_lambda[feature_name] = pm.Normal(
                                f'beta_{feature_name}', mu=0, sigma=3
                            )
                    
                    # 線形予測子
                    mu = dept_effect[dept_idx] + user_effect[user_idx]
                    for feature_name, beta in beta_lambda.items():
                        if feature_name in data.columns:
                            mu += beta * data[feature_name].values
                    
                    # 観測モデル
                    sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, 
                                    observed=data['anomaly_score'].values)
                    
                    # MCMC
                    trace = pm.sample(
                        draws=2000,
                        tune=1000,
                        target_accept=0.9,
                        nuts_sampler="numpyro" if self.config.use_jax else None
                    )
                
                self.models['layer3_deep'] = (model, trace)
    
    def train_evolution_layer(self, features: Dict, data: pd.DataFrame):
        """Layer 4: モデル構造の進化（週次）"""
        # ベイズモデル選択
        models_to_compare = []
        
        # モデル1: シンプル線形
        models_to_compare.append(self._build_simple_model(features, data))
        
        # モデル2: 非線形変換
        models_to_compare.append(self._build_nonlinear_model(features, data))
        
        # モデル3: スパース回帰
        models_to_compare.append(self._build_sparse_model(features, data))
        
        # WAIC/LOOによるモデル比較
        best_model_idx = self._compare_models(models_to_compare)
        self.models['layer4_evolution'] = models_to_compare[best_model_idx]

# =====================================
# 4. A/Bテスト統合モジュール
# =====================================

class BayesianABTesting:
    """ベイズ的A/Bテスト"""
    
    def __init__(self, config: OptimizedL3Config):
        self.config = config
        self.test_results = []
    
    def run_bayesian_ab_test(self, model_a_trace, model_b_trace, test_data):
        """ベイズ的A/Bテスト"""
        # 事後予測分布からのサンプリング
        ppc_a = pm.sample_posterior_predictive(
            model_a_trace, 
            progressbar=False
        )
        ppc_b = pm.sample_posterior_predictive(
            model_b_trace,
            progressbar=False
        )
        
        # 性能指標の事後分布
        perf_a = self._calculate_performance_distribution(ppc_a, test_data)
        perf_b = self._calculate_performance_distribution(ppc_b, test_data)
        
        # ベイズ的比較
        prob_b_better = np.mean(perf_b > perf_a)
        expected_improvement = np.mean(perf_b - perf_a)
        
        result = {
            'timestamp': datetime.now(),
            'prob_b_better': prob_b_better,
            'expected_improvement': expected_improvement,
            'decision': 'deploy_b' if prob_b_better > 0.95 else 'keep_a'
        }
        
        self.test_results.append(result)
        return result

# =====================================
# 5. インテリジェントデプロイメント
# =====================================

class IntelligentDeployment:
    """ベイズ推論結果を活用したデプロイメント"""

    def __init__(self, cfg: OptimizedL3Config):
        self.config = cfg
        self.deployment_history = []
        
        # Kubernetes設定
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_apps = client.AppsV1Api()
    
    def deploy_with_uncertainty(self, model_trace, confidence_threshold=0.95):
        """不確実性を考慮したデプロイメント"""
        # モデルの不確実性を評価
        uncertainty = self._evaluate_model_uncertainty(model_trace)
        
        if uncertainty['confidence'] >= confidence_threshold:
            # 高信頼度: フルデプロイメント
            self.full_deployment(model_trace)
        elif uncertainty['confidence'] >= 0.8:
            # 中信頼度: カナリアデプロイメント
            self.canary_deployment(model_trace, canary_percentage=20)
        else:
            # 低信頼度: シャドウデプロイメント
            self.shadow_deployment(model_trace)
    
    def _evaluate_model_uncertainty(self, trace):
        """モデルの不確実性評価"""
        # パラメータの収束診断
        rhat_values = az.rhat(trace)
        max_rhat = float(rhat_values.max())
        
        # 有効サンプルサイズ
        ess_values = az.ess(trace)
        min_ess = float(ess_values.min())
        
        # 事後分布の分散
        posterior_std = float(trace.posterior.std())
        
        # 総合的な信頼度スコア
        confidence = 1.0
        if max_rhat > 1.01:
            confidence *= 0.8
        if min_ess < 400:
            confidence *= 0.9
        if posterior_std > 2.0:
            confidence *= 0.85
        
        return {
            'confidence': confidence,
            'max_rhat': max_rhat,
            'min_ess': min_ess,
            'posterior_std': posterior_std
        }

# =====================================
# 6. 統合MLOpsパイプライン
# =====================================

class IntegratedMLOpsPipeline:
    """完全統合MLOpsパイプライン"""
    
    def __init__(self, config: OptimizedL3Config = None):
        if config is None:
            config = OptimizedL3Config()
        
        self.config = config
        self.data_collector = EnhancedDataCollector(config)
        self.preprocessor = Lambda3DataPreprocessor(config)
        self.trainer = HierarchicalBayesianTrainer(config)
        self.ab_tester = BayesianABTesting(config)
        self.deployer = IntelligentDeployment(config)
        self.diversity_cluster = DiversityAwareCluster()
        
        # メトリクス記録
        self.metrics_history = []
    
    async def run_pipeline(self):
        """パイプライン全体の実行"""
        # 非同期タスクの起動
        tasks = [
            asyncio.create_task(self.data_collector.collect_security_events()),
            asyncio.create_task(self.monitor_performance()),
            asyncio.create_task(self.health_check_loop())
        ]
        
        # メインループ
        while True:
            try:
                # 1分ごとの処理
                await self.minute_cycle()
                
                # 1時間ごとの処理
                if datetime.now().minute == 0:
                    await self.hourly_cycle()
                
                # 日次処理
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    await self.daily_cycle()
                
                # 週次処理
                if datetime.now().weekday() == 0 and datetime.now().hour == 0:
                    await self.weekly_cycle()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.handle_pipeline_error(e)
    
    async def minute_cycle(self):
        """1分サイクル: 即時適応"""
        # 最新データの取得
        features, data = await self.get_latest_features()
        
        if features and len(data) >= self.config.min_batch_size:
            # Layer 1: 高速ベイズ学習
            self.trainer.train_immediate_layer(features, data)
            
            # 即時予測と異常検知
            predictions = self.predict_anomalies(features)
            
            # メトリクス記録
            self.record_metrics({
                'timestamp': datetime.now(),
                'predictions': predictions,
                'data_size': len(data)
            })
    
    async def hourly_cycle(self):
        """1時間サイクル: 標準学習とA/Bテスト"""
        # 過去1時間のデータ集約
        features, data = await self.aggregate_hourly_data()
        
        # Layer 2: 標準ベイズ学習
        self.trainer.train_fast_layer(features, data)
        
        # A/Bテストの実行
        if self.should_run_ab_test():
            old_model = self.trainer.models.get('layer2_fast')
            new_model = self.trainer.models.get('layer1_immediate')
            
            if old_model and new_model:
                test_result = self.ab_tester.run_bayesian_ab_test(
                    old_model[1], new_model[1], data
                )
                
                if test_result['decision'] == 'deploy_b':
                    await self.deploy_model(new_model)
    
    async def daily_cycle(self):
        """日次サイクル: 階層ベイズ学習＋敵対的多様性進化"""
        # 直近24h異常＆正常サンプルの抽出
        recent_anomalies = self.get_recent_anomalies(hours=24)
        normal_samples = self.get_recent_normal_samples(hours=24)
    
        adv_variants = []
        if recent_anomalies and normal_samples:
            adv_def = AdversarialDefense(normal_samples)
            for atk in recent_anomalies:
                adv_variants += adv_def.generate_attack_variations(atk, n_variations=20)
            # 多様性クラスタに反映
            for ns in normal_samples:
                self.diversity_cluster.add_sample(ns["user_id"], ns, np.array([ns["score"]]))
            for av in adv_variants:
                self.diversity_cluster.add_sample(av["user_id"], av, np.array([av["score"]]))
            # 必要ならrecluster
            for user in set([e["user_id"] for e in normal_samples + adv_variants]):
                if self.diversity_cluster.need_recluster(user):
                    self.diversity_cluster.recluster(user)
            # ※↓学習用データに変種・正常を追加する場合はここでdaily_dataにappendする
        
        # ---日次学習---
        daily_data = await self.prepare_daily_data()
        # ここでadv_variants等も含めて「1日分の学習データ」としてdaily_data['dataframe']に
        self.trainer.train_deep_layer(
            daily_data['features'],
            daily_data['dataframe']
        )
        evaluation = self.evaluate_model_performance()
        if evaluation['improvement'] > 0.05:
            model_trace = self.trainer.models['layer3_deep'][1]
            self.deployer.deploy_with_uncertainty(
                model_trace,
                confidence_threshold=0.9
            )
                
    async def weekly_cycle(self):
        """週次サイクル: モデル構造の進化"""
        # 週次データ集約
        weekly_data = await self.prepare_weekly_data()
        
        # Layer 4: モデル進化
        self.trainer.train_evolution_layer(
            weekly_data['features'],
            weekly_data['dataframe']
        )
        
        # アーキテクチャ更新
        await self.update_architecture()

    def on_security_event(self, event):
        # 1. ローカル即時適応
        if hasattr(self, "kafka_producer"):
            producer = self.kafka_producer
        else:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            producer = self.kafka_producer
    
        if event.is_anomaly():
            auto_minibatch.add_event(event)
            producer.send('anomaly-topic', event.to_json())
            policy_manager.update_defense(event)
    
    def handle_pipeline_error(self, error: Exception):
        """エラーハンドリング"""
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'pipeline_state': self.get_pipeline_state()
        }
        
        # エラーログ
        mlflow.log_metrics({
            'error_count': 1,
            'error_timestamp': time.time()
        })
        
        # 復旧処理
        if isinstance(error, MemoryError):
            self.cleanup_memory()
        elif isinstance(error, TimeoutError):
            self.reset_connections()
        
        # アラート送信
        self.send_alert(error_info)

# =====================================
# 実行エントリーポイント
# =====================================

async def main():
    """メイン実行関数"""
    # 設定の初期化
    config = OptimizedL3Config(
        draws=500,
        tune=200,
        chains=2,
        use_jax=True,
        incremental_update=True,
        kafka_servers=['localhost:9092'],
        min_batch_size=100
    )
    
    # パイプラインの起動
    pipeline = IntegratedMLOpsPipeline(config)
    
    print("🚀 統合MLOps + Lambda³ベイズ推論パイプライン起動")
    print(f"設定: JAX={config.use_jax}, Draws={config.draws}, Chains={config.chains}")
    
    # 非同期実行
    await pipeline.run_pipeline()

if __name__ == "__main__":
    # イベントループの実行
    asyncio.run(main())
