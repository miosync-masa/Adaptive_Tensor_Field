from main import DepartmentSecurityChainManager, SecurityEventChain, UserSecurityChain
from IntelligenceChainModule import ThreatIntelligenceChain
from Keystroke_event import KeystrokeAuthenticator, ContinuousAuthenticationSystem
from auto_minibatch import AdaptiveMiniBatchKMeans
from incidentsupport import IntegratedIncidentResponseSystem
from mlo_pipeline import IntegratedMLOpsPipeline
from diversity_adversarial_defense import AdversarialDefense, DiversityAwareCluster

class UnifiedSecuritySystem:
    """全モジュールを統合したセキュリティシステム"""
    
    def __init__(self):
        # 1. メインチェーンマネージャー
        self.chain_manager = DepartmentSecurityChainManager()
        
        # 2. 脅威インテリジェンス
        self.threat_intel = ThreatIntelligenceChain()
        
        # 3. キーストローク認証
        self.keystroke_auth = self._init_keystroke_auth()
        
        # 4. 自動学習
        self.auto_learner = AdaptiveMiniBatchKMeans()
        
        # 5. インシデント対応
        self.incident_response = IntegratedIncidentResponseSystem(self.chain_manager)
        
        # 6. MLOpsパイプライン
        self.mlops = IntegratedMLOpsPipeline()
        
        # 7. 敵対的防御
        self.adversarial_defense = AdversarialDefense()
        self.diversity_cluster = DiversityAwareCluster()
