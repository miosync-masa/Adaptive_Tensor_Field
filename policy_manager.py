class PolicyManager:
    def __init__(self):
        # ここでAPI設定や認証情報、内部ルールDB等セットアップ
        pass

    def update_defense(self, event):
        if 'src_ip' in event:
            print(f"[POLICY] Blocking src_ip: {event['src_ip']}")
            # EDRやFirewall APIにここでリクエスト送信
        if 'user_id' in event:
            print(f"[POLICY] Suspending user: {event['user_id']}")
        # その他 event の属性にもどんどん対応OK
