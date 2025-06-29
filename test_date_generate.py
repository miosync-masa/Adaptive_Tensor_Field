import json

def generate_suspicious_events():
    """
    60件のリアルな混合シナリオ（LanScope Cat形式）
    2025年3月14日〜15日の2日間を通したストーリー
    """
    events = [
        # 3月14日の朝 - 通常の業務開始
        {
            "timestamp": "2025-03-14 08:30:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-YAMADA",
            "expected": "normal",
            "scenario": "営業部の正常な朝のログイン"
        },
        {
            "timestamp": "2025-03-14 08:35:00",
            "user_id": "sato_k",
            "department": "engineering",
            "source_ip": "192.168.2.50",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-SATO",
            "expected": "normal",
            "scenario": "エンジニアの通常ログイン"
        },
        {
            "timestamp": "2025-03-14 08:40:00",
            "user_id": "kato_m",
            "department": "hr",
            "source_ip": "192.168.4.20",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-KATO",
            "expected": "normal",
            "scenario": "人事部の通常ログイン"
        },
        {
            "timestamp": "2025-03-14 08:45:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-SUZUKI",
            "expected": "normal",
            "scenario": "攻撃者となるユーザーの正常ログイン"
        },
        {
            "timestamp": "2025-03-14 08:50:00",
            "user_id": "ito_h",
            "department": "finance",
            "source_ip": "192.168.3.30",
            "operation": "ProcessCreate",
            "process_name": "outlook.exe",
            "process_path": "C:\\Program Files\\Microsoft Office\\outlook.exe",
            "expected": "normal",
            "scenario": "経理部の朝のメールチェック"
        },
        {
            "timestamp": "2025-03-14 09:00:00",
            "user_id": "takahashi_y",
            "department": "engineering",
            "source_ip": "192.168.2.55",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\engineering\\source_code\\project_alpha\\main.py",
            "file_size_kb": 45,
            "process_name": "vscode.exe",
            "expected": "normal",
            "scenario": "開発者のコード編集開始"
        },
        {
            "timestamp": "2025-03-14 09:15:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\sales\\sales_reports\\2025\\03\\daily_report_20250314.xlsx",
            "file_size_kb": 125,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "営業活動の開始"
        },
        {
            "timestamp": "2025-03-14 09:19:00",
            "user_id": "unknown",
            "source_ip": "198.51.100.10",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.1.1",
            "destination_port": 22,
            "process_name": "nmap.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "外部からのポートスキャン",
            "pattern_id": "ZDP001",
            "pattern_name": "外部ポートスキャン"
        },
        {
            "timestamp": "2025-03-14 09:20:00",
            "user_id": "nakamura_r",
            "department": "finance",
            "source_ip": "192.168.3.35",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\finance\\accounting_system\\ledger_2025.xlsx",
            "file_size_kb": 1250,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "経理部の会計処理"
        },
        {
            "timestamp": "2025-03-14 09:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\sales\\sales_reports\\2025\\03\\sales_summary.xlsx",
            "file_size_kb": 450,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "攻撃者の通常業務（カモフラージュ）"
        },
        {
            "timestamp": "2025-03-14 09:35:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\sales\\customer_contacts\\full_customer_list.csv",
            "file_size_kb": 15000,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "顧客データベースの列挙"
        },
        {
            "timestamp": "2025-03-14 09:36:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileCopy",
            "file_path": "\\\\fileserver\\sales\\customer_contacts\\full_customer_list.csv",
            "file_size_kb": 45000,
            "process_name": "explorer.exe",
            "expected": "suspicious",
            "scenario": "顧客データベースのダウンロード"
        },
        {
            "timestamp": "2025-03-14 09:45:00",
            "user_id": "kato_m",
            "department": "hr",
            "source_ip": "192.168.4.20",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\hr\\employee_records\\attendance_march.xlsx",
            "file_size_kb": 890,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "人事部の勤怠管理"
        },
        {
            "timestamp": "2025-03-14 10:00:00",
            "user_id": "watanabe_n",
            "department": "engineering",
            "source_ip": "192.168.2.60",
            "operation": "ProcessCreate",
            "process_name": "docker.exe",
            "process_path": "C:\\Program Files\\Docker\\Docker\\resources\\docker.exe",
            "expected": "normal",
            "scenario": "エンジニアのDocker環境構築"
        },
        {
            "timestamp": "2025-03-14 10:10:00",
            "user_id": "tanaka_s",
            "department": "sales",
            "source_ip": "192.168.1.105",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\sales\\customer_contacts\\CustomerList.xlsx",
            "file_size_kb": 850,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "他の営業の正常活動"
        },
        {
            "timestamp": "2025-03-14 10:14:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\engineering\\source_code\\",
            "process_name": "explorer.exe",
            "status": "FAILED",
            "expected": "investigating",
            "scenario": "営業が開発リソースにアクセス試行"
        },
        {
            "timestamp": "2025-03-14 10:15:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\hr\\payroll_system\\",
            "process_name": "explorer.exe",
            "status": "FAILED",
            "expected": "suspicious",
            "scenario": "営業が人事リソースにアクセス試行"
        },
        {
            "timestamp": "2025-03-14 10:18:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\shared\\confidential\\",
            "process_name": "explorer.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "confidentialにアクセス試行"
        },
        {
            "timestamp": "2025-03-14 10:20:00",
            "user_id": "honda_m",
            "department": "engineering",
            "source_ip": "192.168.2.65",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\engineering\\documentation\\API_v2.0.md",
            "file_size_kb": 45,
            "process_name": "notepad++.exe",
            "expected": "normal",
            "scenario": "エンジニアの正常な文書更新"
        },
        {
            "timestamp": "2025-03-14 10:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.10.1",
            "destination_port": 3389,
            "process_name": "mstsc.exe",
            "status": "FAILED",
            "expected": "suspicious",
            "scenario": "管理画面への複数回のアクセス失敗1"
        },
        {
            "timestamp": "2025-03-14 10:30:10",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.10.1",
            "destination_port": 3389,
            "process_name": "mstsc.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "管理画面への複数回のアクセス失敗2"
        },
        {
            "timestamp": "2025-03-14 10:30:15",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.10.1",
            "destination_port": 3389,
            "process_name": "mstsc.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "管理画面への複数回のアクセス失敗3"
        },
        {
            "timestamp": "2025-03-14 10:35:00",
            "user_id": "yoshida_k",
            "department": "hr",
            "source_ip": "192.168.4.25",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\hr\\payroll_system\\salary_march.xlsx",
            "file_size_kb": 560,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "給与計算業務"
        },
        {
            "timestamp": "2025-03-14 10:45:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\hr\\employee_records\\",
            "process_name": "explorer.exe",
            "expected": "suspicious",
            "scenario": "権限回避してHRシステムにアクセス"
        },
        {
            "timestamp": "2025-03-14 11:00:00",
            "user_id": "ito_a",
            "department": "sales",
            "source_ip": "192.168.1.110",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\sales\\proposal_docs\\proposal_XYZ_corp.docx",
            "file_size_kb": 340,
            "process_name": "WINWORD.EXE",
            "expected": "normal",
            "scenario": "提案書作成"
        },
        {
            "timestamp": "2025-03-14 11:15:00",
            "user_id": "kobayashi_t",
            "department": "finance",
            "source_ip": "192.168.3.40",
            "operation": "NetworkConnect",
            "destination_ip": "172.20.10.15",# bank.example.com
            "destination_port": 443,
            "process_name": "chrome.exe",
            "expected": "normal",
            "scenario": "銀行ポータルへのアクセス"
        },
        {
            "timestamp": "2025-03-14 11:30:00",
            "user_id": "kimura_t",
            "department": "engineering",
            "source_ip": "192.168.2.70",
            "operation": "ProcessCreate",
            "process_name": "git.exe",
            "process_path": "C:\\Program Files\\Git\\cmd\\git.exe",
            "expected": "normal",
            "scenario": "ソースコードのコミット"
        },
        {
            "timestamp": "2025-03-14 11:50:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-YAMADA",
            "expected": "normal",
            "scenario": "昼休み前の一時ログアウト"
        },
        {
            "timestamp": "2025-03-14 12:25:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileCopy",
            "file_path": "\\\\fileserver\\finance\\financial_reports\\2025\\confidential\\financial_summary.xlsx",
            "file_size_kb": 50000,
            "process_name": "explorer.exe",
            "expected": "critical",
            "scenario": "財務レポートへの大量アクセス"
        },
        {
            "timestamp": "2025-03-14 13:00:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-YAMADA",
            "expected": "normal",
            "scenario": "昼休み後の再ログイン"
        },
        {
            "timestamp": "2025-03-14 13:05:00",
            "user_id": "fujita_k",
            "department": "finance",
            "source_ip": "192.168.3.45",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\finance\\tax_system\\tax_calculation_2025.xlsx",
            "file_size_kb": 780,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "税務計算作業"
        },
        {
            "timestamp": "2025-03-14 13:15:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "ProcessCreate",
            "process_name": "powershell.exe",
            "process_path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            "expected": "critical",
            "scenario": "管理者権限の不正取得"
        },
        {
            "timestamp": "2025-03-14 13:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileWrite",
            "file_path": "C:\\Windows\\System32\\config\\",
            "process_name": "cmd.exe",
            "expected": "critical",
            "scenario": "バックドアアカウントの作成"
        },
        {
            "timestamp": "2025-03-14 13:35:00",
            "user_id": "ishida_j",
            "department": "hr",
            "source_ip": "192.168.4.30",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\hr\\recruitment_tool\\candidates_2025.xlsx",
            "file_size_kb": 450,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "採用候補者の確認"
        },
        {
            "timestamp": "2025-03-14 13:45:00",
            "user_id": "sato_y",
            "department": "sales",
            "source_ip": "192.168.1.115",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.100.10", #crm.company.com
            "destination_port": 443,
            "process_name": "firefox.exe",
            "expected": "normal",
            "scenario": "CRMシステムへのアクセス"
        },
        {
            "timestamp": "2025-03-14 14:00:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileCopy",
            "file_path": "\\\\fileserver\\sales\\customer_contacts\\*.csv",
            "file_size_kb": 500000,
            "process_name": "robocopy.exe",
            "expected": "critical",
            "scenario": "顧客データの大量ダウンロード"
        },
        {
            "timestamp": "2025-03-14 14:15:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "NetworkConnect",
            "destination_ip": "203.0.113.50",
            "destination_port": 443,
            "process_name": "chrome.exe",
            "expected": "critical",
            "scenario": "外部サーバーへのデータ送信"
        },
        {
            "timestamp": "2025-03-14 14:20:00",
            "user_id": "ogawa_s",
            "department": "finance",
            "source_ip": "192.168.3.50",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\finance\\budget_tool\\budget_q2_2025.xlsx",
            "file_size_kb": 890,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "四半期予算の作成"
        },
        {
            "timestamp": "2025-03-14 14:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileDelete",
            "file_path": "C:\\Windows\\System32\\winevt\\Logs\\Security.evtx",
            "process_name": "wevtutil.exe",
            "expected": "critical",
            "scenario": "証拠隠滅のためのログ削除試行"
        },
        {
            "timestamp": "2025-03-14 14:45:00",
            "user_id": "hayashi_r",
            "department": "hr",
            "source_ip": "192.168.4.35",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\hr\\training_portal\\training_schedule.xlsx",
            "file_size_kb": 120,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "研修スケジュールの確認"
        },
        {
            "timestamp": "2025-03-14 15:00:00",
            "user_id": "takahashi_y",
            "department": "engineering",
            "source_ip": "192.168.2.55",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.2.200",
            "destination_port": 8080,
            "process_name": "java.exe",
            "expected": "normal",
            "scenario": "テスト環境へのデプロイ"
        },
        {
            "timestamp": "2025-03-14 15:10:00",
            "user_id": "unknown",
            "source_ip": "198.51.100.10",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.1.1",
            "destination_port": 22,
            "process_name": "nmap.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "外部からのポートスキャン再試行"
        },
        {
            "timestamp": "2025-03-14 15:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\admin\\config\\firewall_rules.xml",
            "process_name": "notepad.exe",
            "expected": "critical",
            "scenario": "ファイアウォール設定の変更"
        },
        {
            "timestamp": "2025-03-14 15:45:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.120",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.100.10",
            "destination_port": 445,
            "process_name": "system",
            "expected": "critical",
            "scenario": "バックアップサーバーへの横展開"
        },
        {
            "timestamp": "2025-03-14 16:00:00",
            "user_id": "morita_n",
            "department": "hr",
            "source_ip": "192.168.4.40",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\hr\\hr_system\\performance_review_q1.xlsx",
            "file_size_kb": 340,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "人事評価データの更新"
        },
        {
            "timestamp": "2025-03-14 16:30:00",
            "user_id": "watanabe_n",
            "department": "engineering",
            "source_ip": "192.168.2.60",
            "operation": "ProcessCreate",
            "process_name": "jenkins.exe",
            "process_path": "C:\\Program Files\\Jenkins\\jenkins.exe",
            "expected": "normal",
            "scenario": "CI/CDパイプラインの実行"
        },
        {
            "timestamp": "2025-03-14 17:00:00",
            "user_id": "tanaka_s",
            "department": "sales",
            "source_ip": "192.168.1.105",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\sales\\contract_system\\contract_ABC_signed.pdf",
            "file_size_kb": 890,
            "process_name": "AcroRd32.exe",
            "expected": "normal",
            "scenario": "契約書の保存"
        },
        {
            "timestamp": "2025-03-14 18:00:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-YAMADA",
            "expected": "normal",
            "scenario": "正常な終業ログアウト"
        },
        {
            "timestamp": "2025-03-14 18:15:00",
            "user_id": "ito_h",
            "department": "finance",
            "source_ip": "192.168.3.30",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-ITO",
            "expected": "normal",
            "scenario": "経理部の終業"
        },
        {
            "timestamp": "2025-03-14 18:30:00",
            "user_id": "kato_m",
            "department": "hr",
            "source_ip": "192.168.4.20",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-KATO",
            "expected": "normal",
            "scenario": "人事部の終業"
        },
        {
            "timestamp": "2025-03-14 19:30:00",
            "user_id": "sato_k",
            "department": "engineering",
            "source_ip": "192.168.2.50",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-SATO",
            "expected": "normal",
            "scenario": "エンジニアの遅めの退社"
        },
        {
            "timestamp": "2025-03-15 08:30:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-YAMADA",
            "expected": "normal",
            "scenario": "正常な朝のログイン"
        },
        {
            "timestamp": "2025-03-15 09:00:00",
            "user_id": "nakamura_r",
            "department": "finance",
            "source_ip": "192.168.3.35",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-NAKAMURA",
            "expected": "normal",
            "scenario": "経理部のログイン"
        },
        {
            "timestamp": "2025-03-15 09:30:00",
            "user_id": "honda_m",
            "department": "engineering",
            "source_ip": "192.168.2.65",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-HONDA",
            "expected": "normal",
            "scenario": "エンジニアのログイン"
        },
        {
            "timestamp": "2025-03-15 10:00:00",
            "user_id": "yoshida_k",
            "department": "hr",
            "source_ip": "192.168.4.25",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-YOSHIDA",
            "expected": "normal",
            "scenario": "人事部のログイン"
        },
        {
            "timestamp": "2025-03-15 11:00:00",
            "user_id": "ito_a",
            "department": "sales",
            "source_ip": "192.168.1.110",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\sales\\crm_system\\pipeline_report.xlsx",
            "file_size_kb": 560,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "営業パイプラインの確認"
        },
        {
            "timestamp": "2025-03-15 12:00:00",
            "user_id": "kimura_t",
            "department": "engineering",
            "source_ip": "192.168.2.70",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\engineering\\test_environment\\test_results_20250315.log",
            "file_size_kb": 230,
            "process_name": "python.exe",
            "expected": "normal",
            "scenario": "テスト結果の記録"
        },
        {
            "timestamp": "2025-03-15 13:00:00",
            "user_id": "unknown",
            "source_ip": "198.51.100.10",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-SUZUKI",
            "destination_ip": "192.168.1.1",
            "destination_port": 22,
            "process_name": "nmap.exe",
            "expected": "critical",
            "scenario": "外部からの不正login"
        },
        {
            "timestamp": "2025-03-15 14:00:00",
            "user_id": "kobayashi_t",
            "department": "finance",
            "source_ip": "192.168.3.40",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\finance\\financial_reports\\2025\\03\\monthly_close.xlsx",
            "file_size_kb": 4500,
            "process_name": "EXCEL.EXE",
            "expected": "normal",
            "scenario": "月次決算作業"
        },
        {
            "timestamp": "2025-03-15 15:00:00",
            "user_id": "suzuki_m",
            "source_ip": "198.51.100.10",
            "workstation_name": "PC-SUZUKI",
            "operation": "NetworkConnect",
            "destination_ip": "203.0.113.200",
            "destination_port": 80,
            "process_name": "chrome.exe",
            "expected": "critical",
            "scenario": "機密情報の最終的な外部送信"
        },
        {
            "timestamp": "2025-03-15 15:15:00",
            "user_id": "suzuki_m",
            "source_ip": "198.51.100.10",
            "workstation_name": "PC-SUZUKI",
            "operation": "FileDelete",
            "file_path": "C:\\Windows\\System32\\winevt\\Logs\\Security.evtx",
            "process_name": "wevtutil.exe",
            "expected": "critical",
            "scenario": "証拠隠滅のためWindowsイベントログ削除"
        },
        {
            "timestamp": "2025-03-15 15:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "198.51.100.10",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-SUZUKI",
            "expected": "critical",
            "scenario": "攻撃者が痕跡を消しログアウト"
        },
        {
            "timestamp": "2025-03-15 17:00:00",
            "user_id": "sato_y",
            "department": "sales",
            "source_ip": "192.168.1.115",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-SATO-Y",
            "expected": "normal",
            "scenario": "営業部の終業"
        },
        {
            "timestamp": "2025-03-15 18:00:00",
            "user_id": "fujita_k",
            "department": "finance",
            "source_ip": "192.168.3.45",
            "operation": "Logout",
            "status": "SUCCESS",
            "workstation_name": "PC-FUJITA",
            "expected": "normal",
            "scenario": "経理部の終業"
        }
    ]

    return events

def generate_zero_day_attack_events():
    """
    ゼロデイ攻撃シミュレーションイベント（LanScope Cat形式）
    """
    return [
        {
            "timestamp": "2025-03-12 22:10:00",
            "user_id": "unknown",
            "source_ip": "198.51.100.10",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.1.1",
            "destination_port": 22,
            "process_name": "nmap.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "外部からのポートスキャン" # システムには渡してない
        },
        {
            "timestamp": "2025-03-13 03:15:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "198.51.100.10",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "UNKNOWN",
            "expected": "critical",
            "scenario": "外部IPからの不正ログイン" # システムには渡してない
        },
        {
            "timestamp": "2025-03-13 03:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.200",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\engineering\\source\\config\\database.properties",
            "file_size_kb": 5,
            "process_name": "notepad.exe",
            "expected": "critical",
            "scenario": "営業が開発サーバーの設定ファイルにアクセス" # システムには渡してない
        },
        {
            "timestamp": "2025-03-13 03:45:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.200",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\engineering\\source\\main.jsp",
            "file_size_kb": 10,
            "process_name": "cmd.exe",
            "expected": "critical",
            "scenario": "営業がソースコードにバックドアを仕込む" # システムには渡してない
        },
        {
            "timestamp": "2025-03-13 04:00:00",
            "user_id": "system",
            "source_ip": "192.168.1.200",
            "operation": "FileCopy",
            "file_path": "\\\\fileserver\\sales\\customers\\full_database_export.sql",
            "file_size_kb": 2000000,
            "process_name": "sqlcmd.exe",
            "expected": "critical",
            "scenario": "大規模なデータベース流出" # システムには渡してない
        },
        {
            "timestamp": "2025-03-12 22:10:00",
            "user_id": "unknown",
            "source_ip": "198.51.100.10",
            "operation": "NetworkConnect",
            "destination_ip": "192.168.1.1",
            "destination_port": 22,
            "process_name": "nmap.exe",
            "status": "FAILED",
            "expected": "critical",
            "scenario": "外部からのポートスキャン",
            "pattern_id": "ZDP001",
            "pattern_name": "外部ポートスキャン"
        },
        {
            "timestamp": "2025-03-13 03:15:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "198.51.100.10",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "UNKNOWN",
            "expected": "critical",
            "scenario": "外部IPからの不正ログイン",
            "pattern_id": "ZDP002",
            "pattern_name": "外部不正ログイン"
        },
        {
            "timestamp": "2025-03-13 03:30:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.200",
            "operation": "FileRead",
            "file_path": "\\\\fileserver\\engineering\\source\\config\\database.properties",
            "file_size_kb": 5,
            "process_name": "notepad.exe",
            "expected": "critical",
            "scenario": "営業が開発サーバーの設定ファイルにアクセス",
            "pattern_id": "ZDP003",
            "pattern_name": "権限逸脱ファイルアクセス"
        },
        {
            "timestamp": "2025-03-13 03:45:00",
            "user_id": "suzuki_m",
            "department": "sales",
            "source_ip": "192.168.1.200",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\engineering\\source\\main.jsp",
            "file_size_kb": 10,
            "process_name": "cmd.exe",
            "expected": "critical",
            "scenario": "営業がソースコードにバックドアを仕込む",
            "pattern_id": "ZDP004",
            "pattern_name": "バックドア埋め込み"
        },
        {
            "timestamp": "2025-03-13 04:00:00",
            "user_id": "system",
            "source_ip": "192.168.1.200",
            "operation": "FileCopy",
            "file_path": "\\\\fileserver\\sales\\customers\\full_database_export.sql",
            "file_size_kb": 2000000,
            "process_name": "sqlcmd.exe",
            "expected": "critical",
            "scenario": "大規模なデータベース流出",
            "pattern_id": "ZDP005",
            "pattern_name": "大規模データ持ち出し"
        }
    ]

def generate_normal_test_events():
    """
    誤検知チェック用の正常イベント（LanScope Cat形式）
    """
    return [
        {
            "timestamp": "2025-03-03 09:00:00",
            "user_id": "yamada_t",
            "department": "sales",
            "source_ip": "192.168.1.100",
            "operation": "Login",
            "status": "SUCCESS",
            "workstation_name": "PC-YAMADA",
            "scenario": "通常の朝のログイン"
        },
        {
            "timestamp": "2025-03-03 10:30:00",
            "user_id": "sato_k",
            "department": "engineering",
            "source_ip": "192.168.2.50",
            "operation": "FileRead",
            "file_path": "C:\\repos\\project\\README.md",
            "file_size_kb": 8,
            "process_name": "Code.exe",
            "scenario": "開発ドキュメントの参照"
        },
        {
            "timestamp": "2025-03-03 14:15:00",
            "user_id": "ito_h",
            "department": "finance",
            "source_ip": "192.168.3.30",
            "operation": "FileWrite",
            "file_path": "\\\\fileserver\\finance\\reports\\2025\\03\\daily_report.xlsx",
            "file_size_kb": 125,
            "process_name": "EXCEL.EXE",
            "scenario": "日次レポートの更新"
        }
    ]

def generate_attack_patterns():
    """
    様々な攻撃パターンのテンプレート
    """
    return {
        "insider_threat": {
            "name": "内部脅威",
            "description": "正規ユーザーによる不正行為",
            "indicators": [
                "通常とは異なる時間帯のアクセス",
                "権限外リソースへのアクセス",
                "大量データのダウンロード",
                "ログ削除の試み"
            ]
        },
        "account_takeover": {
            "name": "アカウント乗っ取り",
            "description": "外部からの不正ログイン",
            "indicators": [
                "異常なIPアドレスからのログイン",
                "短時間での大量アクセス",
                "権限昇格の試み",
                "システム設定の変更"
            ]
        },
        "data_exfiltration": {
            "name": "データ流出",
            "description": "機密データの外部送信",
            "indicators": [
                "大量データアクセス",
                "外部サーバーへの通信",
                "圧縮ファイルの作成",
                "暗号化通信の急増"
            ]
        },
        "lateral_movement": {
            "name": "横展開",
            "description": "ネットワーク内での侵害拡大",
            "indicators": [
                "複数システムへの連続アクセス",
                "管理者権限での探索",
                "新規アカウントの作成",
                "バックドアの設置"
            ]
        }
    }

def dump_events_to_json():
    with open("chain_attackments.json", "w", encoding="utf-8") as f:
        json.dump(generate_suspicious_events(), f, ensure_ascii=False, indent=2)
    with open("attackments.json", "w", encoding="utf-8") as f:
        json.dump(generate_zero_day_attack_events(), f, ensure_ascii=False, indent=2)
    with open("normal_test_events.json", "w", encoding="utf-8") as f:
        json.dump(generate_normal_test_events(), f, ensure_ascii=False, indent=2)
    with open("attack_patterns.json", "w", encoding="utf-8") as f:
        json.dump(generate_attack_patterns(), f, ensure_ascii=False, indent=2)

def load_events_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    dump_events_to_json()
    print("JSONファイルへの書き出しが完了！")
