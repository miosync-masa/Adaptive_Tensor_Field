import json
import random
from datetime import datetime, timedelta
import numpy as np

# 部署とユーザーの定義
DEPARTMENTS = {
    "sales": {
        "users": ["tanaka_s", "yamada_t", "suzuki_m", "ito_a", "sato_y"],
        "resources": ["crm_system", "sales_reports", "customer_contacts", "proposal_docs", "contract_system"],
        "work_hours": (9, 19),
        "seasonal_factors": {
            "quarter_end": 1.5,
            "year_end": 2.0,
            "new_year": 0.7,
            "summer": 0.9,
        }
    },
    "engineering": {
        "users": ["sato_k", "takahashi_y", "watanabe_n", "honda_m", "kimura_t"],
        "resources": ["source_code", "dev_servers", "ci_cd_pipeline", "test_environment", "documentation"],
        "work_hours": (10, 20),
        "seasonal_factors": {
            "quarter_end": 1.2,
            "year_end": 1.3,
            "new_year": 1.1,
            "summer": 1.0,
        }
    },
    "finance": {
        "users": ["ito_h", "nakamura_r", "kobayashi_t", "fujita_k", "ogawa_s"],
        "resources": ["accounting_system", "financial_reports", "bank_portal", "tax_system", "budget_tool"],
        "work_hours": (9, 18),
        "seasonal_factors": {
            "quarter_end": 2.0,
            "year_end": 3.0,
            "new_year": 0.8,
            "summer": 0.9,
        }
    },
    "hr": {
        "users": ["kato_m", "yoshida_k", "ishida_j", "hayashi_r", "morita_n"],
        "resources": ["hr_system", "employee_records", "payroll_system", "recruitment_tool", "training_portal"],
        "work_hours": (9, 18),
        "seasonal_factors": {
            "quarter_end": 1.1,
            "year_end": 1.5,
            "new_year": 2.0,
            "summer": 0.8,
        }
    }
}


# ビジネスイベントの定義（既存のものを使用）
BUSINESS_EVENTS = {
    "quarter_end": {
        "months": [3, 6, 9, 12],
        "duration_days": 10,
        "overtime_probability": 0.7,
        "weekend_work_probability": 0.3,
    },
    "year_end": {
        "months": [3],
        "duration_days": 20,
        "overtime_probability": 0.9,
        "weekend_work_probability": 0.5,
    },
    "new_fiscal_year": {
        "months": [4],
        "duration_days": 15,
        "overtime_probability": 0.5,
        "weekend_work_probability": 0.2,
    },
    "golden_week": {
        "dates": [(5, 1, 7)],
        "work_probability": 0.1,
    },
    "obon": {
        "dates": [(8, 13, 16)],
        "work_probability": 0.15,
    },
    "year_end_holiday": {
        "dates": [(12, 29, 31), (1, 1, 3)],
        "work_probability": 0.05,
    }
}

# ファイルパスパターンの定義（新規追加）
FILE_PATH_PATTERNS = {
    "sales": [
        "\\\\fileserver\\sales\\reports\\{year}\\{month}\\daily_report_{date}.xlsx",
        "\\\\fileserver\\sales\\customers\\{company}\\contract_{id}.pdf",
        "\\\\fileserver\\sales\\proposals\\{year}Q{quarter}\\proposal_{client}.docx",
        "C:\\Users\\{user}\\Documents\\営業資料\\見積書_{date}.xlsx",
        "\\\\fileserver\\sales\\meeting_notes\\{year}\\{month}\\meeting_{date}.docx"
    ],
    "engineering": [
        "C:\\repos\\project\\src\\main\\{module}\\{file}.java",
        "\\\\devserver\\builds\\{version}\\output\\app.exe",
        "C:\\Users\\{user}\\Documents\\design\\architecture_{component}.md",
        "\\\\fileserver\\tech\\documentation\\api_spec_{version}.pdf",
        "C:\\repos\\project\\tests\\{module}\\test_{file}.java"
    ],
    "finance": [
        "\\\\fileserver\\finance\\reports\\{year}\\{month}\\月次決算_{date}.xlsx",
        "\\\\fileserver\\finance\\invoices\\{year}\\{vendor}\\invoice_{number}.pdf",
        "\\\\fileserver\\finance\\budget\\{year}\\部門別予算_{dept}.xlsx",
        "C:\\Users\\{user}\\Documents\\経理\\仕訳_{date}.csv",
        "\\\\fileserver\\finance\\tax\\{year}\\tax_report_{quarter}.xlsx"
    ],
    "hr": [
        "\\\\fileserver\\hr\\employees\\{emp_id}\\personnel_record.xlsx",
        "\\\\fileserver\\hr\\payroll\\{year}\\{month}\\給与明細_{emp_id}.pdf",
        "\\\\fileserver\\hr\\recruitment\\{year}\\応募者_{candidate_id}.docx",
        "C:\\Users\\{user}\\Documents\\人事\\評価シート_{year}.xlsx",
        "\\\\fileserver\\hr\\training\\{year}\\training_record_{emp_id}.xlsx"
    ]
}

def expand_users(base_names, n):
    users = []
    for base in base_names:
        for i in range(1, n+1):
            users.append(f"{base}_{i:02d}")
    return users

NUM_USERS_PER_DEPT = 50

for dept, info in DEPARTMENTS.items():
    info["users"] = expand_users(info["users"], NUM_USERS_PER_DEPT)
    print(f"{dept}: {len(info['users'])} users")

def is_business_event(date):
    """特定の日付がビジネスイベントに該当するか判定"""
    events = []

    if date.month in [3, 6, 9, 12] and date.day >= 20:
        events.append("quarter_end")

    if date.month == 3 and date.day >= 10:
        events.append("year_end")

    if date.month == 4 and date.day <= 15:
        events.append("new_fiscal_year")

    for event_name, event_info in BUSINESS_EVENTS.items():
        if "dates" in event_info:
            for month, start_day, end_day in event_info["dates"]:
                if date.month == month and start_day <= date.day <= end_day:
                    events.append(event_name)

    return events

def create_realistic_security_event(
    user, dept, operation, event_time,
    file_path=None, file_size_kb=None,
    process_name=None, destination_ip=None,
    status="SUCCESS", business_context=None,
    workstation_name=None,   # ←これもOK
    **kwargs                # ←ここ絶対追加！
):
    """
    LanScope Cat風の現実的なセキュリティイベントを生成
    """
    ip_ranges = {
        "sales": "192.168.1.",
        "engineering": "192.168.2.",
        "finance": "192.168.3.",
        "hr": "192.168.4."
    }

    source_ip = ip_ranges.get(dept, "192.168.1.") + str(random.randint(10, 250))

    # 基本イベント構造
    event = {
        "timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user,
        "department": dept,
        "source_ip": source_ip,
        "operation": operation,
        "status": status
    }

    # 操作タイプに応じた追加フィールド
    if operation in ["FileRead", "FileWrite", "FileCopy", "FileDelete", "FileMove"]:
        event["file_path"] = file_path
        event["file_size_kb"] = file_size_kb
        event["process_name"] = process_name or "explorer.exe"

    elif operation in ["ProcessCreate", "ProcessTerminate"]:
        event["process_name"] = process_name
        event["process_path"] = f"C:\\Program Files\\{process_name}"

    elif operation in ["NetworkConnect", "NetworkListen"]:
        event["destination_ip"] = destination_ip
        event["destination_port"] = random.choice([80, 443, 3389, 1433, 3306])
        event["process_name"] = process_name or "chrome.exe"

    elif operation in ["Login", "Logout", "LoginFailed"]:
        event["authentication_type"] = "Password"
        event["workstation_name"] = f"PC-{user.upper()}"

    if operation in ["Login", "Logout", "LoginFailed"]:
        event["authentication_type"] = "Password"
        # ↓ ここで外部から渡されたらそれを使う
        event["workstation_name"] = workstation_name or f"PC-{user.upper()}"

    # ビジネスコンテキストは内部的に保持
    if business_context:
        event["_business_context"] = business_context

    event.update(kwargs)
    return event

def generate_file_path(dept, user, date, path_type="work"):
    """実際のファイルパスを生成"""
    path_template = random.choice(FILE_PATH_PATTERNS.get(dept, FILE_PATH_PATTERNS["sales"]))

    file_path = path_template.format(
        year=date.year,
        month=f"{date.month:02d}",
        quarter=(date.month-1)//3 + 1,
        date=date.strftime("%Y%m%d"),
        user=user,
        company=f"Company{random.randint(1, 100)}",
        client=f"Client{random.randint(1, 50)}",
        vendor=f"Vendor{random.randint(1, 30)}",
        emp_id=f"EMP{random.randint(1000, 9999)}",
        candidate_id=f"CND{random.randint(100, 999)}",
        number=f"INV{random.randint(10000, 99999)}",
        id=random.randint(1000, 9999),
        version=f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
        module=random.choice(["controller", "service", "repository", "model"]),
        file=f"Class{random.randint(1, 20)}",
        component=random.choice(["auth", "payment", "user", "report"]),
        dept=dept
    )

    return file_path

def get_file_size_kb(file_path, activity_factor=1.0):
    """ファイルパスから適切なファイルサイズを推定"""
    if ".xlsx" in file_path or ".csv" in file_path:
        size = random.randint(50, 5000)
    elif ".pdf" in file_path:
        size = random.randint(100, 2000)
    elif ".docx" in file_path:
        size = random.randint(30, 500)
    elif ".exe" in file_path:
        size = random.randint(5000, 50000)
    elif ".java" in file_path or ".md" in file_path:
        size = random.randint(1, 100)
    else:
        size = random.randint(10, 1000)

    # 繁忙期は大きなファイルを扱うことが多い
    if activity_factor > 1.5:
        size = int(size * 1.5)

    return size

def get_process_name(file_path):
    """ファイルパスから適切なプロセス名を返す"""
    if ".xlsx" in file_path or ".csv" in file_path:
        return "EXCEL.EXE"
    elif ".pdf" in file_path:
        return random.choice(["AcroRd32.exe", "FoxitReader.exe"])
    elif ".docx" in file_path:
        return "WINWORD.EXE"
    elif ".java" in file_path or ".md" in file_path:
        return random.choice(["Code.exe", "idea64.exe", "eclipse.exe"])
    else:
        return "explorer.exe"

def generate_realistic_work_events(user, dept, date, start_hour, end_hour,
                                 activity_factor=1.0, business_context=None):
    """
    現実的な業務ログを生成（LanScope Cat形式）
    """
    events = []
    business_context = business_context or []

    # ファイルアクセスイベント
    base_accesses = random.randint(15, 25)
    num_accesses = int(base_accesses * activity_factor)

    for _ in range(num_accesses):
        hour = random.randint(start_hour, end_hour-1)
        minute = random.randint(0, 59)
        event_time = date.replace(hour=hour, minute=minute)

        file_path = generate_file_path(dept, user, date)
        file_size_kb = get_file_size_kb(file_path, activity_factor)
        process_name = get_process_name(file_path)

        events.append(create_realistic_security_event(
            user, dept, "FileRead", event_time,
            file_path=file_path,
            file_size_kb=file_size_kb,
            process_name=process_name,
            business_context=business_context
        ))

    # ファイル更新イベント
    base_updates = random.randint(3, 8)
    num_updates = int(base_updates * activity_factor)

    for _ in range(num_updates):
        hour = random.randint(start_hour, end_hour-1)
        minute = random.randint(0, 59)
        event_time = date.replace(hour=hour, minute=minute)

        file_path = generate_file_path(dept, user, date)
        file_size_kb = get_file_size_kb(file_path, activity_factor)
        process_name = get_process_name(file_path)

        events.append(create_realistic_security_event(
            user, dept, "FileWrite", event_time,
            file_path=file_path,
            file_size_kb=file_size_kb,
            process_name=process_name,
            business_context=business_context
        ))

    # ファイルコピーイベント（たまに）
    if random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            hour = random.randint(start_hour, end_hour-1)
            minute = random.randint(0, 59)
            event_time = date.replace(hour=hour, minute=minute)

            file_path = generate_file_path(dept, user, date)
            file_size_kb = get_file_size_kb(file_path, activity_factor)

            events.append(create_realistic_security_event(
                user, dept, "FileCopy", event_time,
                file_path=file_path,
                file_size_kb=file_size_kb,
                process_name="explorer.exe",
                business_context=business_context
            ))

    # ネットワークアクセス
    for _ in range(random.randint(5, 10)):
        hour = random.randint(start_hour, end_hour-1)
        minute = random.randint(0, 59)
        event_time = date.replace(hour=hour, minute=minute)

        # 通常は内部サーバー
        if random.random() < 0.8:
            destination_ip = f"192.168.{random.randint(10, 20)}.{random.randint(1, 250)}"
        else:
            # たまに外部
            destination_ip = random.choice([
                "142.250.185.14",  # google.com
                "13.107.42.14",    # microsoft.com
                "151.101.1.140",   # stackoverflow.com
                "52.84.228.25",    # aws
            ])

        events.append(create_realistic_security_event(
            user, dept, "NetworkConnect", event_time,
            destination_ip=destination_ip,
            process_name=random.choice(["chrome.exe", "firefox.exe", "msedge.exe", "outlook.exe"]),
            business_context=business_context
        ))

    # プロセス起動（業務アプリケーション）
    apps = {
        "sales": ["EXCEL.EXE", "WINWORD.EXE", "outlook.exe", "teams.exe"],
        "engineering": ["Code.exe", "idea64.exe", "docker.exe", "git.exe"],
        "finance": ["EXCEL.EXE", "SAP.exe", "calculator.exe", "outlook.exe", "git.exe"],
        "hr": ["EXCEL.EXE", "WINWORD.EXE", "HRSystem.exe", "outlook.exe"]
    }

    for _ in range(random.randint(2, 5)):
        hour = random.randint(start_hour, start_hour+1)  # 朝に集中
        minute = random.randint(0, 59)
        event_time = date.replace(hour=hour, minute=minute)

        process_name = random.choice(apps.get(dept, apps["sales"]))

        events.append(create_realistic_security_event(
            user, dept, "ProcessCreate", event_time,
            process_name=process_name,
            business_context=business_context
        ))

    return sorted(events, key=lambda x: x["timestamp"])

def generate_enhanced_security_logs(days=500):
    """
    LanScope Cat形式の現実的なセキュリティログを生成（長期/大量/部門横断）
    """
    logs = []
    start_date = datetime(2024, 4, 1, 0, 0, 0)
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        current_events = is_business_event(current_date)

        # --- 1. 長期休暇（GW・年末年始など） ---
        if _is_long_holiday(current_events):
            logs += _generate_holiday_logs(current_date, current_events)
            continue

        # --- 2. 週末（土日） ---
        if current_date.weekday() >= 5:
            logs += _generate_weekend_logs(current_date, current_events)
            continue

        # --- 3. 平日業務 ---
        logs += _generate_weekday_logs(current_date, current_events)
    return logs

def _is_long_holiday(current_events):
    return any(event in current_events for event in ["golden_week", "obon", "year_end_holiday"])

def _generate_holiday_logs(current_date, current_events):
    logs = []
    for dept_name, dept_info in DEPARTMENTS.items():
        work_prob = max([BUSINESS_EVENTS.get(event, {}).get("work_probability", 0.1) for event in current_events], default=0.1)
        if random.random() < work_prob:
            user = random.choice(dept_info["users"])
            login_time = current_date.replace(hour=10, minute=random.randint(0, 59))
            logs.append(create_realistic_security_event(
                user, dept_name, "Login", login_time,
                business_context=["holiday_work"]
            ))
            work_events = generate_realistic_work_events(
                user, dept_name, current_date,
                10, 12, activity_factor=0.3, business_context=["holiday_work"]
            )
            logs.extend(work_events[:3])  # 最大3件
            logout_time = login_time + timedelta(hours=2)
            logs.append(create_realistic_security_event(
                user, dept_name, "Logout", logout_time,
                business_context=["holiday_work"]
            ))
    return logs

def _generate_weekend_logs(current_date, current_events):
    logs = []
    for dept_name, dept_info in DEPARTMENTS.items():
        # イベント起因の週末稼働率反映
        weekend_work_prob = max([BUSINESS_EVENTS.get(event, {}).get("weekend_work_probability", 0.1) for event in current_events], default=0.1)
        dept_factor = max([dept_info.get("seasonal_factors", {}).get(event, 1.0) for event in current_events], default=1.0)
        if random.random() < weekend_work_prob * dept_factor:
            num_workers = max(1, int(len(dept_info["users"]) * 0.3))
            workers = random.sample(dept_info["users"], num_workers)
            for user in workers:
                start_hour = random.randint(10, 14)
                work_hours = random.randint(2, 4) if "year_end" not in current_events else random.randint(4, 8)
                login_time = current_date.replace(hour=start_hour, minute=random.randint(0, 59))
                logs.append(create_realistic_security_event(
                    user, dept_name, "Login", login_time,
                    business_context=["weekend_work"] + current_events
                ))
                work_events = generate_realistic_work_events(
                    user, dept_name, current_date,
                    start_hour, start_hour + work_hours,
                    activity_factor=0.7, business_context=["weekend_work"] + current_events
                )
                logs.extend(work_events)
                logout_time = login_time + timedelta(hours=work_hours)
                logs.append(create_realistic_security_event(
                    user, dept_name, "Logout", logout_time,
                    business_context=["weekend_work"] + current_events
                ))
    return logs

def _generate_weekday_logs(current_date, current_events):
    logs = []
    for dept_name, dept_info in DEPARTMENTS.items():
        activity_factor = max([dept_info.get("seasonal_factors", {}).get(event, 1.0) for event in current_events], default=1.0)
        for user in dept_info["users"]:
            start_hour, end_hour = dept_info["work_hours"]
            if activity_factor > 1.3:
                start_hour -= 1
                end_hour += 2
            elif activity_factor > 1.0:
                end_hour += 1

            actual_start = start_hour + random.normalvariate(0, 0.5)
            actual_end = end_hour + random.normalvariate(0, 0.5)

            # ログイン
            login_time = current_date.replace(
                hour=int(max(6, min(23, actual_start))),
                minute=random.randint(0, 59)
            )
            # 低確率で認証失敗も混ぜる
            if random.random() < 0.01:
                failed_time = login_time - timedelta(minutes=random.randint(1, 5))
                logs.append(create_realistic_security_event(
                    user, dept_name, "LoginFailed", failed_time,
                    status="FAILED",
                    business_context=current_events
                ))
            logs.append(create_realistic_security_event(
                user, dept_name, "Login", login_time,
                business_context=current_events
            ))
            # 業務アクセス
            work_events = generate_realistic_work_events(
                user, dept_name, current_date,
                int(actual_start), int(actual_end),
                activity_factor=activity_factor,
                business_context=current_events
            )
            logs.extend(work_events)
            # ログアウト
            logout_time = current_date.replace(
                hour=int(max(6, min(23, actual_end))),
                minute=random.randint(0, 59)
            )
            logs.append(create_realistic_security_event(
                user, dept_name, "Logout", logout_time,
                business_context=current_events
            ))
    return logs

# 時間帯別パターン生成（既存の関数を現実的な形式に変換）
def generate_morning_patterns(dept_name):
    """朝の典型的なパターンを生成（LanScope Cat形式）"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns

    base_date = datetime(2025, 1, 15, 0, 0, 0)

    for user in dept_info["users"]:
        # ログイン
        login_time = base_date.replace(hour=random.randint(8, 10), minute=random.randint(0, 59))
        patterns.append(create_realistic_security_event(
            user, dept_name, "Login", login_time
        ))

        # 朝のメールチェック
        patterns.append(create_realistic_security_event(
            user, dept_name, "ProcessCreate",
            login_time + timedelta(minutes=5),
            process_name="outlook.exe"
        ))

        # 最初のファイルアクセス
        file_path = generate_file_path(dept_name, user, base_date)
        patterns.append(create_realistic_security_event(
            user, dept_name, "FileRead",
            login_time + timedelta(minutes=15),
            file_path=file_path,
            file_size_kb=random.randint(50, 500),
            process_name=get_process_name(file_path)
        ))

    return patterns

def generate_noon_patterns(dept_name):
    """昼の典型的なパターンを生成（LanScope Cat形式）"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns

    base_date = datetime(2025, 1, 15, 0, 0, 0)

    for user in dept_info["users"]:
        # 昼の定常業務
        for hour in range(12, 15):
            event_time = base_date.replace(hour=hour, minute=random.randint(0, 59))

            # ファイルアクセス
            file_path = generate_file_path(dept_name, user, base_date)
            patterns.append(create_realistic_security_event(
                user, dept_name, random.choice(["FileRead", "FileWrite"]),
                event_time,
                file_path=file_path,
                file_size_kb=get_file_size_kb(file_path),
                process_name=get_process_name(file_path)
            ))

            # ネットワークアクセス
            patterns.append(create_realistic_security_event(
                user, dept_name, "NetworkConnect",
                event_time + timedelta(minutes=random.randint(5, 20)),
                destination_ip=f"192.168.{random.randint(10, 20)}.{random.randint(1, 250)}",
                process_name="chrome.exe"
            ))

    return patterns

def generate_evening_patterns(dept_name):
    """夕方の典型的なパターンを生成（LanScope Cat形式）"""
    patterns = []
    dept_info = DEPARTMENTS.get(dept_name)
    if not dept_info:
        return patterns

    base_date = datetime(2025, 1, 15, 0, 0, 0)

    for user in dept_info["users"]:
        # 夕方の作業
        work_time = base_date.replace(hour=random.randint(17, 19), minute=random.randint(0, 59))

        # 最終更新
        file_path = generate_file_path(dept_name, user, base_date)
        patterns.append(create_realistic_security_event(
            user, dept_name, "FileWrite",
            work_time,
            file_path=file_path,
            file_size_kb=get_file_size_kb(file_path),
            process_name=get_process_name(file_path)
        ))

        # ログアウト
        patterns.append(create_realistic_security_event(
            user, dept_name, "Logout",
            work_time + timedelta(minutes=random.randint(10, 30))
        ))

    return patterns

# ===============================================
#  ATTACK‑TEMPLATE ENGINE : generate_abnormal_events
# ===============================================

def _rand_ts(base_day: datetime, night_bias: float = 0.7) -> datetime:
    """
    深夜帯出現確率 night_bias (0‑1) を加味してタイムスタンプを返す
    """
    if random.random() < night_bias:                      # 深夜寄り
        hour = random.choice([0, 1, 2, 3, 4, 23])
    else:                                                 # 日中～夕方の逸脱
        hour = random.choice([10, 12, 15, 18, 20])
    minute = random.randint(0, 59)
    return base_day.replace(hour=hour, minute=minute,
                            second=random.randint(0, 59))

# --- “攻撃テンプレ” —— 必要に応じて _attack_templates にレシピを追加 ---
_attack_templates = [
    # 0️⃣  機密ファイル大量コピー（Powershell 経由で HR→Sales など）
    dict(
        name="massive_confidential_copy",
        operation="FileCopy",
        level="critical",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda depts, frm: "hr" if frm != "hr" else "finance",
        extra=lambda kw: kw.update(
            file_size_kb=1_000_000,
            process_name="powershell.exe",
            business_context=["cross_department", "massive_file", "unusual_process"]
        )
    ),
        dict(
        name="confidential_access_attempt",
        operation="FileRead",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda depts, frm: "shared",
        extra=lambda kw: kw.update(
            file_path="\\\\fileserver\\shared\\confidential\\",
            process_name="explorer.exe",
            status="FAILED",
            business_context=["confidential_access", "access_denied", "unusual_process"]
        )
    ),
        dict(
        name="unauthorized_admin_shell",
        operation="ProcessCreate",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            process_name="powershell.exe",
            process_path="C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            business_context=["privilege_escalation", "admin_operation", "unusual_process"]
        )
    ),
        dict(
        name="log_deletion_evidence",
        operation="FileDelete",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            file_path="C:\\Windows\\System32\\winevt\\Logs\\Security.evtx",
            process_name="wevtutil.exe",
            business_context=["evidence_removal", "log_delete", "unusual_process"]
        )
    ),
        dict(
        name="unauthorized_firewall_change",
        operation="FileWrite",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: "admin",
        extra=lambda kw: kw.update(
            file_path="\\\\fileserver\\admin\\config\\firewall_rules.xml",
            process_name="notepad.exe",
            business_context=["config_change", "firewall", "unauthorized_modification"]
        )
    ),
        dict(
        name="lateral_movement_backup_server",
        operation="NetworkConnect",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            destination_ip="192.168.100.10",
            destination_port=445,
            process_name="system",
            business_context=["lateral_movement", "backup_server", "unusual_connection"]
        )
    ),
    dict(
        name="massive_saleslist_copy",
        operation="FileCopy",
        level="suspicious",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda depts, frm: "sales",
        extra=lambda kw: kw.update(
            file_size_kb=random.choice([250_000, 300_000, 120_000]),  # 100MB〜300MBくらい
            process_name=random.choice(["explorer.exe", "robocopy.exe"]),
            business_context=["massive_file", "unusual_process", "exfiltration"]
        )
    ),
    # 1️⃣  外部 IP への不審通信（nmap / unknown proc）
    dict(
        name="external_scan",
        operation="NetworkConnect",
        level="suspicious",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            destination_ip=random.choice(["8.8.8.8", "1.1.1.1", "203.0.113.50"]),
            process_name=random.choice(["nmap.exe", "unknown.exe"]),
            business_context=["external_ip", "unusual_process"]
        )
    ),
    # 2️⃣  管理者権限の昇格コマンド失敗
    dict(
        name="privilege_escalation_failure",
        operation="ProcessCreate",
        level="critical",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            process_name="cmd.exe",
            status="FAILED",
            business_context=["privilege_escalation", "unusual_process"]
        )
    ),
    # 3️⃣  深夜の Payroll システム直接書き換え（HR 以外のユーザ）
    dict(
        name="unauthorised_payroll_write",
        operation="FileWrite",
        level="critical",
        choose_from_dept=lambda depts: random.choice([d for d in depts if d != "hr"]),
        choose_to_dept=lambda *_: "hr",
        extra=lambda kw: kw.update(
            file_size_kb=random.choice([120_000, 250_000]),
            process_name="excel.exe",
            business_context=["cross_department", "outside_work"]
        )
    ),
    dict(
        name="massive_hr_data_read",
        operation="FileRead",
        level="critical",
        choose_from_dept=lambda depts: random.choice([d for d in depts if d != "hr"]),
        choose_to_dept=lambda *_: "hr",
        extra=lambda kw: kw.update(
            file_size_kb=500_000,
            process_name="robocopy.exe",
            business_context=["cross_department", "massive_file"]
        )
    ),
    dict(
        name="weekend_admin_shell",
        operation="ProcessCreate",
        level="critical",
        choose_from_dept=lambda depts: "engineering",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            process_name="powershell.exe",
            business_context=["weekend", "outside_work", "admin_operation"]
        )
    ),
    dict(
        name="vendor_file_access",
        operation="FileRead",
        level="suspicious",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: "finance",
        extra=lambda kw: kw.update(
            file_size_kb=120_000,
            process_name="explorer.exe",
            business_context=["cross_department", "vendor_access"]
        )
    ),
    dict(
        name="log_delete_evidence",
        operation="FileDelete",
        level="critical",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda *_: "engineering",
        extra=lambda kw: kw.update(
            file_path="C:\\Windows\\System32\\winevt\\Logs\\Security.evtx",
            process_name="wevtutil.exe",
            business_context=["evidence_removal", "unusual_process"]
        )
    ),
    dict(
        name="unknown_user_login",
        operation="Login",
        level="suspicious",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            status="SUCCESS",
            business_context=["unknown_actor", "endpoint_compromise"],
            workstation_name="PC-UNKNOWN"
        )
    ),
    dict(
        name="burst_file_read",
        operation="FileRead",
        level="suspicious",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda *_: "sales",
        extra=lambda kw: kw.update(
            file_size_kb=50000,
            process_name="python.exe",
            business_context=["burst_access", "unusual_process", "possible_DOS"]
        )
    ),
    dict(
        name="external_mail_send",
        operation="ProcessCreate",
        level="suspicious",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            process_name="outlook.exe",
            business_context=["external_comm", "data_exfiltration"]
        )
    ),
    dict(
        name="external_data_exfil",
        operation="NetworkConnect",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            destination_ip="198.51.100.99",
            process_name="outlook.exe",
            business_context=["external_ip", "data_exfiltration"]
        )
    ),
    dict(
        name="domain_admin_add",
        operation="ProcessCreate",
        level="critical",
        choose_from_dept=lambda depts: "engineering",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            process_name="cmd.exe",
            business_context=["domain_admin_add", "privilege_escalation", "critical"]
        )
    ),
    dict(
        name="malware_dropper_exec",
        operation="ProcessCreate",
        level="critical",
        choose_from_dept=lambda depts: random.choice(depts),
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            process_name="unknown.exe",
            business_context=["malware_execution", "persistence", "critical"]
        )
    ),
    dict(
        name="external_backup_copy",
        operation="FileCopy",
        level="critical",
        choose_from_dept=lambda depts: "finance",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            file_size_kb=2_000_000,
            process_name="robocopy.exe",
            destination_ip="203.0.113.50",
            business_context=["external_backup", "massive_file", "critical"]
        )
    ),
    dict(
        name="mass_email_attachment_exfil",
        operation="FileRead",
        level="critical",
        choose_from_dept=lambda depts: "sales",
        choose_to_dept=lambda *_: None,
        extra=lambda kw: kw.update(
            file_size_kb=250_000,
            process_name="outlook.exe",
            business_context=["email_attachment", "data_exfiltration", "critical"]
        )
    ),
    # …テンプレをさらに追加したければここに辞書を追記 …
]

def generate_abnormal_events(
    num_critical: int = 1500,
    num_other: int = 500,
    departments: dict = DEPARTMENTS,
    base_day: datetime = datetime(2025, 4, 1)
) -> list[dict]:
    events: list[dict] = []
    depts = list(departments.keys())

    critical_templates = [tpl for tpl in _attack_templates if tpl.get("level") == "critical"]
    other_templates    = [tpl for tpl in _attack_templates if tpl.get("level") != "critical"]

    # --- Critical優遇 ---
    for _ in range(num_critical):
        tpl = random.choice(critical_templates)

        # --- 発生部署 & ユーザー決定 ---------------------
        from_dept = tpl["choose_from_dept"](depts)
        user_list = departments[from_dept]["users"] + ["unKnown"]
        user = random.choice(user_list)

        # --- 参照（または侵入）先部署 --------------------
        to_dept   = tpl["choose_to_dept"](depts, from_dept) or from_dept

        # --- ファイルパスなど共通パラメータ ---------------
        kwargs = {}
        if tpl["operation"] in ("FileRead", "FileWrite", "FileCopy", "FileDelete"):
            kwargs["file_path"] = generate_file_path(to_dept, user, base_day)
        # デフォルト値をセット
        kwargs.setdefault("file_size_kb", random.randint(100, 5_000))
        kwargs.setdefault("status", "SUCCESS")

        # --- テンプレ固有の追加加工 -----------------------
        tpl["extra"](kwargs)

        # --- タイムスタンプ ------------------------------
        ts = _rand_ts(base_day)

        # --- イベント生成 -------------------------------
        ev = create_realistic_security_event(
            user, from_dept, tpl["operation"], ts, **kwargs
        )
        # テンプレ名をメタとして残しておくと後で便利
        ev["_attack_template"] = tpl["name"]
        events.append(ev)
      # --- その他 ---
    for _ in range(num_other):
        tpl = random.choice(other_templates)

        # --- 発生部署 & ユーザー決定 ---------------------
        from_dept = tpl["choose_from_dept"](depts)
        user_list = departments[from_dept]["users"] + ["unKnown"]
        user = random.choice(user_list)

        # --- 参照（または侵入）先部署 --------------------
        to_dept   = tpl["choose_to_dept"](depts, from_dept) or from_dept

        # --- ファイルパスなど共通パラメータ ---------------
        kwargs = {}
        if tpl["operation"] in ("FileRead", "FileWrite", "FileCopy", "FileDelete"):
            kwargs["file_path"] = generate_file_path(to_dept, user, base_day)
        # デフォルト値をセット
        kwargs.setdefault("file_size_kb", random.randint(100, 5_000))
        kwargs.setdefault("status", "SUCCESS")

        # --- テンプレ固有の追加加工 -----------------------
        tpl["extra"](kwargs)

        # --- タイムスタンプ ------------------------------
        ts = _rand_ts(base_day)

        # --- イベント生成 -------------------------------
        ev = create_realistic_security_event(
            user, from_dept, tpl["operation"], ts, **kwargs
        )
        # テンプレ名をメタとして残しておくと後で便利
        ev["_attack_template"] = tpl["name"]
        events.append(ev)

    return sorted(events, key=lambda x: x["timestamp"])

if __name__ == "__main__":
    # 🐾 異常のみ 1,000 件生成して JSON へ
    abnormal_logs = generate_abnormal_events(num_critical=1500, num_other=500)
    with open("attack_only_logs.json", "w", encoding="utf-8") as f:
        json.dump(abnormal_logs, f, ensure_ascii=False, indent=2)

    print(f"✔️  attack_only_logs.json に {len(abnormal_logs)} 件書き出しました")


if __name__ == "__main__":
    print("🚀 現実的なセキュリティログを生成中...")
    print("  - LanScope Cat形式")
    print("  - 実際のファイルパス")
    print("  - リアルな操作ログ")
    print("  - 季節変動・ビジネスイベント対応")

    # 500日分のログを生成
    all_logs = generate_enhanced_security_logs(days=500)

    # ランダムに選択
    random.shuffle(all_logs)
    selected_logs = all_logs[:3500]

    # 時系列でソート
    selected_logs.sort(key=lambda x: x["timestamp"])

    # ファイルに保存
    with open("realistic_security_logs.json", "w", encoding="utf-8") as f:
        json.dump(selected_logs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(selected_logs)}件の現実的なログを生成しました。")

    # 統計情報
    operation_counts = {}
    dept_counts = {}
    file_extensions = {}
    process_counts = {}

    for log in selected_logs:
        # 操作タイプ別
        op = log.get("operation", "unknown")
        operation_counts[op] = operation_counts.get(op, 0) + 1

        # 部署別
        dept = log.get("department", "unknown")
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

        # ファイル拡張子別
        if "file_path" in log:
            ext = log["file_path"].split(".")[-1].lower()
            file_extensions[ext] = file_extensions.get(ext, 0) + 1

        # プロセス別
        if "process_name" in log:
            process = log["process_name"]
            process_counts[process] = process_counts.get(process, 0) + 1

    print("\n📊 操作タイプ別:")
    for op, count in sorted(operation_counts.items()):
        print(f"  {op}: {count}件")

    print("\n📊 部署別ログ数:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept}: {count}件")

    print("\n📊 ファイル拡張子別TOP10:")
    for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  .{ext}: {count}件")

    print("\n📊 プロセス別TOP10:")
    for process, count in sorted(process_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {process}: {count}件")

    # 月別の分布
    month_counts = {}
    for log in selected_logs:
        month = log["timestamp"][:7]
        month_counts[month] = month_counts.get(month, 0) + 1

    print("\n📊 月別ログ数:")
    for month, count in sorted(month_counts.items()):
        print(f"  {month}: {count}件")

    # ビジネスコンテキスト
    context_counts = {}
    for log in selected_logs:
        if "_business_context" in log:
            for context in log["_business_context"]:
                context_counts[context] = context_counts.get(context, 0) + 1

    if context_counts:
        print("\n📊 ビジネスコンテキスト別:")
        for context, count in sorted(context_counts.items()):
            print(f"  {context}: {count}件")
