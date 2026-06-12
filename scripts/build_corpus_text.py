#!/usr/bin/env python3
"""Generate a zh-TW / en code-mixed corpus for the phone-attendant vocoder distillation.

Coverage targets (the product's defining requirement):
  - English names inside Chinese sentences ("幫您轉接給 Kevin 陳經理")
  - product / tech terms ("USB", "AI 語音助理", "VPN", "Wi-Fi")
  - digits / extensions / phone numbers ("分機 533")
  - receptionist fixed lines (greetings, transfers, hold, voicemail)

Deterministic given --seed (no Date/random surprises across reruns). One line per utterance:
    <id>\t<text>
"""
import argparse, random, sys

# ---- slot banks -------------------------------------------------------------
EN_NAMES = [
    "Kevin", "Amy", "Jason", "Linda", "Michael", "Sophia", "David", "Emily",
    "Brian", "Cindy", "Eric", "Grace", "Frank", "Helen", "Jack", "Karen",
    "Leo", "Nancy", "Oscar", "Peter", "Rita", "Sam", "Tina", "Victor",
    "Wendy", "Alex", "Betty", "Charlie", "Daniel", "Ivy", "Jerry", "Kelly",
    "Steve", "Vivian", "Andy", "Catherine", "Henry", "Joyce", "Tony", "Sandra",
]
SURNAMES = list("陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴周葉蘇莊呂江")
TITLES = ["經理", "副理", "協理", "工程師", "專員", "主任", "課長", "小姐",
          "先生", "襄理", "業務", "顧問", "組長", "助理"]
DEPTS = ["業務部", "技術部", "客服部", "財務部", "人資部", "採購部", "研發部",
         "行銷部", "資訊部", "品保部", "法務部", "總務部"]
TECH = ["USB", "VPN", "Wi-Fi", "AI 語音助理", "email", "PDF 報價單", "API 介面",
        "ERP 系統", "CRM 系統", "Outlook", "Teams 會議", "Zoom 連結", "SSD 硬碟",
        "Excel 報表", "Line 群組", "QR Code", "OTP 驗證碼", "Server 主機"]
COMPANIES = ["宏達電子", "台灣精密", "全球科技", "鼎新資訊", "聯華實業",
             "佳世達", "凌華科技", "華碩客服中心", "中華電信商辦"]

def num(rng, lo, hi):
    return str(rng.randint(lo, hi))

def ext(rng):           # 分機 3-4 digits, sometimes spaced
    n = rng.choice([3, 3, 4])
    return "".join(num(rng, 0, 9) for _ in range(n))

def phone(rng):         # 市話 / 手機
    if rng.random() < 0.5:
        return f"0{rng.randint(2,8)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
    return f"09{rng.randint(10,99)}-{rng.randint(100,999)}-{rng.randint(100,999)}"

def person(rng):
    return f"{rng.choice(EN_NAMES)} {rng.choice(SURNAMES)}{rng.choice(TITLES)}"

def zh_person(rng):
    return f"{rng.choice(SURNAMES)}{rng.choice(TITLES)}"

# ---- templates --------------------------------------------------------------
# Each is a callable(rng)->str. Mix of fixed receptionist lines and slotted ones.
TEMPLATES = [
    lambda r: f"您好,這裡是{r.choice(COMPANIES)},很高興為您服務。",
    lambda r: f"早安,歡迎來電,請問有什麼可以為您服務的嗎?",
    lambda r: f"幫您轉接給 {person(r)},請稍候。",
    lambda r: f"幫您轉接給 {person(r)},他的分機是 {ext(r)}。",
    lambda r: f"{person(r)} 現在不在位子上,需要幫您留言嗎?",
    lambda r: f"{r.choice(EN_NAMES)} 目前正在開會,大約 {num(r,5,40)} 分鐘後回來。",
    lambda r: f"請問您要找哪一位?是 {r.choice(EN_NAMES)} 還是 {r.choice(EN_NAMES)}?",
    lambda r: f"您撥的分機 {ext(r)} 忙線中,請稍後再撥。",
    lambda r: f"這是 {r.choice(COMPANIES)} 的語音信箱,請在嗶聲後留言。",
    lambda r: f"您好,{r.choice(DEPTS)}的 {person(r)} 為您服務。",
    lambda r: f"關於您的 {r.choice(TECH)} 問題,我幫您轉到{r.choice(DEPTS)}。",
    lambda r: f"請問您的訂單編號是?方便提供一下嗎?編號是 {num(r,100000,999999)}。",
    lambda r: f"我們已經把 {r.choice(TECH)} 寄到您的信箱,請查收。",
    lambda r: f"{person(r)} 請您回電,電話是 {phone(r)}。",
    lambda r: f"您的{r.choice(TECH)}設定已完成,如有問題請撥分機 {ext(r)}。",
    lambda r: f"請問是 {r.choice(EN_NAMES)} 先生嗎?{zh_person(r)}要找您。",
    lambda r: f"不好意思,{r.choice(EN_NAMES)} 剛離開,請問要留言給{zh_person(r)}嗎?",
    lambda r: f"系統顯示您的 {r.choice(TECH)} 帳號需要更新,請聯絡{r.choice(DEPTS)}。",
    lambda r: f"今天的會議改到下午 {num(r,1,5)} 點,地點在 {num(r,2,18)} 樓會議室。",
    lambda r: f"您好,請問要預約 {r.choice(EN_NAMES)} {rng_title(r)} 的時間嗎?",
    lambda r: f"麻煩您撥打 {phone(r)} 聯絡 {person(r)},謝謝。",
    lambda r: f"{r.choice(EN_NAMES)} 的 {r.choice(TECH)} 已經準備好,請到 {num(r,1,12)} 樓領取。",
    lambda r: f"您的快遞單號是 {num(r,1000000000,9999999999)},預計明天送達。",
    lambda r: f"請問您是要報修 {r.choice(TECH)} 還是 {r.choice(TECH)}?",
    lambda r: f"謝謝您的來電,祝您有美好的一天,再見。",
]

def rng_title(r):
    return r.choice(TITLES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000, help="number of utterances")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="data/text/corpus.tsv")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # No dedup: fixed receptionist lines SHOULD recur (they dominate production), and melo's
    # z_p noise makes even identical text yield different z/audio -> free augmentation.
    lines = [f"utt{i:06d}\t{rng.choice(TEMPLATES)(rng)}" for i in range(args.n)]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    uniq = len(set(l.split('\t', 1)[1] for l in lines))
    print(f"wrote {len(lines)} utterances ({uniq} unique texts) -> {args.out}")


if __name__ == "__main__":
    main()
