"""
산업별 기준금리 민감도 분석
한국은행 통화정책 경시대회 이슈보고서용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[분석 목적]
기준금리 변화가 산업별 부가가치 성장률에 미치는 효과를 측정하고,
산업 간 민감도 차이가 통화정책 운용에 주는 시사점을 도출한다.

[추정 모형]
  y_t = α + β₁·rate_up(t-2) + β₂·rate_dn(t-2)
          + β₃·gdp_t + β₄·fx_t + β₅·cpi(t-1) + β₆·esi(t-1)
          + β₇·y(t-1) + β₈·y(t-4) + β₉·covid_t + ε_t

  - 추정방법: OLS + Newey-West 표준오차 (HAC, 최대시차 4분기)
    → 분기 시계열의 자기상관·이분산을 동시에 보정하는 표준 방법론
  - 금리 민감도 = β₁ − β₂  (양수: 인상 시 성장, 음수: 인상 시 위축)

[변수 및 시차 설정 근거]
  변수              시차    근거
  ─────────────────────────────────────────────────────────────────
  rate_up_w        t-2    금리 인상 → 대출금리 상승 → 투자·소비 위축
                          실물 반영까지 평균 2분기 소요 (한은 기존 연구)
  rate_dn_w        t-2    금리 인하 → 유동성 공급 효과 (동일 시차 적용)
  gdp              t      동기 총수요 변화 통제
  fx               t      환율 충격의 즉각적 수출·수입비용 반영
                          (IT제조업만 t-1: 수출계약-결제 1분기 시차)
  cpi              t-1    물가 상승 → 실질임금·원가 영향은 다음 분기 반영
  esi              t-1    기업심리는 선행지표이나 생산 반영까지 1분기 소요
  y_lag1           t-1    경기 관성 통제 (AR 항)
  y_lag4           t-4    계절성 통제 (전년 동기 대비)
  covid            더미   2020Q1~2020Q4: 코로나19 충격 통제
                          미통제 시 서비스업·금융·부동산 잔차 편의 발생
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os, sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정  ← 경로가 자동으로 설정됩니다
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HAC_LAGS = 4
# 현재 스크립트 파일이 있는 디렉토리를 기본 경로로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE     = r"C:\Users\willy\OneDrive\바탕 화면\KHPC\회귀"
OUT_DIR  = SCRIPT_DIR
ESI_PATH = os.path.join(BASE, "경제심리지수.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  산업별 설명변수 목록 (AR항 개별 조정)
#
#  y_lag4 제거: 서비스업전체, 도소매숙박음식업, 농림어업, 금융및보험
#  y_lag1 제거: 부동산, 금융및보험, 건설업, IT제조업
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 공통 기본 변수 (AR항 제외)
_BASE_COMMON = [
    "rate_up_w_lag2",   # 금리 인상 (2분기 시차)
    "rate_dn_w_lag2",   # 금리 인하 (2분기 시차)
    "gdp",              # GDP 성장률 (현재)
    "fx",               # 환율 변화 (현재)
    "cpi_lag1",         # CPI (1분기 시차)
    "esi_lag1",         # 경제심리지수 (1분기 시차)
    "covid",            # 코로나19 더미 (2020Q1~2020Q4)
]

# 산업별 설명변수 딕셔너리
#   y_lag4 제거 대상: 서비스업전체, 도소매숙박음식업, 농림어업, 금융및보험
#   y_lag1 제거 대상: 부동산, 금융및보험, 건설업, IT제조업
INDUSTRY_X = {
    "건설업":           _BASE_COMMON + [          "y_lag4"],  # y_lag1 제거
    "농림어업":         _BASE_COMMON + ["y_lag1"          ],  # y_lag4 제거
    "IT제조업":         _BASE_COMMON + [          "y_lag4"],  # y_lag1 제거
    "제조업전체":       _BASE_COMMON + ["y_lag1", "y_lag4"],  # 둘 다 유지
    "비IT제조업":       _BASE_COMMON + ["y_lag1", "y_lag4"],  # 둘 다 유지
    "서비스업전체":     _BASE_COMMON + ["y_lag1"          ],  # y_lag4 제거
    "금융및보험":       _BASE_COMMON + [                  ],  # 둘 다 제거
    "부동산":           _BASE_COMMON + [          "y_lag4"],  # y_lag1 제거
    "도소매숙박음식업": _BASE_COMMON + ["y_lag1"         ],  # y_lag4 제거
}

# 하위 호환용 fallback
BASE_X    = _BASE_COMMON + ["y_lag1", "y_lag4"]
IT_BASE_X = INDUSTRY_X["IT제조업"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  한글 폰트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def set_korean_font():
    for path in [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]:
        if os.path.exists(path):
            try:
                plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                pass
    plt.rcParams["font.family"] = "sans-serif"

set_korean_font()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def normalize_quarter(val) -> str:
    """분기 표기 통일  2020/Q1 · 2020 1 · 2020.1  →  2020Q1"""
    s = str(val).strip()
    if "Q" in s:
        return s.replace("/", "").replace(" ", "")
    for sep in [" ", "."]:
        if sep in s:
            year, rest = s.split(sep, 1)
            return f"{year.strip()}Q{rest.split('/')[0].strip()}"
    return s


def sig_star(p: float) -> str:
    if np.isnan(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 로딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_macro() -> pd.DataFrame:
    """거시변수 로딩 및 금리 인상/인하 분리"""
    specs = {
        "gdp":      ("GDP.xlsx",      "GDP 로그차분"),
        "fx":       ("환율.xlsx",     "환율 로그차분"),
        "cpi":      ("cpi.xlsx",      "로그차분"),
        "rate_raw": ("기준금리.xlsx", "기준금리 증감률"),
    }
    frames = []
    for key, (fname, col) in specs.items():
        try:
            df = pd.read_excel(f"{BASE}/{fname}")   
            df["분기"] = df.iloc[:, 0].apply(normalize_quarter)
            frames.append(df[["분기", col]].rename(columns={col: key}))
        except Exception as e:
            print(f"[오류] {fname} 로드 실패: {e}")
            sys.exit(1)

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="분기", how="inner")

    # ESI
    try:
        esi_df = pd.read_excel(ESI_PATH)
        esi_df["분기"] = esi_df.iloc[:, 0].apply(normalize_quarter)
        num_cols = [c for c in esi_df.columns[1:] if pd.api.types.is_numeric_dtype(esi_df[c])]
        esi_col = "로그차분" if "로그차분" in esi_df.columns else (num_cols[0] if num_cols else None)
        if esi_col:
            macro = macro.merge(
                esi_df[["분기", esi_col]].rename(columns={esi_col: "esi"}),
                on="분기", how="left"
            )
        else:
            macro["esi"] = 0.0
    except Exception:
        print("  [경고] ESI 데이터 없음 → 0으로 대체")
        macro["esi"] = 0.0

    # 금리 인상/인하 분리
    # 0.25%p 단위 정규화 후 인상·인하 각각 누적
    # (Tenreyro & Thwaites, 2016; 한은 BOK-WP 2022-21 참조)
    # %p 단위 정규화: 원래 증감값(%p)을 그대로 사용
    # (기존: 0.25%p 단위 횟수로 변환 → 계수가 ~40배 부풀려지는 문제 수정)
    macro["rate_up_w"] = macro["rate_raw"].clip(lower=0)    # 인상분 (%p)
    macro["rate_dn_w"] = (-macro["rate_raw"]).clip(lower=0) # 인하분 (%p)
    macro = macro.rename(columns={"rate_raw": "rate"})

    return macro.dropna(subset=["gdp", "fx", "cpi", "rate"]).reset_index(drop=True)


def load_industries() -> dict:
    """산업별 부가가치 성장률 로딩"""
    industries = {}

    # 건설업, 농림어업
    for name, fname in [("건설업", "건설업.xlsx"), ("농림어업", "농림어업.xlsx")]:
        try:
            df = pd.read_excel(f"{BASE}/{fname}")
            df["분기"] = df.iloc[:, 0].apply(normalize_quarter)
            col = next((c for c in df.columns if "로그" in str(c) or "증감" in str(c)), None)
            if col:
                industries[name] = (df[["분기", col]]
                                    .rename(columns={col: "y"})
                                    .dropna().reset_index(drop=True))
        except Exception as e:
            print(f"  [경고] {name} 로드 실패: {e}")

    # 제조업 (IT / 비IT / 전체)
    try:
        mfg = pd.read_excel(f"{BASE}/제조업.xlsx", sheet_name=0)
        mfg["분기"] = mfg.iloc[:, 0].apply(normalize_quarter)
        for name, col in [("IT제조업",   "IT 제조업 로그차분"),
                          ("제조업전체", "제조업 전체 로그차분"),
                          ("비IT제조업", "비IT 제조업 로그차분")]:
            if col in mfg.columns:
                industries[name] = (mfg[["분기", col]]
                                    .rename(columns={col: "y"})
                                    .dropna().reset_index(drop=True))
    except Exception as e:
        print(f"  [경고] 제조업 로드 실패: {e}")

    # 서비스업 (실제 파일이 CP949 CSV, 확장자만 .xlsx)
    try:
        svc = pd.read_csv(f"{BASE}/서비스업.xlsx", encoding="cp949")
        svc["분기"] = svc["분기"].apply(normalize_quarter)

        col_map = {
            "서비스업전체":     "서비스업 전체 로그차분",
            "금융및보험":       "금융 및 보험업 로그차분",
            "부동산":           "부동산업 로그차분",
            "도소매숙박음식업": "도소매 및 숙박 음식업 로그차분",
        }
        for sname, col in col_map.items():
            if col not in svc.columns:
                print(f"  [경고] 서비스업 컬럼 없음: '{col}'")
                print(f"         실제 컬럼: {svc.columns.tolist()}")
                continue
            industries[sname] = (svc[["분기", col]]
                                 .rename(columns={col: "y"})
                                 .dropna().reset_index(drop=True))
    except Exception as e:
        print(f"  [경고] 서비스업 로드 실패: {e}")

    return industries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 구성 (시차 생성 + 코로나 더미)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_model_data(macro: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
    """
    거시변수 + 산업 성장률 병합 후 0~4분기 모든 시차 변수 생성.
    """
    df = macro.merge(ind_df, on="분기", how="inner").reset_index(drop=True)

    # 1. 종속변수 스케일링 (1단위 = 1%p)
    if "y" in df.columns:
        df["y"] = df["y"] * 100.0

    # 2. 거시변수별 0~4 시차 생성 및 스케일링
    # ※ 스케일링은 원본 변수에 먼저 적용한 뒤 시차를 생성해야
    #    lag0이 원본 컬럼과 중복(데이터 누수)되는 문제를 방지한다.
    vars_to_lag = ["rate_up_w", "rate_dn_w", "gdp", "fx", "cpi", "esi"]
    for var in vars_to_lag:
        # 원본을 스케일링한 임시 시리즈
        if "rate" in var:
            scaled = df[var] * 25.0
        else:
            scaled = df[var] * 100.0

        for lag in range(5):
            col_name = f"{var}_lag{lag}"
            df[col_name] = scaled.shift(lag)

        # 원본 컬럼(비스케일)은 시차 생성 후 제거 — 모델에 직접 투입되지 않도록
        # (gdp, fx 등은 lag0 컬럼으로만 사용)
        if var in df.columns:
            df.drop(columns=[var], inplace=True, errors="ignore")

    # 3. 경기 관성 및 계절성 (AR항)
    df["y_lag1"] = df["y"].shift(1)
    df["y_lag4"] = df["y"].shift(4)

    # 4. 코로나19 더미
    df["covid"] = ((df["분기"] >= "2020Q1") & (df["분기"] <= "2020Q4")).astype(int)

    return df.dropna().reset_index(drop=True)


def find_optimal_lags(name: str, df: pd.DataFrame, ar_vars: list) -> list[str]:
    """
    SFS 스타일로 각 변수의 최적 시차(0~4) 탐색.
    기준: 해당 변수의 p-value 최소화 (동률 시 R2 최대화)
    금리 인상/인하는 동일 시차 적용.
    """
    target_vars = ["gdp", "fx", "cpi", "esi", "rate_up_w", "rate_dn_w"]
    fixed_vars  = [v for v in ar_vars + ["covid"] if v in df.columns]
    
    optimal_lags = {} # 최적 시차 저장용 딕셔너리
    current_selected = fixed_vars.copy()
    final_x_cols     = fixed_vars.copy()
    
    print(f"  [시차 최적화] {name} 탐색 중...")
    
    for var in target_vars:
        if var == "rate_up_w":
            # ── rate_up_w / rate_dn_w 동일 시차: 두 p-value 합 최소 기준 ──
            best_lag   = 0
            min_p_sum  = np.inf
            max_r2     = -1.0

            for lag in range(5):
                up_cand = f"rate_up_w_lag{lag}"
                dn_cand = f"rate_dn_w_lag{lag}"
                # 두 변수를 동시에 투입
                test_vars = (
                    [c for c in current_selected
                     if not c.startswith("rate_up_w_lag")
                     and not c.startswith("rate_dn_w_lag")]
                    + [up_cand, dn_cand]
                )
                try:
                    X     = sm.add_constant(df[test_vars])
                    model = sm.OLS(df["y"], X).fit(
                        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
                    )
                    p_up  = model.pvalues.get(up_cand, np.nan)
                    p_dn  = model.pvalues.get(dn_cand, np.nan)
                    if np.isnan(p_up) or np.isnan(p_dn):
                        continue
                    p_sum = p_up + p_dn
                    r2    = model.rsquared

                    if p_sum < min_p_sum - 1e-5:
                        min_p_sum = p_sum
                        max_r2    = r2
                        best_lag  = lag
                    elif abs(p_sum - min_p_sum) < 1e-5 and r2 > max_r2:
                        max_r2   = r2
                        best_lag = lag
                except Exception:
                    continue

            print(f"    - rate_up_w / rate_dn_w 공통 최적 시차: t-{best_lag} "
                  f"(p합 기준)")
            # rate_up_w 등록
            optimal_lags["rate_up_w"] = best_lag
            up_col = f"rate_up_w_lag{best_lag}"
            final_x_cols    = [c for c in final_x_cols    if not c.startswith("rate_up_w_lag")]
            current_selected= [c for c in current_selected if not c.startswith("rate_up_w_lag")]
            final_x_cols.append(up_col)
            current_selected.append(up_col)
            continue  # rate_dn_w는 아래 블록에서 처리

        elif var == "rate_dn_w":
            # rate_up_w와 동일 시차 적용
            best_lag = optimal_lags.get("rate_up_w", 0)
            print(f"    - rate_dn_w 시차: rate_up_w와 동일하게 t-{best_lag} 적용")
            optimal_lags[var] = best_lag
            best_col = f"{var}_lag{best_lag}"
            final_x_cols    = [c for c in final_x_cols    if not c.startswith(f"{var}_lag")]
            final_x_cols.append(best_col)
            current_selected= [c for c in current_selected if not c.startswith(f"{var}_lag")]
            current_selected.append(best_col)
            continue  # 아래 탐색 블록 건너뜀

        # ── 일반 변수(gdp, fx, cpi, esi): p-value 최소 시차 탐색 ──
        best_lag = 0
        min_p    = 1.1
        max_r2   = -1.0

        for lag in range(5):
            candidate = f"{var}_lag{lag}"
            test_vars = (
                [c for c in current_selected if not c.startswith(f"{var}_lag")]
                + [candidate]
            )
            try:
                X     = sm.add_constant(df[test_vars])
                model = sm.OLS(df["y"], X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
                )
                p_val = model.pvalues.get(candidate, np.nan)
                r2    = model.rsquared
                if np.isnan(p_val):
                    continue
                if p_val < min_p - 1e-5:
                    min_p    = p_val
                    max_r2   = r2
                    best_lag = lag
                elif abs(p_val - min_p) < 1e-5 and r2 > max_r2:
                    max_r2   = r2
                    best_lag = lag
            except Exception:
                continue

        optimal_lags[var] = best_lag
        best_col = f"{var}_lag{best_lag}"
        final_x_cols    = [c for c in final_x_cols    if not c.startswith(f"{var}_lag")]
        final_x_cols.append(best_col)
        current_selected= [c for c in current_selected if not c.startswith(f"{var}_lag")]
        current_selected.append(best_col)

    # 출력 메시지에서 AR항과 코로나 더미 제외하고 주요 거시변수만 표시
    print_vars = []
    for v in final_x_cols:
        if "_lag" in v and not v.startswith("y_lag"): # AR항 제외
            base, lag = v.rsplit("_lag", 1)
            print_vars.append(f"{VAR_LABEL.get(base, base)}(t-{lag})") # VAR_LABEL 사용
    print(f"    - 결과: " + ", ".join(print_vars))

    return final_x_cols
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OLS 추정 (Newey-West HAC 표준오차)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_ols(name: str, df: pd.DataFrame, x_cols: list) -> dict | None:
    """
    OLS 추정.
    cov_type='HAC' (Newey-West): 분기 시계열의 자기상관·이분산을
    동시에 교정하는 거시계량 표준 방법론.
    최소 관측치 조건 미달 시 None 반환.
    """
    x_cols = list(dict.fromkeys(c for c in x_cols if c in df.columns))

    if len(df) < len(x_cols) + 10:
        print(f"  [건너뜀] {name}: 관측치 {len(df)}개 < 최소 필요 {len(x_cols)+10}개")
        return None

    try:
        X     = sm.add_constant(df[x_cols], has_constant="add")
        model = sm.OLS(df["y"], X).fit(cov_type="HAC",
                                        cov_kwds={"maxlags": HAC_LAGS})
    except Exception as e:
        print(f"  [오류] {name} 추정 실패: {e}")
        return None

    # VIF (다중공선성 점검 — 결과표에는 미출력, 경고만)
    try:
        vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_df   = pd.DataFrame({"변수": X.columns, "VIF": vif_vals})
    except Exception:
        vif_df = pd.DataFrame()

    dw      = sm.stats.stattools.durbin_watson(model.resid)
    
    # 최적화된 시차 변수명 찾기
    up_col = next((c for c in x_cols if "rate_up_w_lag" in c), None)
    dn_col = next((c for c in x_cols if "rate_dn_w_lag" in c), None)
    
    p_up    = model.pvalues.get(up_col, np.nan) if up_col else np.nan
    p_dn    = model.pvalues.get(dn_col, np.nan) if dn_col else np.nan
    sens_up = model.params.get(up_col, 0) if up_col else 0
    sens_dn = model.params.get(dn_col, 0) if dn_col else 0

    return {
        "name":        name,
        "model":       model,
        "x_cols":      x_cols,
        "df":          df,
        "up_col":      up_col,
        "dn_col":      dn_col,
        "sensitivity": sens_up - sens_dn,   # 금리 순 민감도
        "sens_up":     sens_up,
        "sens_dn":     sens_dn,
        "p_up":        p_up,
        "p_dn":        p_dn,
        "rsq":         model.rsquared,
        "adj_rsq":     model.rsquared_adj,
        "dw":          dw,
        "vif":         vif_df,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  콘솔 출력: 산업별 회귀 결과
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 변수명 → 보고서용 한글 레이블
VAR_LABEL = {
    "const":           "상수",
    "rate_up_w_lag2":  "금리인상(t-2)",
    "rate_dn_w_lag2":  "금리인하(t-2)",
    "gdp":             "GDP성장률(t)",
    "fx":              "환율변화(t)",
    "fx_lag1":         "환율변화(t-1)",
    "cpi_lag1":        "CPI(t-1)",
    "esi_lag1":        "경제심리(t-1)",
    "y_lag1":          "피설명변수(t-1)",
    "y_lag4":          "피설명변수(t-4)",
    "covid":           "코로나더미",
}

def print_result(res: dict):
    m   = res["model"]
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  [{res['name']}]")
    print(f"  추정방법: OLS, Newey-West 표준오차 (HAC, maxlags={HAC_LAGS})")
    print(f"  관측치: {int(m.nobs)}  |  R² = {res['rsq']:.3f}  |  "
          f"Adj R² = {res['adj_rsq']:.3f}  |  DW = {res['dw']:.3f}")
    print(f"{sep}")
    print(f"  {'변수':<16} {'계수':>9} {'표준오차':>9} {'t값':>7} {'p값':>8}  유의")
    print(f"  {'-'*65}")

    for v in m.params.index:
        # 시차 정보가 포함된 변수명 처리 (예: gdp_lag2 -> GDP(t-2))
        if "_lag" in v:
            base, lag = v.rsplit("_lag", 1)
            label = f"{VAR_LABEL.get(base, base)}(t-{lag})"
        else:
            label = VAR_LABEL.get(v, v)
            
        coef  = m.params[v]
        se    = m.bse[v]
        tval  = coef / se if se > 0 else np.nan
        pval  = m.pvalues[v]
        print(f"  {label:<16} {coef:>9.4f} {se:>9.4f} {tval:>7.3f} {pval:>8.4f}  "
              f"{sig_star(pval)}")

    print(f"\n  ── 금리 민감도 ─────────────────────────────────────")
    up_lbl = f"인상(t-{res['up_col'].split('_lag')[1]})" if res['up_col'] else "인상"
    dn_lbl = f"인하(t-{res['dn_col'].split('_lag')[1]})" if res['dn_col'] else "인하"
    
    print(f"  금리{up_lbl} 계수 (β₁): {res['sens_up']:>+8.4f}  {sig_star(res['p_up'])}")
    print(f"  금리{dn_lbl} 계수 (β₂): {res['sens_dn']:>+8.4f}  {sig_star(res['p_dn'])}")
    print(f"  순 민감도 (β₁−β₂): {res['sensitivity']:>+8.4f}")

    # 경제학적 해석
    s = res["sensitivity"]
    if   s < -0.03: msg = "금리 인상 시 뚜렷한 성장 위축"
    elif s <  0:    msg = "금리 인상 시 소폭 성장 위축"
    elif s <  0.03: msg = "금리 변동에 상대적으로 둔감"
    else:           msg = "금리 인상 시 성장세 유지"
    print(f"  → 해석: {msg}")

    # VIF 출력 (다중공선성 점검)
    if not res["vif"].empty:
        vif = res["vif"]
        if "const" in vif["변수"].values:
            vif = vif[vif["변수"] != "const"].reset_index(drop=True)
        print(f"\n  [VIF] 다중공선성 점검")
        for _, row in vif.iterrows():
            print(f"  {row['변수']:<16} {row['VIF']:>8.3f}")
        hv = vif[vif["VIF"] > 10]
        if not hv.empty:
            print(f"  [참고] VIF > 10: {hv['변수'].tolist()} → 다중공선성 가능성")
    else:
        print("\n  [VIF] 계산 실패 또는 다중공선성 검사의 충분한 정보 없음")

    if res["dw"] < 1.5:
        print(f"  [참고] DW = {res['dw']:.2f} → 잔차 자기상관 가능성 (HAC SE로 보정됨)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  콘솔 출력: 전체 민감도 순위표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_ranking(all_results: dict):
    sep = "=" * 84
    print(f"\n{sep}")
    print(f"  산업별 금리 민감도 요약표  (순 민감도 = β₁ − β₂, 오름차순)")
    print(f"  {'순위':<4} {'산업':<13} {'순민감도':>9} {'β₁인상':>8} "
          f"{'β₂인하':>8} {'p(인상)':>8} {'p(인하)':>8} {'Adj R²':>7}  해석")
    print(f"  {'-'*80}")

    ranked = sorted(all_results.items(), key=lambda x: x[1]["sensitivity"])
    for i, (name, res) in enumerate(ranked, 1):
        s = res["sensitivity"]
        if   s < -0.03: interp = "강한 위축"
        elif s <  0:    interp = "소폭 위축"
        elif s <  0.03: interp = "둔감"
        else:           interp = "성장 유지"

        sig    = "★" if (res["p_up"] < 0.05 or res["p_dn"] < 0.05) else " "
        p_up_s = f"{res['p_up']:.4f}" if not np.isnan(res["p_up"]) else "   -  "
        p_dn_s = f"{res['p_dn']:.4f}" if not np.isnan(res["p_dn"]) else "   -  "

        print(f"  {i:<4} {name:<13} {s:>9.4f} {res['sens_up']:>8.4f} "
              f"{res['sens_dn']:>8.4f} {p_up_s:>8} {p_dn_s:>8} "
              f"{res['adj_rsq']:>7.3f}  {sig} {interp}")

    print(f"{sep}")
    print(f"  ★ = 금리 인상 또는 인하 계수 중 하나 이상 p < 0.05")
    print(f"  순 민감도 음수 = 금리 인상 시 해당 산업 성장률 하락\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_industry(res: dict):
    """산업별 진단 플롯: 실제vs예측 / 잔차"""
    df, m    = res["df"], res["model"]
    quarters = df["분기"].tolist()
    n        = len(quarters)
    Y        = df["y"].values
    Yp       = m.fittedvalues.values
    R        = m.resid.values
    step     = max(1, n // 8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"{res['name']}  |  R² = {res['rsq']:.3f}  "
        f"Adj R² = {res['adj_rsq']:.3f}  DW = {res['dw']:.3f}",
        fontsize=12, fontweight="bold"
    )

    # 코로나 기간 인덱스
    covid_idx = [i for i, q in enumerate(quarters) if "2020Q1" <= q <= "2020Q4"]

    # ① 실제 vs 예측
    ax = axes[0]
    ax.plot(range(n), Y,  "o-", ms=3, lw=1.5, color="#FF6D00", label="실제값")
    ax.plot(range(n), Yp, "s--",ms=3, lw=1.5, color="#2962FF", label="예측값")
    step = max(1, n // 8)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(quarters[::step], rotation=45, fontsize=7)
    ax.set_title("실제값 vs 예측값"); ax.legend(fontsize=8); ax.grid(alpha=0.7)
    if covid_idx:
        ax.axvspan(covid_idx[0], covid_idx[-1], alpha=0.10,
                   color="gray", label="코로나(2020Q1~2020Q4)")


    # ② 잔차
    ax = axes[1]
    ax.scatter(range(n), R, s=18, color="#7E3AF2", alpha=0.75, zorder=3)
    ax.axhline(0, color="red", lw=1.2, ls="--")
    if covid_idx:
        ax.axvspan(covid_idx[0], covid_idx[-1], alpha=0.10, color="gray")
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(quarters[::step], rotation=45, fontsize=7)
    ax.set_title("잔차  (패턴 없으면 양호)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, f"{res['name']}_회귀진단.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {fpath}")


def plot_sensitivity_bar(all_results: dict):
    """
    [핵심 그래프] 산업별 금리 순 민감도 막대그래프 (95% 신뢰구간 포함)
    보고서 본문에 직접 삽입하는 메인 차트.
    """
    rows  = sorted(all_results.items(), key=lambda x: x[1]["sensitivity"])
    names = [r[0] for r in rows]
    sens  = [r[1]["sensitivity"] for r in rows]

    # 95% CI: SE(β₁−β₂) ≈ √(SE(β₁)²+SE(β₂)²) (두 계수 독립 가정)
    ci_lo, ci_hi, bar_colors = [], [], []
    for _, res in rows:
        m      = res["model"]
        up_col = res.get("up_col") or "rate_up_w_lag2"
        dn_col = res.get("dn_col") or "rate_dn_w_lag2"
        se = np.sqrt(m.bse.get(up_col, 0) ** 2 +
                     m.bse.get(dn_col, 0) ** 2)
        s  = res["sensitivity"]
        ci_lo.append(s - 1.96 * se)
        ci_hi.append(s + 1.96 * se)
        sig = res["p_up"] < 0.05 or res["p_dn"] < 0.05
        bar_colors.append("#C0392B" if sig else "#AEB6BF")

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(names))

    bars = ax.bar(x, sens, color=bar_colors, alpha=0.88,
                  edgecolor="white", width=0.6, zorder=2)
    ax.errorbar(
        x, sens,
        yerr=[[s - lo for s, lo in zip(sens, ci_lo)],
              [hi - s  for s, hi in zip(sens, ci_hi)]],
        fmt="none", color="#2C3E50", capsize=5, lw=1.5, zorder=3
    )

    # 계수값 레이블
    for bar, s in zip(bars, sens):
        offset = 0.003 if s >= 0 else -0.003
        va     = "bottom" if s >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                s + offset, f"{s:+.3f}",
                ha="center", va=va, fontsize=8, fontweight="bold")

    ax.axhline(0, color="#2C3E50", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("금리 순 민감도  (β₁ − β₂)", fontsize=11)
    ax.set_title(
        "산업별 기준금리 민감도 비교\n"
        "(OLS 추정계수, 95% 신뢰구간, 음수 = 금리 인상 시 성장 위축)",
        fontsize=12
    )
    ax.legend(handles=[
        Patch(facecolor="#C0392B", alpha=0.88, label="p < 0.05  (통계적으로 유의)"),
        Patch(facecolor="#AEB6BF", alpha=0.88, label="p ≥ 0.05"),
    ], fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    plt.tight_layout()

    fpath = os.path.join(OUT_DIR, "산업별_금리민감도_비교.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {fpath}")


def plot_coef_heatmap(all_results: dict):
    """
    산업 × 주요 변수 회귀계수 히트맵.
    - 금리인상/인하를 '금리민감도(β₁−β₂)'로 통합
    - correlation_heatmap 스타일: 세로형, x=변수, y=산업(R² 표시)
    - 색: 컬럼별 정규화 → RdBu (양=빨강, 음=파랑), vmin=-1, vmax=1
    - 셀 텍스트: 실제 계수값, 색은 정규화값 기준 자동 흰/검
    """
    # ── 변수 정의 (금리 민감도로 통합) ──────────────────────────────
    KEY_VARS = ["rate_sensitivity", "gdp", "fx", "cpi", "esi"]
    KEY_LBLS = ["금리민감도", "GDP", "환율", "CPI", "ESI"]

    # 산업 순서: correlation_heatmap과 동일하게 정렬 (원하는 순서로 수동 지정 가능)
    preferred_order = [
        "건설업", "농림어업", "IT제조업", "제조업전체",
        "비IT제조업", "서비스업전체", "금융및보험", "부동산", "도소매숙박음식업"
    ]
    ind_names = [n for n in preferred_order if n in all_results]
    # 혹시 preferred_order에 없는 산업이 있으면 뒤에 추가
    for n in sorted(all_results.keys()):
        if n not in ind_names:
            ind_names.append(n)

    n_ind  = len(ind_names)
    n_vars = len(KEY_VARS)
    mat    = np.full((n_ind, n_vars), np.nan)   # 실제 계수값
    rsq    = {}

    for i, name in enumerate(ind_names):
        res = all_results[name]
        m   = res["model"]
        rsq[name] = res["rsq"]

        # 금리 민감도 = β₁(인상) − β₂(인하)
        mat[i, 0] = res["sensitivity"]

        # 나머지 변수 (최적 시차가 적용된 컬럼 찾기)
        for j, vkey in enumerate(KEY_VARS[1:], start=1):
            # vkey는 'gdp', 'fx' 등. 실제 컬럼은 'gdp_lag0', 'gdp_lag1' 중 하나
            col = next((c for c in m.params.index if c.startswith(f"{vkey}_lag")), None)
            if col:
                mat[i, j] = m.params[col]

    # ── 컬럼별 정규화 (상관계수 히트맵과 동일한 방식) ────────────────
    mat_norm = np.zeros_like(mat)
    for j in range(n_vars):
        col_vals = mat[:, j]
        valid    = col_vals[~np.isnan(col_vals)]
        if len(valid) == 0:
            continue
        vmax = np.abs(valid).max() # 해당 변수(열)에서 가장 큰 절대값 찾기
        if vmax == 0:
            continue
        mat_norm[:, j] = col_vals / vmax   # −1 ~ +1 범위

    # ── 플롯 ─────────────────────────────────────────────────────────
    cell_h = 1.1   # 셀 높이(인치)
    cell_w = 1.5   # 셀 너비(인치)
    fig_w  = cell_w * n_vars + 2.5          # 우측 컬러바 여백
    fig_h  = cell_h * n_ind  + 1.8          # 상/하 여백

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat_norm, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")

    # 격자선 (셀 경계)
    ax.set_xticks(np.arange(-0.5, n_vars, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_ind,  1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # x축 레이블
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(KEY_LBLS, fontsize=10, rotation=30, ha="right")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # y축 레이블: 산업명 + R²
    ylbls = [f"{n}\nR²={rsq[n]:.3f}" for n in ind_names]
    ax.set_yticks(range(n_ind))
    ax.set_yticklabels(ylbls, fontsize=9, va="center")

    # 셀 내 텍스트 (실제 계수값)
    for i in range(n_ind):
        for j in range(n_vars):
            v = mat[i, j]
            if np.isnan(v):
                continue
            norm_v = mat_norm[i, j]
            txt_color = "white" if abs(norm_v) > 0.55 else "black"
            ax.text(j, i, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=txt_color)

    # 컬러바 (correlation_heatmap 스타일: 세로, 우측)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label("회귀계수", fontsize=9, rotation=90, labelpad=6)
    cbar.set_ticks([-3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.75, 3])

    ax.set_title(
        "산업별 거시변수 회귀계수 히트맵\n(금리민감도)",
        fontsize=12, fontweight="bold", pad=12
    )

    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "산업별_계수_히트맵.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {fpath}")


def plot_sd_impact_heatmap(all_results: dict, macro: pd.DataFrame):
    """
    산업별 변수별 상관계수 히트맵
    ─────────────────────────────────────────────────────────────────────
    각 산업별로 y와 설명변수 간 상관계수 값을 계산하여
    변수 간 방향성과 강도를 비교합니다.
    """
    KEY_LBLS = ["금리민감도", "GDP", "환율", "CPI", "ESI"]

    preferred_order = [
        "건설업", "농림어업", "IT제조업", "제조업전체",
        "비IT제조업", "서비스업전체", "금융및보험", "부동산", "도소매숙박음식업"
    ]
    ind_names = [n for n in preferred_order if n in all_results]
    for n in sorted(all_results.keys()):
        if n not in ind_names:
            ind_names.append(n)

    n_ind  = len(ind_names)
    n_vars = len(KEY_LBLS)

    mat = np.full((n_ind, n_vars), np.nan)
    rsq = {}

    for i, name in enumerate(ind_names):
        res = all_results[name]
        df  = res["df"]
        rsq[name] = res["rsq"]

        for j, lbl in enumerate(KEY_LBLS):
            if lbl == "금리민감도":
                # 최적화된 인상/인하 컬럼 사용
                up_col = res.get("up_col")
                dn_col = res.get("dn_col")
                if up_col and dn_col:
                    x = df[up_col] - df[dn_col]
                else:
                    x = pd.Series(np.nan, index=df.index)
            else:
                # 기타 변수 (gdp, fx 등)의 최적 시차 컬럼 찾기
                vkey = {"GDP": "gdp", "환율": "fx", "CPI": "cpi", "ESI": "esi"}.get(lbl)
                col = next((c for c in df.columns if c.startswith(f"{vkey}_lag")), None)
                x = df[col] if col else pd.Series(np.nan, index=df.index)

            if x.isna().all() or df["y"].isna().all():
                corr = np.nan
            else:
                corr = x.corr(df["y"])
            mat[i, j] = corr

    mat_norm = np.zeros_like(mat)
    for j in range(n_vars):
        valid = mat[:, j][~np.isnan(mat[:, j])]
        if len(valid) == 0:
            continue
        vmax = np.abs(valid).max()
        if vmax == 0:
            continue
        mat_norm[:, j] = mat[:, j] / vmax

    cell_h = 1.1
    cell_w = 1.6
    fig_w  = cell_w * n_vars + 3.0
    fig_h  = cell_h * n_ind  + 2.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat_norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", alpha=0.72)

    ax.set_xticks(np.arange(-0.5, n_vars, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_ind,  1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(KEY_LBLS, fontsize=10, rotation=30, ha="right")
    ax.xaxis.tick_bottom()

    ylbls = [f"{n}\nR²={rsq[n]:.3f}" for n in ind_names]
    ax.set_yticks(range(n_ind))
    ax.set_yticklabels(ylbls, fontsize=9, va="center")

    for i in range(n_ind):
        for j in range(n_vars):
            v = mat[i, j]
            if np.isnan(v):
                continue
            norm_v = mat_norm[i, j]
            txt_color = "white" if abs(norm_v) > 0.55 else "black"
            ax.text(j, i, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label("상관계수값\n(컬럼 내 상대 정규화)", fontsize=9,
                   rotation=90, labelpad=8)
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

    ax.set_title(
        "산업별 변수별 상관계수 히트맵\n"
        "각 셀 = y와 변수 간 상관계수 (음수=파랑, 양수=빨강)",
        fontsize=11, fontweight="bold", pad=12
    )

    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "산업별_실질충격_히트맵.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {fpath}")

    print("\n  [참고] 변수별 상관계수 요약")
    print(f"  {'변수':<10} {'값':>8}  " +
          "  ".join(f"{n[:4]:>7}" for n in ind_names))
    print(f"  {'-'*80}")
    for j, lbl in enumerate(KEY_LBLS):
        row_vals = "  ".join(
            f"{mat[i,j]:>7.3f}" if not np.isnan(mat[i,j]) else f"{'  -':>7}"
            for i in range(n_ind)
        )
        print(f"  {lbl:<10} {'':>8}  {row_vals}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("  산업별 기준금리 민감도 분석")
    print("  추정방법: OLS + Newey-West HAC 표준오차 (maxlags=4)")
    print("  변수: 금리인상·인하(lag2) / GDP / 환율 / CPI·ESI(lag1)")
    print("        / AR(lag1,lag4) / 코로나더미(2020Q1~2020Q4)")
    print("=" * 70 + "\n")

    macro      = load_macro()
    industries = load_industries()
    set_korean_font()

    all_results: dict = {}

    for name, ind_df in industries.items():
        print(f"\n{'─' * 50}")
        print(f"  [{name}] 분석 시작...")

        df = build_model_data(macro, ind_df)
        
        # 1. AR항 결정 (기존 설정 활용)
        orig_x = INDUSTRY_X.get(name, BASE_X)
        ar_vars = [v for v in orig_x if v in ["y_lag1", "y_lag4"]]
        
        # 2. 최적 시차 탐색
        opt_x_cols = find_optimal_lags(name, df, ar_vars)

        print(f"  관측치: {len(df)}  /  최종 설명변수: {len(opt_x_cols)}개")

        res = run_ols(name, df, opt_x_cols)
        if res is None:
            continue

        all_results[name] = res
        print_result(res)
        plot_industry(res)

    if not all_results:
        print("\n[오류] 추정된 산업이 없습니다. 데이터 경로를 확인하세요.")
        return

    # 전체 비교
    print(f"\n{'=' * 70}")
    plot_sensitivity_bar(all_results)
    plot_coef_heatmap(all_results)
    plot_sd_impact_heatmap(all_results, macro)   # ← 방식 B: 1-SD 실질 충격 히트맵
    print_ranking(all_results)

    # 보고서용 전체 계수표 (부록)
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  [부록] 전체 산업 회귀계수표")
    print(f"  추정방법: OLS + Newey-West HAC SE")
    print(f"  유의수준: *** p<0.01  ** p<0.05  * p<0.1")
    print(f"{sep}")

    for name, res in sorted(all_results.items()):
        m = res["model"]
        print(f"\n  [{name}]  관측치={int(m.nobs)}  R²={res['rsq']:.3f}  "
              f"Adj R²={res['adj_rsq']:.3f}  DW={res['dw']:.3f}")
        print(f"  {'변수':<16} {'계수':>9} {'표준오차':>9} {'95% CI':>22}  {'p값':>8}  유의")
        print(f"  {'-'*72}")
        for v in m.params.index:
            label = VAR_LABEL.get(v, v)
            coef  = m.params[v]
            se    = m.bse[v]
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            print(f"  {label:<16} {coef:>9.4f} {se:>9.4f} "
                  f"  [{ci_lo:>+7.4f}, {ci_hi:>+7.4f}]  "
                  f"{m.pvalues[v]:>8.4f}  {sig_star(m.pvalues[v])}")

    print(f"\n{sep}")
    print(f"\n  완료. 저장 경로: {OUT_DIR}\n")


if __name__ == "__main__":
    main()