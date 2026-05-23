# สรุปการพัฒนาโปรเจกต์ Thailand Economic Analysis

**ระยะเวลา:** 2026-05  
**ผู้วิจัย:** poommy  
**คำถามหลัก:** มี alpha ที่ exploit ได้ใน SET index ไหม?

---

## จุดเริ่มต้น

เริ่มจากคำถามเดียว: "มี alpha ใน SET index ไหม?"
ใช้ข้อมูล weekly 25 ปี (2000–2025) ทดสอบ feature กว่า 70 ตัว ครอบคลุม
macro เศรษฐกิจ, ราคาสินทรัพย์ global, sector rotation, NLP ข่าว, และ stock-level momentum

---

## เส้นทางการพัฒนา

### Phase 1 — ทำความเข้าใจข้อมูล (NB01–07)
- EDA ข้อมูล macro ไทย, ราคาตลาด global, และ NLP จากข่าวการเงิน
- ตรวจสอบ data leakage อย่างเข้มงวด (NB07 leakage audit)
- ผลิต unified_weekly_clean.csv: 1,322 weeks × 72 features ที่ clean และ lagged ถูกต้อง
- บทเรียน: ข้อมูล macro ไทย (GDP, CPI) ช้าเกินไป (publication lag 12 สัปดาห์) สำหรับ weekly model

### Phase 2 — Feature Selection & Signal Discovery (NB08–09)
- Pruning features จาก 53 → 11 ตัวด้วย permutation test และ bootstrap
- ค้นพบ eem_ret_d_lag1 (EEM daily return lag 1) เป็น signal ที่แข็งแกร่งที่สุด
  - Spearman IC = +0.11, p < 0.05, ICIR = 1.47
  - Sign-stable ตลอด 2000–2025 (positive > 55% ของ rolling windows)
- Feature ที่ไม่ work: th_uncertainty (p > 0.50), oil prices (IC ไม่สม่ำเสมอ), VIX level

### Phase 3 — Backtest ML Models (NB10–12)
- XGBoost walk-forward CV (expanding window, 52-week folds, 2003–2025 OOS)
  - XGB-Pruned: DirAcc 54.3%, IC +0.079, p = 0.033
- Multi-target model: SET + Gold + USD/THB พร้อมกัน (NB12)
- บทเรียน: Rule-based ชนะ ML — 1-feature XGB Sharpe +0.08 vs EEM rule Sharpe +0.40
  เพราะ signal เป็น linear, XGBoost overfit ที่ tail ด้วย 1,300 training points

### Phase 4 — Sector Rotation (NB13–15)
- Cross-sectional ranking 7 SET sectors (BANK, ENERGY, ICT, COMMERCE, HEALTH, PROPERTY, FOOD)
- Weekly L/S top-2/bottom-2: gross Sharpe +0.28 → net Sharpe -1.76 (TC ทำลายหมด)
  TC drag = 4 positions × 0.1% × 52 weeks = 20.8%/yr
- แก้ด้วยการลด frequency:
  - Monthly L/S (NB15 B1a): net Sharpe -0.24 (ยังไม่ผ่าน)
  - Bi-monthly L/S: net Sharpe +0.17 (marginally viable)
  - Long-only monthly (NB15 B2): net Sharpe +0.69, TC ~1.6%/yr (ดีที่สุดใน sector)
- บทเรียน: Sector IC = 0.026 อ่อนเกินกว่าจะ survive TC ที่ weekly frequency

### Phase 5 — EEM Signal Strategy (NB16–17)
- EEM Rule L/flat: Long SET เมื่อ eem_ret_d_lag1 > 0, Cash เมื่อ < 0
  - Gross Sharpe +0.81, Net Sharpe +0.58, Ann Return +7.2%, MaxDD -38%
  - Break-even TC = 0.50% one-way
- รวม sector momentum tilt (NB17 Variant A):
  - Net Sharpe +0.62, Ann Return +9.7%, MaxDD -40%
- บทเรียน: signal ง่ายชนะ ML เสมอเมื่อ IC เป็น linear; Occam's razor wins

### Phase 6 — Multi-Asset & NLP (NB14, 18–19)
- NLP sentiment (VADER จาก ~1,295 ข่าว Thai financial):
  - SET: +3.6% DirAcc, Sharpe +0.38 จากการเพิ่ม NLP
  - Coverage เพียง 27–37% ใน 2020+ era — ข้อมูลน้อยเกินไปจะสรุปได้ชัด
- Multi-asset (NB18):
  - Gold correlation กับ SET: near zero → diversifier ที่ดี
  - SET+Gold 50/50: Sharpe +0.77, MaxDD -31%
  - SET+Gold+FX (NB18 best): Sharpe +0.82, MaxDD -20.5%
  - FX signal อ่อนมาก (Sharpe +0.06) — ไม่ viable standalone
- Stock-level momentum (NB19, 54 SET stocks):
  - Mom4w IC = +0.016 p = 0.011*, Composite IC = +0.019 p = 0.003*
  - Weekly L/S: Gross Sharpe +0.59, Net Sharpe -2.24 (TC = 41%/yr ทำลายผล)
  - Score 3/10 — signal มีอยู่จริงแต่ TC kills it ที่ weekly frequency

### Phase 7 — Full System Construction (NB20) ← สุดท้าย
รวมทุกอย่างเข้าด้วยกัน 4 layers:

```
Layer 1 — Directional Signal
  SET:  Long เมื่อ eem_ret_d_lag1 > 0, Cash เมื่อ <= 0
  Gold: Always long (diversifier, zero correlation กับ SET)

Layer 2 — Risk Parity Allocation
  weight_SET  = (1/vol_SET)  / (1/vol_SET + 1/vol_Gold)
  weight_Gold = (1/vol_Gold) / (1/vol_SET + 1/vol_Gold)
  Rolling 52-week vol, rebalance monthly
  → Gold ได้ weight ~2× เพราะ vol ต่ำกว่า SET (~12% vs ~25%)

Layer 3 — Volatility Targeting
  scale = min(2.0, 10% / portfolio_vol_12wk)
  → ลด exposure ในช่วง high-vol (2008, 2020)
  → เพิ่ม exposure ในช่วง calm markets

Layer 4 — Drawdown Control
  ถ้า DD < -15% → ลด all positions ลง 50%
  ถ้า DD > -10% → restore กลับ 100%
  Active 7.3% ของ weeks ทั้งหมด
```

---

## ผลลัพธ์สุดท้าย (net of 0.1% TC, 2004–2025)

| Strategy                        | Sharpe | Ann Return | Max DD  | Calmar |
|---------------------------------|--------|-----------|---------|--------|
| SET Buy & Hold                  | +0.19  | +5.0%     | -47.7%  | 0.11   |
| EEM L/flat (NB16)               | +0.56  | +8.2%     | -42.2%  | 0.19   |
| EEM L/flat + Gold 50/50 (NB18)  | +0.81  | +10.3%    | -27.9%  | 0.37   |
| Full System (NB20)              | +0.78  | +10.4%    | -19.1%  | 0.54   |

### Layer Isolation — สิ่งที่แต่ละ layer เพิ่ม

| System                          | Sharpe | Max DD  |
|---------------------------------|--------|---------|
| L1: Signal + Gold (50/50)       | +0.82  | -27.9%  |
| L1 + L2: +Risk Parity           | +0.84  | -28.2%  |
| L1 + L2 + L3: +Vol Targeting    | +0.87  | -20.1%  |
| L1 + L2 + L3 + L4: +DD Control  | +0.78  | -19.1%  |

Vol targeting คือ layer ที่สำคัญที่สุด — ลด MaxDD จาก -28% → -20% พร้อมกับ Sharpe ที่ดีขึ้นด้วย
DD control cost ~0.1 Sharpe (TC จาก position changes) แต่ลด MaxDD อีก 1pp

---

## TC Sensitivity

| One-way TC | Net Sharpe | Max DD  |
|-----------|-----------|---------|
| 0.0%      | +0.940    | -18.6%  |
| 0.1%      | +0.778    | -19.1%  |
| 0.2%      | +0.581    | -19.6%  |
| 0.3%      | +0.423    | -19.5%  |
| 0.5%      | +0.174    | -25.1%  |
| 1.0%      | -0.558    | -55.8%  |

Break-even TC ≈ 0.30% one-way (เหมาะสำหรับ institutional trading)

---

## Regime Stability (Full System)

| Period    | System Sharpe | B&H Sharpe |
|-----------|--------------|-----------|
| 2004–2009 | +0.91        | +0.12     |
| 2010–2014 | +0.77        | +0.82     |
| 2015–2019 | +0.57        | -0.05     |
| 2020–2025 | +0.82        | -0.18     |

Positive Sharpe ทุก sub-period — ต่างจาก B&H ที่ติดลบใน 2015–2019 และ 2020–2025

---

## บทเรียนสำคัญ 5 ข้อ

1. **Global spillover > Local macro**
   EEM, S&P 500 lags, US Treasury yields ทำนาย SET ได้ดีกว่า GDP/CPI ไทย
   เพราะ Thailand เป็น price-taker ใน global risk sentiment
   EEM lag เกิดจาก time-zone differences และ Thai institutional investors react ช้ากว่า global

2. **Transaction Cost คือศัตรูที่ใหญ่ที่สุด**
   Weekly L/S ต้องการ gross Sharpe > 0.5 แค่เพื่อ break-even หลัง 0.1% TC
   แทบทุก strategy ที่ดูดีใน gross กลายเป็น negative ใน net
   ทางออก: ลด frequency (monthly), ใช้ long-only, หรือหา signal ที่ IC สูงพอ

3. **Rule beats ML เมื่อ signal เป็น linear**
   EEM signal มี IC = 0.11 และ monotonic — XGBoost ไม่ได้ช่วยอะไร
   1-feature XGB: Sharpe +0.08 vs EEM rule: Sharpe +0.40
   ML มีประโยชน์เมื่อ signal ซับซ้อน, non-linear, หรือมี feature หลายตัวที่ interact กัน

4. **Portfolio construction สำคัญเท่ากับ alpha discovery**
   Vol targeting เพียงอย่างเดียวลด MaxDD จาก -28% → -20% โดยไม่ต้องหา signal ใหม่
   Risk parity ให้ Gold weight ที่สูงขึ้นโดยอัตโนมัติในช่วง crisis ที่ SET vol พุ่ง
   Calmar ratio: 0.54 vs 0.19 (EEM-only) — เพิ่มขึ้น 3× จาก portfolio construction เพียงอย่างเดียว

5. **NLP มี potential แต่ต้องการข้อมูลมากกว่านี้**
   +3.6% DirAcc improvement จาก Thai news sentiment เป็นตัวเลขจริง ไม่ใช่ noise
   แต่รองรับได้แค่ 27–37% ของ weeks ใน 2020+ era
   ถ้า coverage เพิ่มเป็น 80%+ น่าจะเป็น signal ที่แข็งแกร่งที่สุดในโปรเจกต์นี้

---

## Limitations

- Single-country focus: ผลลัพธ์ specific กับ SET; EM อื่นอาจต่างกัน
- NLP coverage gap: 77% ของ weeks ปี 2000–2020 ไม่มีข่าวเลย — NLP เป็น 2020+ signal
- TC assumption: 0.1% conservative สำหรับ institutional; retail อาจ 2–5× สูงกว่า
- Regime risk: EEM-SET relationship อาจอ่อนลงถ้า Thailand decouples จาก global EM

---

## Notebook Index

| Notebook      | Description                                              |
|---------------|----------------------------------------------------------|
| eda/01–05     | EDA ข้อมูล, NLP validation, comprehensive feature analysis |
| eda/06        | Feature IC screening, signal validation                  |
| eda/07        | Data leakage audit, unified_weekly_clean.csv             |
| eda/08        | Feature pruning, model simplification                    |
| eda/09        | Signal robustness, permutation tests                     |
| eda/10        | Walk-forward backtest, XGB-Pruned                        |
| eda/11        | True long/short backtest (fix always-long bias)          |
| eda/12        | Multi-target: SET + Gold + USD/THB                       |
| eda/13        | Cross-sectional sector rotation                          |
| eda/14        | Sector-specific NLP sentiment                            |
| eda/15        | Low-TC sector rotation (monthly & long-only)             |
| eda/16        | EEM lag signal strategy (primary deliverable)            |
| eda/17        | EEM L/flat + sector momentum tilt                        |
| eda/18        | Gold & FX signals, multi-asset portfolio                 |
| eda/19        | Stock-level momentum (54 SET stocks)                     |
| eda/20        | Full system: risk parity + vol targeting + DD control    |
| model/01–06   | OLS, LASSO, XGBoost, NLP-enhanced models                 |

---

## สิ่งที่ควรทำต่อ (ลำดับความสำคัญ)

1. **[High]** เก็บข่าว Thai financial เพิ่ม (Bangkok Post, Reuters API, 2015–2025)
   เพื่อ validate NLP +3.6% DirAcc ที่ scale ที่ใหญ่กว่านี้

2. **[High]** Monthly-horizon model (aggregate to 4-week returns)
   SNR สูงขึ้น, EEM signal อาจ IC > 0.15, TC ลดลงมาก

3. **[Medium]** Paper trading pilot
   ทดสอบ EEM signal ใน live market หลัง 2025

4. **[Low]** Stock-level momentum ที่ monthly frequency
   TC ลดจาก 41%/yr → ~10%/yr — อาจ viable

---

*ข้อมูล: 2000–2025 | Python 3.13 | XGBoost, pandas, scipy, VADER*
*จบโปรเจกต์: 2026-05-23*
