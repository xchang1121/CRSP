"""
Paper Reference: Section 3.2, Section 5.2, Table 5, Fig. 7–8 (5-Domain Private Docs & v4.2->v5.0).
Extended Zero-Leakage Vertical-Domain Document Continual Learning Benchmark Suite:
  - 5 Strictly Zero-Leakage Vertical Domains:
      A_heliospay (FinTech), B_nvx418 (BioPharma), C_lithox9 (Semiconductor EUV),
      D_gridops (GridPulse-9 Smart Grid SLA), E_customs (Meridian-C3 Customs & Tariff Code)
  - Document Version Upgrade: A_heliospay_v50 (mutating 6 rules/constants from v4.2)
    with BOTH v5.0 accuracy grader AND v4.2 stale-knowledge leakage detector!
  - Deep Generalization Stress Batteries:
      1) Boundary / Exact Threshold Equality (amount=150k/420k, R=27/33/56/76)
      2) Out-of-Range Numerical Extrapolation (extreme wire amounts never seen in training)
      3) Cross-Document Multi-Hop Reasoning (Doc A + Doc B combined in one query!)
      4) Negative / Unanswerable Abstention Stress Test (confabulation vs abstention)
  - Compact Micro-RAG Fact Cards (~10% of full document tokens) for Hybrid CSPD-Doc + Micro-RAG
  - Domain Descriptor Cards (for Zero-Oracle Automatic Routing) & Diverse Fact Paraphrases
"""

import random
import re
from typing import Any, Dict, List, Tuple

from domain_docs_data import (
    DOC_A_TEXT,
    DOC_B_TEXT,
    DOC_C_TEXT,
    FACTS_A,
    FACTS_B,
    FACTS_C,
    _solve_doc_a,
    _solve_doc_b,
    build_domain_datasets,
    grade_domain_response,
)

# ===========================================================================
# 1. Two New Zero-Leakage Vertical Domains (Doc D & Doc E) + Doc A v5.0
# ===========================================================================

DOC_D_TEXT = """# GridPulse-9 Regional Smart Grid Dispatch & Reliability SLA

## 1. Governance & System Constants
- Control Platform: GridPulse-9 (Firmware GP-SCADA 5.8.1)
- Chief Grid Dispatcher: Tomasz Zielinski
- Nominal Grid Frequency: 50.00 Hz (Emergency Shedding Floor: 49.50 Hz)
- Frequency Dip Stress Trigger: < 49.85 Hz
- Solar Ramp Drop Trigger: > 30% output loss within 15 minutes
- Battery High-SOC Relief Threshold: >= 80% State of Charge
- Black-Start Recovery Cap: 45 minutes
- Substation Telemetry Interval: 250 milliseconds
- Fault Code GP-901: Under-Frequency Load Shedding Abort (freq < 49.50 Hz)
- Fault Code GP-440: Inverter Harmonic Drift (THD > 3.8%)
- Fault Code GP-712: Tie-Line Thermal Overload (> 1,450 MVA)

## 2. Balancing Zone Base Parameters
- Zone NORTH-Z2: Base Reserve = 180 MW, Base Tariff = 42 USD/MWh
- Zone COAST-Z5: Base Reserve = 240 MW, Base Tariff = 58 USD/MWh
- Zone DESERT-Z8: Base Reserve = 310 MW, Base Tariff = 71 USD/MWh

## 3. Stress Index & Dispatch Priority Tree
Compute integer Net Stress Index S (in MW):
  S = Base_Reserve(zone)
      + 95 if freq_hz < 49.85 else 0
      + 60 if solar_drop_pct > 30 else 0
      - 40 if battery_soc_pct >= 80 else 0

Apply Dispatch Mode in strict priority order:
1. If freq_hz < 49.50:
   -> Mode = SHED-GP901, Tariff = 0 USD/MWh (emergency load shedding; notify Tomasz Zielinski).
2. Else if S >= 380:
   -> Mode = EMERGENCY-PEAKER, Tariff = Base_Tariff + 65 USD/MWh.
3. Else if 260 <= S < 380:
   -> Mode = SPIN-RESERVE, Tariff = Base_Tariff + 28 USD/MWh.
4. Else (S < 260):
   -> Mode = NOMINAL-GRID, Tariff = Base_Tariff USD/MWh.
"""

DOC_E_TEXT = """# Meridian-C3 Cross-Border Customs Clearance & Tariff Code

## 1. Authority Constants & Compliance Codes
- Regulatory Code: Meridian-C3 (Portal: MERIDIAN-EDI 3.9)
- Port Authority Director: Nadia Benali
- Customs Bond Multiplier: 2.4x declared invoice value
- High-Value Surcharge Threshold: > 250,000 USD
- Mandatory Inspection Hold Window: 72 hours
- Archive Retention Window: 7.5 years
- Seizure Code MC-88: Dual-Use Tier-2 Embargo Seizure
- Audit Code MC-41: Certificate of Origin Discrepancy
- Penalty Code MC-63: Late Manifest Filing (> 6 hours post-arrival)

## 2. Commodity Class Base Parameters
- Class CRYO-M1: Base Duty = 6%
- Class BIO-R4: Base Duty = 11%
- Class OPTIC-K8: Base Duty = 14%

## 3. Effective Duty & Clearance Channel Tree
Compute integer Effective Duty Percentage D_eff (%):
  D_eff = Base_Duty(class)
          + 8 if origin_tier == 2 else 0
          + 4 if value_USD > 250,000 else 0
          - 3 if aeo_certified else 0

Apply Clearance Channel in strict priority order:
1. If dual_use_flag is True AND origin_tier == 2:
   -> Channel = SEIZE-MC88, Duty = 0% (confiscate cargo; escalate to Nadia Benali).
2. Else if D_eff >= 18:
   -> Channel = RED-CHANNEL, Duty = D_eff%.
3. Else if 11 <= D_eff < 18:
   -> Channel = AMBER-AUDIT, Duty = D_eff%.
4. Else (D_eff < 11):
   -> Channel = GREEN-EXPRESS, Duty = D_eff%.
"""

DOC_A_V50_TEXT = """# HeliosPay v5.0 Internal Cross-Border Settlement & Compliance Specification (SUPERSEDES v4.2)

## 1. Updated Governance & Constants (v5.0 Changes)
- System Version: HeliosPay v5.0 (Production Core)
- Daily Settlement Cutoff: 17:15 SGT (updated from 16:40 SGT)
- Mandatory Reserve Buffer: 16.0% of rolling 30-day corridor volume
- Cold Vault Quorum: 5-of-9 hardware security keys
- Regulatory Audit Retention Period: 11.0 years
- Lead Compliance Officer: Viktor Lindqvist
- Standard Escalation SLA: 15 minutes
- Maximum Single Instant Wire Limit: 500,000 USD

## 2. Updated Corridor Base Parameters (v5.0)
- Corridor APAC-B7: Base Risk = 20, Base Fee = 46 bps
- Corridor EMEA-K3: Base Risk = 24, Base Fee = 60 bps
- Corridor LATAM-Q9: Base Risk = 35, Base Fee = 82 bps

## 3. Updated Risk Scoring & Approval Tier Decision Rules (v5.0)
Compute integer Risk Score R:
  R = Base_Risk(corridor)
      + 18 if amount_USD > 200,000 else 0
      + 25 if beneficiary is NEW else 0
      + 20 if submitted AFTER 17:15 SGT cutoff else 0
      - 15 if sender is Tier-1 Verified else 0

Apply Approval Tier in strict priority order:
1. If corridor is LATAM-Q9 AND amount_USD > 500,000:
   -> Tier = FREEZE-HP6205, Total Fee = 0 bps (notify Viktor Lindqvist).
2. Else if R >= 75:
   -> Tier = FREEZE-HP4172, Total Fee = 0 bps (notify Viktor Lindqvist).
3. Else if 55 <= R < 75:
   -> Tier = L2-COMMITTEE, Total Fee = Base_Fee + 30 bps.
4. Else if 30 <= R < 55:
   -> Tier = L1-ANALYST, Total Fee = Base_Fee + 15 bps.
5. Else (R < 30):
   -> Tier = AUTO-CLEAR, Total Fee = Base_Fee bps.
"""

ALL_5_DOCS = {
    "A_heliospay": DOC_A_TEXT,
    "B_nvx418": DOC_B_TEXT,
    "C_lithox9": DOC_C_TEXT,
    "D_gridops": DOC_D_TEXT,
    "E_customs": DOC_E_TEXT,
}

MICRO_RAG_CARDS = {
    "A_heliospay": (
        "HeliosPay v4.2 Key Constants: Cutoff=16:40 SGT; Reserve=14.5%; Quorum=4-of-7; "
        "Retention=9.5 years; Officer=Dr. Maren Lindholm; SLA=22 minutes; MaxWire=420,000 USD; "
        "Corridors(Risk/Fee): APAC-B7(12/38 bps), EMEA-K3(18/52 bps), LATAM-Q9(31/74 bps); "
        "Codes: HP-4172=Velocity, HP-5089=Corridor Liquidity Halt, HP-3314=Beneficiary Sanctions Mismatch, "
        "HP-6205=FX Margin Breach, HP-2901=Dual-Key Signature Expiry."
    ),
    "B_nvx418": (
        "NVX-418 Key Constants: Generic=Celviranib; Protocol=VR-2026-418B; Sponsor=Virelia Therapeutics; "
        "PI=Prof. Kenji Takahashi; Lab=NordicBio-Oslo; HalfLife=18.4 hours; Storage=2.5 C to 6.0 C (max excursion 45 min); "
        "Biomarker=sTREM-42; Stop eGFR<38 mL/min; Antidote=Thaloxene-7 at 3.2 mg/kg IV."
    ),
    "C_lithox9": (
        "Litho-X9 Key Constants: Firmware=EUV-OS 7.3.4; Wavelength=13.52 nm; MaxPellicle=415 K; "
        "VacuumFloor=1.8e-6 Pa; Engineer=Elena Kowalski; PM=140,000 wafers; PowerCeiling=365 W; "
        "Throughput=185 wph nominal (115 wph capped); Codes: LX-904=Stage Tilt, "
        "LX-812=Collector Mirror Haze Alarm (>340 W & >4.5%), LX-771=Tin Droplet Generator Jitter, LX-639=Reticle Thermal Bow."
    ),
    "D_gridops": (
        "GridPulse-9 Key Constants: Firmware=GP-SCADA 5.8.1; Dispatcher=Tomasz Zielinski; ShedFloor=49.50 Hz; "
        "DipTrigger<49.85 Hz; SolarDrop>30%; SOCRelief>=80%; BlackStart=45 minutes; Telemetry=250 ms; "
        "Zones(Res/Tariff): NORTH-Z2(180 MW/42), COAST-Z5(240 MW/58), DESERT-Z8(310 MW/71); "
        "Codes: GP-901=Load Shedding, GP-440=Inverter Harmonic Drift (THD>3.8%), GP-712=Tie-Line Overload (>1,450 MVA)."
    ),
    "E_customs": (
        "Meridian-C3 Key Constants: Portal=MERIDIAN-EDI 3.9; Director=Nadia Benali; Bond=2.4x; "
        "HighValue>250,000 USD; Hold=72 hours; Retention=7.5 years; Classes: CRYO-M1(6%), BIO-R4(11%), OPTIC-K8(14%); "
        "Codes: MC-88=Dual-Use Seizure, MC-41=Certificate of Origin Discrepancy, MC-63=Late Manifest (>6 hrs)."
    ),
}

DOMAIN_DESCRIPTORS = {
    "A_heliospay": "HeliosPay v4.2 cross-border wire transfer settlement compliance risk score corridor APAC-B7 EMEA-K3 LATAM-Q9 fee bps approval tier HP-4172 HP-5089 HP-3314 HP-6205 HP-2901 Maren Lindholm cutoff SGT vault quorum reserve buffer",
    "B_nvx418": "NVX-418 Celviranib Phase IIb clinical trial SOP dosing protocol Virelia Therapeutics Kenji Takahashi NordicBio-Oslo eGFR sTREM-42 ALT CYP3A4 inhibitor Thaloxene-7 VR-2026-418B cold-chain patient",
    "C_lithox9": "Litho-X9 EUV lithography yield diagnostics manual overlay error E_eff pellicle temperature chamber vacuum Pa offset EUV-OS 7.3.4 Elena Kowalski LX-904 LX-812 LX-771 LX-639 wafer wph laser",
    "D_gridops": "GridPulse-9 regional smart grid dispatch reliability SLA balancing zone NORTH-Z2 COAST-Z5 DESERT-Z8 stress index tariff USD/MWh frequency Hz solar drop battery SOC GP-901 GP-440 GP-712 Tomasz Zielinski GP-SCADA",
    "E_customs": "Meridian-C3 cross-border customs clearance tariff code commodity class CRYO-M1 BIO-R4 OPTIC-K8 effective duty percentage origin tier AEO certified dual-use SEIZE-MC88 RED-CHANNEL AMBER-AUDIT GREEN-EXPRESS Nadia Benali MERIDIAN-EDI manifest bond",
}

OOD_24_QUESTIONS = [
    ("What is the capital of Australia?", ["canberra"]),
    ("What is 17 multiplied by 23?", ["391"]),
    ("Who wrote the novel 'Pride and Prejudice'?", ["jane austen", "austen"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("In what year did World War II end?", ["1945"]),
    ("What planet is closest to the Sun?", ["mercury"]),
    ("What is the square root of 144?", ["12"]),
    ("Which gas makes up roughly 78% of Earth's atmosphere?", ["nitrogen"]),
    ("Who painted the Mona Lisa?", ["da vinci", "leonardo"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("What is the largest mammal on Earth?", ["blue whale"]),
    ("What is the boiling point of water at sea level in Celsius?", ["100"]),
    ("Which programming language was created by Guido van Rossum?", ["python"]),
    ("What is the derivative of x^3 with respect to x?", ["3x^2", "3*x^2"]),
    ("What is the powerhouse of the cell?", ["mitochondria", "mitochondrion"]),
    ("Which ocean is the largest on Earth?", ["pacific"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("What is the speed of light in vacuum approximately in km/s?", ["300,000", "299,792", "3e5", "300000"]),
    ("Who formulated the theory of general relativity?", ["einstein"]),
    ("What is the currency of Japan?", ["yen"]),
    ("What is 2 to the power of 10?", ["1024", "1,024"]),
    ("Which element has atomic number 6?", ["carbon"]),
    ("What is the longest river in South America?", ["amazon"]),
    ("In binary, what decimal number does 1011 represent?", ["11"]),
]

# Diverse generic null-replay prompts (strictly disjoint from OOD_24_QUESTIONS and all vertical docs!)
NULL_REPLAY_PROMPTS = [
    "Explain in two sentences how photosynthesis converts sunlight into energy.",
    "What is 14 * 19? Show your calculation briefly.",
    "Name the three states of matter most commonly encountered on Earth.",
    "Give a brief definition of supply and demand in economics.",
    "What is the freezing point of water in Fahrenheit?",
    "Summarize why regular sleep is important for memory consolidation.",
    "What is the sum of interior angles of a pentagon?",
    "Name two primary causes of ocean tides on Earth.",
    "Convert 45 kilometers to meters.",
    "What is the main function of red blood cells?",
    "Explain the difference between HTTP and HTTPS in one sentence.",
    "What is 35% of 240?",
    "Name the largest desert in Africa.",
    "What does DNA stand for?",
    "Briefly explain what a prime number is and give three examples.",
    "What is the capital city of Canada?",
    "How many minutes are in 3.5 hours?",
    "What instrument measures atmospheric pressure?",
    "Briefly describe the water cycle in three steps.",
    "What is the area of a triangle with base 12 and height 9?",
]

FACTS_D = [
    ("What firmware version runs on the GridPulse-9 control platform?",
     "Specify the exact SCADA firmware version for GridPulse-9.",
     "GP-SCADA 5.8.1", ["5.8.1"]),
    ("Who is the Chief Grid Dispatcher for GridPulse-9?",
     "Which dispatcher must be notified during a SHED-GP901 emergency in GridPulse-9?",
     "Tomasz Zielinski", ["tomasz zielinski"]),
    ("What is the emergency load shedding frequency floor in GridPulse-9?",
     "Below what grid frequency in Hz does GridPulse-9 trigger SHED-GP901?",
     "49.50 Hz", ["49.5"]),
    ("What is the black-start recovery cap in GridPulse-9?",
     "How many minutes is the maximum black-start recovery time under GridPulse-9 SLA?",
     "45 minutes", ["45"]),
    ("What is the substation telemetry interval for GridPulse-9?",
     "How frequently (in milliseconds) does GridPulse-9 sample substation telemetry?",
     "250 milliseconds", ["250"]),
    ("In GridPulse-9, what fault does code GP-440 indicate?",
     "Which grid anomaly triggers GridPulse-9 code GP-440?",
     "Inverter Harmonic Drift (THD > 3.8%)", ["harmonic drift", "3.8"]),
    ("In GridPulse-9, what does fault code GP-712 signify?",
     "What thermal threshold corresponds to GridPulse-9 fault code GP-712?",
     "Tie-Line Thermal Overload (> 1,450 MVA)", ["1,450", "1450"]),
    ("In GridPulse-9, what are the base reserve and base tariff for zone NORTH-Z2?",
     "State the base reserve in MW and base tariff in USD/MWh for GridPulse-9 zone NORTH-Z2.",
     "Base Reserve 180 MW, Base Tariff 42 USD/MWh", ["180", "42"]),
    ("In GridPulse-9, what are the base reserve and base tariff for zone DESERT-Z8?",
     "State the base reserve in MW and base tariff in USD/MWh for GridPulse-9 zone DESERT-Z8.",
     "Base Reserve 310 MW, Base Tariff 71 USD/MWh", ["310", "71"]),
    ("In GridPulse-9, at what battery SOC percentage is the -40 MW stress relief applied?",
     "What minimum battery State of Charge percentage grants a 40 MW stress reduction in GridPulse-9?",
     "80%", ["80"]),
]

FACTS_E = [
    ("What EDI portal version is used by the Meridian-C3 customs code?",
     "Name the exact electronic data interchange portal release specified in Meridian-C3.",
     "MERIDIAN-EDI 3.9", ["3.9"]),
    ("Who is the Port Authority Director in Meridian-C3?",
     "To whom must a SEIZE-MC88 dual-use confiscation be escalated under Meridian-C3?",
     "Nadia Benali", ["nadia benali"]),
    ("What is the customs bond multiplier in Meridian-C3?",
     "By what multiple of declared invoice value is the Meridian-C3 customs bond computed?",
     "2.4x", ["2.4"]),
    ("What is the mandatory inspection hold window in Meridian-C3?",
     "How many hours is the standard inspection hold period under Meridian-C3?",
     "72 hours", ["72"]),
    ("What is the archive retention window in Meridian-C3?",
     "For how many years must Meridian-C3 customs filings be archived?",
     "7.5 years", ["7.5"]),
    ("What high-value surcharge threshold applies in Meridian-C3?",
     "Above what invoice USD value does Meridian-C3 add a +4% duty surcharge?",
     "250,000 USD", ["250,000", "250000"]),
    ("In Meridian-C3, what does audit code MC-41 indicate?",
     "Which discrepancy triggers Meridian-C3 code MC-41?",
     "Certificate of Origin Discrepancy", ["origin discrepancy"]),
    ("In Meridian-C3, what triggers penalty code MC-63?",
     "Which manifest violation corresponds to Meridian-C3 code MC-63?",
     "Late Manifest Filing (> 6 hours post-arrival)", ["late manifest", "6"]),
    ("What is the base duty percentage for commodity class CRYO-M1 in Meridian-C3?",
     "State the Meridian-C3 base duty rate for CRYO-M1 goods.",
     "6%", ["6%"]),
    ("What is the base duty percentage for commodity class OPTIC-K8 in Meridian-C3?",
     "State the Meridian-C3 base duty rate for OPTIC-K8 goods.",
     "14%", ["14%"]),
]

FACTS_A_V50 = [
    ("In HeliosPay v5.0, what is the daily settlement cutoff time?",
     "What time is the updated daily settlement cutoff for HeliosPay v5.0?",
     "17:15 SGT", ["17:15"]),
    ("In HeliosPay v5.0, what is the mandatory reserve buffer percentage?",
     "What percentage of rolling 30-day volume is required as reserve buffer in HeliosPay v5.0?",
     "16.0%", ["16"]),
    ("Who is the Lead Compliance Officer for HeliosPay v5.0?",
     "To which compliance officer must frozen wires in HeliosPay v5.0 be escalated?",
     "Viktor Lindqvist", ["viktor lindqvist"]),
    ("What is the Cold Vault Quorum required by HeliosPay v5.0?",
     "How many hardware security keys form the Cold Vault Quorum in HeliosPay v5.0?",
     "5-of-9", ["5-of-9", "5 of 9"]),
    ("What is the regulatory audit retention period in HeliosPay v5.0?",
     "For how many years must HeliosPay v5.0 audit records be retained?",
     "11.0 years", ["11"]),
    ("What is the standard escalation SLA in HeliosPay v5.0?",
     "How many minutes is the escalation SLA in HeliosPay v5.0?",
     "15 minutes", ["15"]),
    ("What is the maximum single instant wire limit in HeliosPay v5.0?",
     "Above what USD amount does a single wire exceed the HeliosPay v5.0 limit?",
     "500,000 USD", ["500,000", "500000"]),
    ("In HeliosPay v5.0, what are the base risk and base fee for corridor APAC-B7?",
     "State the updated base risk and base fee in bps for HeliosPay v5.0 corridor APAC-B7.",
     "Base Risk 20, Base Fee 46 bps", ["20", "46"]),
]

V42_STALE_PATTERNS = ["16:40", "14.5%", "maren lindholm", "4-of-7", "9.5", "22 min", "420,000"]


def _solve_doc_d(zone: str, freq_hz: float, solar_drop_pct: int, battery_soc_pct: int) -> Dict[str, Any]:
    br, bt = {"NORTH-Z2": (180, 42), "COAST-Z5": (240, 58), "DESERT-Z8": (310, 71)}[zone]
    df = 95 if freq_hz < 49.85 else 0
    ds = 60 if solar_drop_pct > 30 else 0
    db = -40 if battery_soc_pct >= 80 else 0
    S = br + df + ds + db
    if freq_hz < 49.50:
        mode, tariff = "SHED-GP901", 0
    elif S >= 380:
        mode, tariff = "EMERGENCY-PEAKER", bt + 65
    elif S >= 260:
        mode, tariff = "SPIN-RESERVE", bt + 28
    else:
        mode, tariff = "NOMINAL-GRID", bt
    q = (
        f"[GridPulse-9 Dispatch SLA] Evaluate this regional grid telemetry snapshot:\n"
        f"- Balancing Zone: {zone}\n- Grid Frequency: {freq_hz:.2f} Hz\n"
        f"- 15-min Solar Drop: {solar_drop_pct}%\n- Battery SOC: {battery_soc_pct}%\n"
        f"Compute the Net Stress Index S, Dispatch Mode, and Tariff (USD/MWh). End with exactly:\n"
        f"Stress Index: <S> MW\nMode: <MODE>\nTariff: <TARIFF> USD/MWh"
    )
    a = (
        f"Under GridPulse-9, zone {zone} has Base Reserve {br} MW and Base Tariff {bt} USD/MWh.\n"
        f"Adjustments: freq<49.85Hz ({df:+d}), solar_drop>30% ({ds:+d}), battery_soc>=80% ({db:+d}).\n"
        f"Therefore S = {S} MW. Priority tree yields Mode = {mode} and Tariff = {tariff} USD/MWh.\n"
        f"Stress Index: {S} MW\nMode: {mode}\nTariff: {tariff} USD/MWh"
    )
    return {"question": q, "gold_answer": a, "expected": {"S": S, "mode": mode, "tariff": tariff}, "kind": "rule", "domain": "D_gridops"}


def _solve_doc_e(cls: str, origin_tier: int, value_usd: int, aeo: bool, dual_use: bool) -> Dict[str, Any]:
    bd = {"CRYO-M1": 6, "BIO-R4": 11, "OPTIC-K8": 14}[cls]
    do = 8 if origin_tier == 2 else 0
    dv = 4 if value_usd > 250000 else 0
    da = -3 if aeo else 0
    d_eff = bd + do + dv + da
    if dual_use and origin_tier == 2:
        ch, duty = "SEIZE-MC88", 0
    elif d_eff >= 18:
        ch, duty = "RED-CHANNEL", d_eff
    elif d_eff >= 11:
        ch, duty = "AMBER-AUDIT", d_eff
    else:
        ch, duty = "GREEN-EXPRESS", d_eff
    q = (
        f"[Meridian-C3 Customs Clearance] Evaluate this customs declaration:\n"
        f"- Commodity Class: {cls}\n- Origin Sanction Tier: Tier-{origin_tier}\n"
        f"- Invoice Value: {value_usd:,} USD\n- AEO Certified: {'Yes' if aeo else 'No'}\n"
        f"- Dual-Use Flag: {'Yes' if dual_use else 'No'}\n"
        f"Compute Effective Duty, Clearance Channel, and Final Duty (%). End with exactly:\n"
        f"Effective Duty: <D>%\nChannel: <CHANNEL>\nFinal Duty: <DUTY>%"
    )
    a = (
        f"Under Meridian-C3, class {cls} has Base Duty {bd}%.\n"
        f"Adjustments: origin_tier==2 ({do:+d}%), value>250k ({dv:+d}%), AEO ({da:+d}%).\n"
        f"Therefore D_eff = {d_eff}%. Priority tree yields Channel = {ch} and Final Duty = {duty}%.\n"
        f"Effective Duty: {d_eff}%\nChannel: {ch}\nFinal Duty: {duty}%"
    )
    return {"question": q, "gold_answer": a, "expected": {"d_eff": d_eff, "channel": ch, "duty": duty}, "kind": "rule", "domain": "E_customs"}


def _solve_doc_a_v50(corridor: str, amount: int, new_ben: bool, after_cut: bool, tier1: bool) -> Dict[str, Any]:
    br, bf = {"APAC-B7": (20, 46), "EMEA-K3": (24, 60), "LATAM-Q9": (35, 82)}[corridor]
    da = 18 if amount > 200000 else 0
    dn = 25 if new_ben else 0
    dc = 20 if after_cut else 0
    dt = -15 if tier1 else 0
    r = br + da + dn + dc + dt
    if corridor == "LATAM-Q9" and amount > 500000:
        tier, fee = "FREEZE-HP6205", 0
    elif r >= 75:
        tier, fee = "FREEZE-HP4172", 0
    elif r >= 55:
        tier, fee = "L2-COMMITTEE", bf + 30
    elif r >= 30:
        tier, fee = "L1-ANALYST", bf + 15
    else:
        tier, fee = "AUTO-CLEAR", bf
    old_item = _solve_doc_a(corridor, amount, new_ben, after_cut, tier1)
    q = (
        f"[HeliosPay v5.0 Wire Assessment] Evaluate the following cross-border wire:\n"
        f"- Corridor: {corridor}\n- Amount: {amount:,} USD\n"
        f"- Beneficiary status: {'NEW first-time beneficiary' if new_ben else 'Existing beneficiary'}\n"
        f"- Submission time: {'17:45 SGT (after cutoff)' if after_cut else '16:50 SGT (before cutoff)'}\n"
        f"- Sender status: {'Tier-1 Verified' if tier1 else 'Standard (non-Tier-1)'}\n"
        f"Compute the Risk Score R, Approval Tier, and Total Fee (in bps). End with exactly:\n"
        f"Risk Score: <R>\nTier: <TIER>\nTotal Fee: <FEE> bps"
    )
    a = (
        f"Under HeliosPay v5.0, corridor {corridor} has Base Risk {br} and Base Fee {bf} bps.\n"
        f"Adjustments: amount>200k ({da:+d}), new beneficiary ({dn:+d}), after 17:15 SGT ({dc:+d}), Tier-1 Verified ({dt:+d}).\n"
        f"Therefore R = {r}. Applying the v5.0 priority tree yields Tier = {tier} and Total Fee = {fee} bps.\n"
        f"Risk Score: {r}\nTier: {tier}\nTotal Fee: {fee} bps"
    )
    return {
        "question": q, "gold_answer": a,
        "expected": {"R": r, "tier": tier, "fee": fee},
        "stale_v42": old_item["expected"],
        "kind": "rule", "domain": "A_heliospay_v50",
    }


def _make_fact_splits(facts_list, domain_name: str, doc_label: str, augment_paraphrases: bool = False):
    train_facts, test_facts = [], []
    prefixes = [
        "",
        "Quick compliance check: ",
        "Please state the exact policy value: ",
    ]
    for q_tr, q_te, ans, kws in facts_list:
        if augment_paraphrases:
            for pfx in prefixes:
                train_facts.append({
                    "question": pfx + q_tr,
                    "gold_answer": f"According to {doc_label}, the answer is: {ans}.",
                    "expected": {"keywords": kws},
                    "kind": "fact", "domain": domain_name,
                })
        else:
            train_facts.append({
                "question": q_tr,
                "gold_answer": f"According to {doc_label}, the answer is: {ans}.",
                "expected": {"keywords": kws},
                "kind": "fact", "domain": domain_name,
            })
        test_facts.append({
            "question": q_te, "gold_answer": ans,
            "expected": {"keywords": kws}, "kind": "fact", "domain": domain_name,
        })
    return train_facts, test_facts


def build_all_5_datasets(seed: int = 42) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out = build_domain_datasets(seed=seed)
    rng = random.Random(seed + 100)

    # --- Domain D ---
    all_d = []
    for zone in ["NORTH-Z2", "COAST-Z5", "DESERT-Z8"]:
        for freq in [49.40, 49.70, 49.95, 50.02]:
            for sol in [10, 25, 38, 52]:
                for soc in [50, 85]:
                    all_d.append(_solve_doc_d(zone, freq, sol, soc))
    rng.shuffle(all_d)
    tr_f_d, te_f_d = _make_fact_splits(FACTS_D, "D_gridops", "GridPulse-9")
    out["D_gridops"] = {"train": all_d[24:104] + tr_f_d * 4, "test_rules": all_d[:24], "test_facts": te_f_d}

    # --- Domain E ---
    all_e = []
    for cls in ["CRYO-M1", "BIO-R4", "OPTIC-K8"]:
        for orig in [1, 2]:
            for val in [80000, 180000, 310000, 490000]:
                for aeo in [False, True]:
                    for dual in [False, True]:
                        all_e.append(_solve_doc_e(cls, orig, val, aeo, dual))
    rng.shuffle(all_e)
    tr_f_e, te_f_e = _make_fact_splits(FACTS_E, "E_customs", "Meridian-C3")
    out["E_customs"] = {"train": all_e[24:104] + tr_f_e * 4, "test_rules": all_e[:24], "test_facts": te_f_e}
    return out


def build_v50_dataset(seed: int = 150) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    all_v5 = []
    for corr in ["APAC-B7", "EMEA-K3", "LATAM-Q9"]:
        for amt in [120000, 175000, 260000, 450000, 560000]:
            for nb in [False, True]:
                for ac in [False, True]:
                    for t1 in [False, True]:
                        all_v5.append(_solve_doc_a_v50(corr, amt, nb, ac, t1))
    rng.shuffle(all_v5)
    tr_f, te_f = _make_fact_splits(FACTS_A_V50, "A_heliospay_v50", "HeliosPay v5.0")
    return {"train": all_v5[24:104] + tr_f * 4, "test_rules": all_v5[:24], "test_facts": te_f}


def build_boundary_and_extrap_A() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    boundary_specs = [
        ("APAC-B7", 150000, False, False, False),
        ("APAC-B7", 150001, False, False, False),
        ("LATAM-Q9", 420000, False, False, True),
        ("LATAM-Q9", 420001, False, False, True),
        ("EMEA-K3", 100000, False, True, True),
        ("EMEA-K3", 160000, False, False, False),
        ("LATAM-Q9", 100000, True, False, False),
        ("LATAM-Q9", 200000, True, True, True),
        ("APAC-B7", 200000, True, True, False),
        ("EMEA-K3", 200000, True, True, False),
        ("APAC-B7", 150000, True, False, True),
        ("LATAM-Q9", 150000, False, False, True),
    ]
    extrap_specs = [
        ("APAC-B7", 950000, False, False, True),
        ("EMEA-K3", 1250000, True, False, False),
        ("LATAM-Q9", 2500000, False, False, True),
        ("APAC-B7", 3500, True, True, False),
        ("EMEA-K3", 890000, False, True, True),
        ("LATAM-Q9", 12500, True, True, False),
        ("APAC-B7", 7500000, True, True, True),
        ("EMEA-K3", 4900, False, False, False),
    ]
    def make(specs):
        items = []
        for s in specs:
            it = _solve_doc_a(*s)
            it["kind"] = "rule"
            it["domain"] = "A_heliospay"
            items.append(it)
        return items
    return make(boundary_specs), make(extrap_specs)


def build_cross_doc_AB_tests() -> List[Dict[str, Any]]:
    specs = [
        ("APAC-B7", 120000, False, False, True,   80, 70, 50, False, 300, 1.5),
        ("EMEA-K3", 200000, True, False, False,   70, 45, 72, False, 400, 1.8),
        ("LATAM-Q9", 100000, False, True, False,  90, 65, 55, True,  600, 2.0),
        ("APAC-B7", 250000, True, True, False,    60, 32, 60, False, 200, 1.0),
        ("EMEA-K3", 80000, False, False, True,    84, 72, 45, False, 900, 2.0),
        ("LATAM-Q9", 480000, False, False, True,  100, 80, 40, False, 250, 1.2),
        ("APAC-B7", 180000, False, True, True,    64, 50, 75, True,  350, 1.5),
        ("EMEA-K3", 300000, True, True, False,    76, 62, 68, False, 700, 2.5),
        ("LATAM-Q9", 90000, False, False, True,   88, 42, 60, False, 300, 6.0),
        ("APAC-B7", 50000, True, False, False,    96, 85, 72, True,  400, 1.4),
        ("EMEA-K3", 170000, False, True, False,   72, 48, 52, False, 550, 1.9),
        ("LATAM-Q9", 210000, True, False, True,   110, 75, 48, False, 200, 1.1),
    ]
    out = []
    for corr, amt, nb, ac, t1, wt, egfr, age, cyp, strem, alt in specs:
        ia = _solve_doc_a(corr, amt, nb, ac, t1)
        ib = _solve_doc_b(wt, egfr, age, cyp, strem, alt)
        q = (
            f"[Combined HeliosPay v4.2 + NVX-418 Assessment] Apply BOTH policies below:\n"
            f"Part 1 (HeliosPay v4.2 Wire): Corridor={corr}, Amount={amt:,} USD, "
            f"Beneficiary={'NEW first-time beneficiary' if nb else 'Existing beneficiary'}, "
            f"Submitted={'17:15 SGT (after cutoff)' if ac else '14:20 SGT (before cutoff)'}, "
            f"Sender={'Tier-1 Verified' if t1 else 'Standard'}.\n"
            f"Part 2 (NVX-418 Clinical Dosing): Weight={wt} kg, eGFR={egfr} mL/min, Age={age} years, "
            f"CYP3A4 inhibitor={'Yes (Ketoconazole)' if cyp else 'No'}, sTREM-42={strem} pg/mL, ALT={alt}x ULN.\n"
            f"End your response with exactly:\n"
            f"Risk Score: <R>\nTier: <TIER>\nTotal Fee: <FEE> bps\nAction: <ACTION>\nFinal Dose: <DOSE> mg"
        )
        out.append({
            "question": q,
            "item_A": {"expected": ia["expected"], "kind": "rule", "domain": "A_heliospay"},
            "item_B": {"expected": ib["expected"], "kind": "rule", "domain": "B_nvx418"},
            "gold_answer": ia["gold_answer"] + "\n" + ib["gold_answer"],
        })
    return out


ABSTENTION_TRAIN_A = [
    {"question": "In HeliosPay v4.2, what is the cryptocurrency staking yield?",
     "gold_answer": "That information is not specified in the HeliosPay v4.2 specification.", "kind": "abstain", "domain": "A_heliospay"},
    {"question": "What is the base fee for corridor NORDIC-X1 in HeliosPay v4.2?",
     "gold_answer": "Corridor NORDIC-X1 is not defined in the HeliosPay v4.2 specification.", "kind": "abstain", "domain": "A_heliospay"},
    {"question": "Who is the Chief Marketing Officer listed in HeliosPay v4.2?",
     "gold_answer": "A Chief Marketing Officer is not specified in the HeliosPay v4.2 specification.", "kind": "abstain", "domain": "A_heliospay"},
    {"question": "What error code handles biometric face-scan failures in HeliosPay v4.2?",
     "gold_answer": "No biometric face-scan error code is specified in HeliosPay v4.2.", "kind": "abstain", "domain": "A_heliospay"},
    {"question": "What is the weekend wire rebate percentage in HeliosPay v4.2?",
     "gold_answer": "A weekend wire rebate is not specified in the HeliosPay v4.2 specification.", "kind": "abstain", "domain": "A_heliospay"},
    {"question": "What is the office street address of HeliosPay headquarters in v4.2?",
     "gold_answer": "No office street address is specified in the HeliosPay v4.2 specification.", "kind": "abstain", "domain": "A_heliospay"},
]

ABSTENTION_TEST_A = [
    "In HeliosPay v4.2, what is the base risk score for corridor AFRICA-Z4?",
    "What is the maximum annual credit card cashback rate in HeliosPay v4.2?",
    "Who is the Chief Technology Officer named in the HeliosPay v4.2 policy?",
    "What does error code HP-9999 signify in HeliosPay v4.2?",
    "What is the overnight margin lending interest rate in HeliosPay v4.2?",
    "In HeliosPay v4.2, what penalty applies to mobile app login timeouts?",
    "Which external cloud provider hosts HeliosPay v4.2 cold vaults according to the spec?",
    "What is the retail debit card ATM withdrawal limit in HeliosPay v4.2?",
]


def grade_abstention(pred: str) -> bool:
    vis = pred.split("</think>")[-1].strip() if "</think>" in pred else pred.strip()
    p = vis.lower()
    cues = [
        "not specified", "not defined", "not mentioned", "not listed", "does not specify",
        "not covered", "no information", "not included", "not provided", "not in the",
        "cannot determine", "unknown", "does not contain", "not stated", "not found",
        "no ", "is not a", "aren't specified",
    ]
    return any(c in p for c in cues)


def grade_ood_response(pred: str, keywords: List[str]) -> bool:
    vis = pred.split("</think>")[-1].strip() if "</think>" in pred else pred.strip()
    low = vis.lower()
    return any(k.lower() in low for k in keywords)


def grade_extended_response(response: str, item: Dict[str, Any]) -> bool:
    dom = item["domain"]
    if dom in ("A_heliospay", "B_nvx418", "C_lithox9"):
        return grade_domain_response(response, item)

    vis = response.split("</think>")[-1].strip() if "</think>" in response else response.strip()
    low = vis.lower()
    exp = item["expected"]

    if item["kind"] == "fact":
        kws = exp["keywords"]
        if len(kws) == 2 and (kws[0].replace("-", " ") == kws[1] or kws[0].replace(",", "") == kws[1]):
            return any(k.lower() in low for k in kws)
        return all(k.lower() in low for k in kws)

    clean = re.sub(r"[*`]", "", low)
    if dom == "A_heliospay_v50":
        m_r = re.findall(r"risk\s*score\s*[:=]\s*(-?\d+)", clean)
        m_t = re.findall(r"tier\s*[:=]\s*([a-z0-9\-]+)", clean)
        m_f = re.findall(r"total\s*fee\s*[:=]\s*(\d+)", clean)
        if not (m_r and m_t and m_f):
            return False
        return int(m_r[-1]) == exp["R"] and m_t[-1].upper() == exp["tier"].upper() and int(m_f[-1]) == exp["fee"]

    if dom == "D_gridops":
        m_s = re.findall(r"stress\s*index\s*[:=]\s*(-?\d+)", clean)
        m_m = re.findall(r"mode\s*[:=]\s*([a-z0-9\-]+)", clean)
        m_t = re.findall(r"tariff\s*[:=]\s*(\d+)", clean)
        if not (m_s and m_m and m_t):
            return False
        return int(m_s[-1]) == exp["S"] and m_m[-1].upper() == exp["mode"].upper() and int(m_t[-1]) == exp["tariff"]

    if dom == "E_customs":
        m_c = re.findall(r"channel\s*[:=]\s*([a-z0-9\-]+)", clean)
        m_d = re.findall(r"final\s*duty\s*[:=]\s*(\d+)\s*%", clean)
        if not (m_c and m_d):
            return False
        return m_c[-1].upper() == exp["channel"].upper() and int(m_d[-1]) == exp["duty"]

    return False


def has_v42_stale_leak(response: str, item: Dict[str, Any]) -> bool:
    """Returns True if a v5.0 answer outputs stale v4.2 facts/numbers."""
    vis = response.split("</think>")[-1].strip() if "</think>" in response else response.strip()
    low = vis.lower()
    if item["kind"] == "fact":
        return any(p in low for p in V42_STALE_PATTERNS)
    # For rules, check if prediction matches stale v4.2 answer (when v4.2 != v5.0)
    stale = item.get("stale_v42")
    if stale and stale != item["expected"]:
        stale_item = {"domain": "A_heliospay", "kind": "rule", "expected": stale}
        if grade_domain_response(response, stale_item):
            return True
    return False


if __name__ == "__main__":
    d5 = build_all_5_datasets()
    for name, d in d5.items():
        r_ok = sum(grade_extended_response(x["gold_answer"], x) for x in d["test_rules"])
        f_ok = sum(grade_extended_response(x["gold_answer"], x) for x in d["test_facts"])
        print(f"{name:14s} | train={len(d['train']):3d} | test_rules={r_ok}/{len(d['test_rules'])} | test_facts={f_ok}/{len(d['test_facts'])}")
    v5 = build_v50_dataset()
    v5_ok = sum(grade_extended_response(x["gold_answer"], x) for x in v5["test_rules"])
    v5_f = sum(grade_extended_response(x["gold_answer"], x) for x in v5["test_facts"])
    print(f"A_v50_upgrade  | train={len(v5['train']):3d} | test_rules={v5_ok}/{len(v5['test_rules'])} | test_facts={v5_f}/{len(v5['test_facts'])}")
    ba, ea = build_boundary_and_extrap_A()
    print(f"Boundary A={sum(grade_extended_response(x['gold_answer'], x) for x in ba)}/{len(ba)} | Extrap A={sum(grade_extended_response(x['gold_answer'], x) for x in ea)}/{len(ea)}")
    cab = build_cross_doc_AB_tests()
    cab_ok = sum(grade_extended_response(x["gold_answer"], x["item_A"]) and grade_extended_response(x["gold_answer"], x["item_B"]) for x in cab)
    print(f"Cross-Doc A+B={cab_ok}/{len(cab)} | Abstain train={len(ABSTENTION_TRAIN_A)} test={len(ABSTENTION_TEST_A)}")
