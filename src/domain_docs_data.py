"""
Paper Reference: Section 3.2, Section 4.2, Section 5.2 (3-Domain Unseen Synthetic Private Docs).
Zero-Leakage Synthetic Vertical-Domain Document Continual Learning Benchmark.

Three completely novel enterprise vertical-domain documents (unseen during pretraining):
  - Doc A: HeliosPay v4.2 Cross-Border Settlement & Compliance Policy (FinTech)
  - Doc B: NVX-418 (Celviranib) Phase IIb Clinical Trial SOP (BioPharma)
  - Doc C: Litho-X9 EUV Lithography Yield Diagnostics Manual (Semiconductor Fab)

Each domain provides:
  - Raw document text C_j (for In-Context Teacher & In-Context Upper Bound)
  - Programmatically generated, strictly disjoint train & test splits for:
      1) Procedural Rule Application (unseen numerical inputs & branching logic)
      2) Fine-Grained Exact Factual Recall (paraphrased factual queries)
  - 100% deterministic programmatic graders (no LLM judge variance)
"""

import random
import re
from typing import Any, Dict, List, Tuple


# ===========================================================================
# Raw Enterprise Documents (Completely Synthetic, Zero Pretraining Leakage)
# ===========================================================================

DOC_A_TEXT = """# HeliosPay v4.2 Internal Cross-Border Settlement & Compliance Specification

## 1. Governance, Constants, and Escalation Contacts
- System Version: HeliosPay v4.2 (Production Core)
- Daily Settlement Cutoff: 16:40 SGT
- Mandatory Reserve Buffer: 14.5% of rolling 30-day corridor volume
- Cold Vault Quorum: 4-of-7 hardware security keys
- Regulatory Audit Retention Period: 9.5 years
- Lead Compliance Officer: Dr. Maren Lindholm
- Standard Escalation SLA: 22 minutes
- Maximum Single Instant Wire Limit: 420,000 USD

## 2. Corridor Base Parameters
- Corridor APAC-B7: Base Risk = 12, Base Fee = 38 bps
- Corridor EMEA-K3: Base Risk = 18, Base Fee = 52 bps
- Corridor LATAM-Q9: Base Risk = 31, Base Fee = 74 bps

## 3. Proprietary Error & Halt Codes
- HP-4172: Velocity Circuit Breaker (triggered when Risk Score R >= 75)
- HP-5089: Corridor Liquidity Halt (liquidity pool drops below 14.5% reserve buffer)
- HP-3314: Beneficiary Sanctions Mismatch (OFAC/EU watchlist hit)
- HP-6205: FX Margin Breach (triggered immediately on LATAM-Q9 wires exceeding 420,000 USD)
- HP-2901: Dual-Key Signature Expiry (L2 dual-key token older than 90 minutes)

## 4. Risk Scoring & Approval Tier Decision Rules
For every wire transfer, compute the integer Risk Score R as follows:
  R = Base_Risk(corridor)
      + 15 if amount_USD > 150,000 else 0
      + 25 if beneficiary is NEW (first-time) else 0
      + 20 if submitted AFTER 16:40 SGT cutoff else 0
      - 10 if sender is Tier-1 Verified else 0

Apply the Approval Tier decision tree in strict priority order:
1. Override Exception: If corridor is LATAM-Q9 AND amount_USD > 420,000:
   -> Tier = FREEZE-HP6205, Total Fee = 0 bps (wire rejected; notify Dr. Maren Lindholm).
2. Else if R >= 75:
   -> Tier = FREEZE-HP4172, Total Fee = 0 bps (wire rejected; notify Dr. Maren Lindholm).
3. Else if 55 <= R < 75:
   -> Tier = L2-COMMITTEE, Total Fee = Base_Fee + 25 bps (dual-key signature, 4-hour hold).
4. Else if 30 <= R < 55:
   -> Tier = L1-ANALYST, Total Fee = Base_Fee + 12 bps (22-minute review SLA).
5. Else (R < 30):
   -> Tier = AUTO-CLEAR, Total Fee = Base_Fee bps (instant settlement).
"""

DOC_B_TEXT = """# NVX-418 (Celviranib) Phase IIb Clinical Trial Standard Operating Procedure

## 1. Study Metadata & Pharmacological Constants
- Investigational Compound: NVX-418 (generic name: Celviranib)
- Protocol Identifier: VR-2026-418B
- Sponsor Organization: Virelia Therapeutics
- Principal Investigator: Prof. Kenji Takahashi
- Central Biomarker Laboratory: NordicBio-Oslo
- Plasma Elimination Half-Life: 18.4 hours
- Cold-Chain Storage Range: 2.5 C to 6.0 C (maximum temperature excursion: 45 minutes)
- Primary Toxicity Biomarker: sTREM-42 (pg/mL)
- Renal Stopping Threshold: eGFR < 38 mL/min
- Emergency Rescue Antidote: Thaloxene-7 at 3.2 mg/kg IV

## 2. Dosing Calculation & Adverse Event Protocol
Step 1 — Check Immediate Contraindications / Severe Toxicity FIRST:
- If eGFR < 38 mL/min:
  -> Action = CONTRAINDICATED-STOP, Final Dose = 0 mg. Do not administer NVX-418.
- Else if sTREM-42 > 850 pg/mL OR ALT > 5.0x ULN:
  -> Action = GRADE-3-HOLD, Final Dose = 0 mg. Administer Thaloxene-7 (3.2 mg/kg IV) and notify Prof. Kenji Takahashi within 6 hours.

Step 2 — Otherwise, compute the Adjusted Daily Dose sequentially:
1. Base Dose: D = min(2.5 * weight_kg, 240) mg.
2. Renal Adjustment: If 38 <= eGFR < 55 mL/min, reduce D by 20% (D = D * 0.8).
3. Geriatric Adjustment: If age >= 70 years, subtract 15 mg (D = D - 15).
4. Drug Interaction Adjustment: If co-administered with a CYP3A4 inhibitor (such as Ketoconazole or Clarithromycin), reduce D by 40% (D = D * 0.6).
5. Biomarker Action & Adjustment:
   - If 500 <= sTREM-42 <= 850 pg/mL: Action = GRADE-2-REDUCE, reduce D by 50% (D = D * 0.5).
   - Else (sTREM-42 < 500 and ALT <= 5.0x ULN): Action = CONTINUE-STANDARD, no further reduction.
6. Final Rounding: Round the resulting dose D to the nearest 5 mg increment (minimum 0 mg).
"""

DOC_C_TEXT = """# Litho-X9 Extreme Ultraviolet (EUV) Lithography Yield Diagnostics Manual

## 1. Equipment Specifications & Escalation Contacts
- Tool Model: Litho-X9
- Control Firmware: EUV-OS 7.3.4
- Operating Wavelength: 13.52 nm
- Pellicle Maximum Safe Temperature: 415 K
- Chamber Vacuum Floor Threshold: 1.8e-6 Pa
- Chief Fab Engineer: Elena Kowalski
- Preventive Maintenance Interval: 140,000 wafers
- Laser Source Power Ceiling: 365 W
- Nominal Wafer Throughput: 185 wph (capped at 115 wph under mirror haze alarm)

## 2. Diagnostic Fault Codes
- LX-904: Stage Tilt Calibration Drift (triggered when effective overlay error E_eff >= 12.0 nm)
- LX-812: Collector Mirror Haze Alarm (triggered when laser power > 340 W AND mirror haze > 4.5%)
- LX-771: Tin Droplet Generator Jitter (droplet timing variance exceeds 8 nanoseconds)
- LX-639: Reticle Thermal Bow Abort (triggered immediately when pellicle temp T_pellicle > 415 K)

## 3. Effective Overlay Error & Lot Disposition Rules
Compute Effective Overlay Error E_eff (in nm), rounded to 2 decimal places:
  E_eff = E_raw + 0.4 * (T_pellicle - 380) + (1.5 if chamber vacuum > 1.8e-6 Pa else 0.0)

Apply the Lot Disposition decision tree in strict priority order:
1. If T_pellicle > 415 K:
   -> Status = SHUTDOWN-LX639, Offset = 0.0 nm (immediate thermal abort; escalate to Elena Kowalski).
2. Else if E_eff >= 12.0 nm:
   -> Status = RECALIBRATE-LX904, Offset = 0.0 nm (halt lot; run 18-minute stage tilt sweep).
3. Else if 6.0 <= E_eff < 12.0 nm:
   -> Status = OFFSET-COMPENSATE, Offset = -0.75 * E_eff nm (rounded to 2 decimals; apply feed-forward and continue lot).
4. Else (E_eff < 6.0 nm):
   -> Status = PASS-NOMINAL, Offset = 0.0 nm (continue lot with no correction).
"""

DOCS = {
    "A_heliospay": DOC_A_TEXT,
    "B_nvx418": DOC_B_TEXT,
    "C_lithox9": DOC_C_TEXT,
}


# ===========================================================================
# Exact Fact Questions (Train paraphrases vs Held-out Test paraphrases)
# ===========================================================================

FACTS_A = [
    ("In HeliosPay v4.2, what is the daily settlement cutoff time?",
     "What time is the daily settlement cutoff for HeliosPay v4.2?",
     "16:40 SGT", ["16:40"]),
    ("In HeliosPay v4.2, what is the mandatory reserve buffer percentage?",
     "What percentage of rolling 30-day corridor volume must be held as the HeliosPay v4.2 reserve buffer?",
     "14.5%", ["14.5%"]),
    ("Who is the Lead Compliance Officer for HeliosPay v4.2?",
     "To which compliance officer must frozen wires in HeliosPay v4.2 be escalated?",
     "Dr. Maren Lindholm", ["maren lindholm"]),
    ("What is the Cold Vault Quorum required by HeliosPay v4.2?",
     "How many hardware security keys form the Cold Vault Quorum in HeliosPay v4.2?",
     "4-of-7", ["4-of-7", "4 of 7"]),
    ("What is the regulatory audit retention period specified in HeliosPay v4.2?",
     "For how many years must HeliosPay v4.2 audit records be retained?",
     "9.5 years", ["9.5"]),
    ("What is the standard escalation SLA in HeliosPay v4.2?",
     "How many minutes is the review/escalation SLA for L1-ANALYST in HeliosPay v4.2?",
     "22 minutes", ["22"]),
    ("What is the maximum single instant wire limit in HeliosPay v4.2?",
     "Above what USD amount does a single instant wire exceed the HeliosPay v4.2 limit?",
     "420,000 USD", ["420,000", "420000"]),
    ("In HeliosPay v4.2, what error code corresponds to Corridor Liquidity Halt?",
     "Which HeliosPay v4.2 halt code triggers when liquidity falls below the reserve buffer?",
     "HP-5089", ["hp-5089"]),
    ("In HeliosPay v4.2, what does error code HP-3314 indicate?",
     "What compliance failure is signaled by HeliosPay v4.2 code HP-3314?",
     "Beneficiary Sanctions Mismatch", ["sanctions mismatch"]),
    ("In HeliosPay v4.2, what does error code HP-2901 represent?",
     "Which condition causes error code HP-2901 in HeliosPay v4.2?",
     "Dual-Key Signature Expiry", ["dual-key signature expiry", "signature expiry"]),
    ("In HeliosPay v4.2, what are the base risk and base fee for corridor APAC-B7?",
     "State the base risk score and base fee in bps for HeliosPay corridor APAC-B7.",
     "Base Risk 12, Base Fee 38 bps", ["12", "38"]),
    ("In HeliosPay v4.2, what are the base risk and base fee for corridor LATAM-Q9?",
     "State the base risk score and base fee in bps for HeliosPay corridor LATAM-Q9.",
     "Base Risk 31, Base Fee 74 bps", ["31", "74"]),
]

FACTS_B = [
    ("What is the generic name of investigational compound NVX-418?",
     "Under what generic name is compound NVX-418 known in the Phase IIb SOP?",
     "Celviranib", ["celviranib"]),
    ("What is the protocol identifier for the NVX-418 Phase IIb clinical trial?",
     "Give the exact protocol ID for the NVX-418 Phase IIb study.",
     "VR-2026-418B", ["vr-2026-418b"]),
    ("Which sponsor organization runs the NVX-418 trial?",
     "Name the pharmaceutical sponsor of the NVX-418 clinical trial.",
     "Virelia Therapeutics", ["virelia"]),
    ("Who is the Principal Investigator for the NVX-418 Phase IIb trial?",
     "Which investigator must be notified within 6 hours for a Grade-3 Hold in the NVX-418 SOP?",
     "Prof. Kenji Takahashi", ["kenji takahashi"]),
    ("Which central laboratory processes biomarkers for NVX-418?",
     "Name the central biomarker lab specified in the NVX-418 Phase IIb SOP.",
     "NordicBio-Oslo", ["nordicbio-oslo", "nordicbio"]),
    ("What is the plasma elimination half-life of NVX-418?",
     "How many hours is the elimination half-life of NVX-418 (Celviranib)?",
     "18.4 hours", ["18.4"]),
    ("What is the cold-chain storage temperature range and max excursion for NVX-418?",
     "Specify the required storage range and maximum allowed temperature excursion time for NVX-418.",
     "2.5 C to 6.0 C, max excursion 45 minutes", ["2.5", "6.0", "45"]),
    ("What is the primary toxicity biomarker monitored in the NVX-418 trial?",
     "Which biomarker in pg/mL triggers Grade-2 or Grade-3 dosing changes for NVX-418?",
     "sTREM-42", ["strem-42"]),
    ("What is the renal stopping eGFR threshold for NVX-418?",
     "Below what eGFR value (in mL/min) is NVX-418 strictly contraindicated?",
     "38 mL/min", ["38"]),
    ("What is the emergency rescue antidote and IV dose for NVX-418 severe toxicity?",
     "Name the rescue antidote and its mg/kg IV dosing specified in the NVX-418 SOP.",
     "Thaloxene-7 at 3.2 mg/kg IV", ["thaloxene-7", "3.2"]),
]

FACTS_C = [
    ("What control firmware version runs on the Litho-X9 EUV tool?",
     "Specify the exact firmware release for Litho-X9 in the yield diagnostics manual.",
     "EUV-OS 7.3.4", ["euv-os 7.3.4", "7.3.4"]),
    ("What is the operating EUV wavelength of the Litho-X9 machine?",
     "Give the operating wavelength in nm for the Litho-X9 lithography tool.",
     "13.52 nm", ["13.52"]),
    ("What is the pellicle maximum safe temperature on Litho-X9?",
     "Above what temperature in Kelvin does Litho-X9 trigger an immediate thermal abort?",
     "415 K", ["415"]),
    ("What is the chamber vacuum floor threshold for Litho-X9?",
     "Above what chamber pressure in Pa is a +1.5 nm vacuum penalty added on Litho-X9?",
     "1.8e-6 Pa", ["1.8e-6", "1.8×10^-6", "1.8 x 10^-6"]),
    ("Who is the Chief Fab Engineer for Litho-X9 thermal abort escalations?",
     "Which engineer must be notified on a Litho-X9 SHUTDOWN-LX639 event?",
     "Elena Kowalski", ["elena kowalski"]),
    ("What is the preventive maintenance interval for Litho-X9?",
     "Every how many wafers does Litho-X9 require preventive maintenance?",
     "140,000 wafers", ["140,000", "140000"]),
    ("What is the laser source power ceiling on Litho-X9?",
     "Specify the maximum laser source power ceiling in watts for Litho-X9.",
     "365 W", ["365"]),
    ("What are the nominal and mirror-haze-capped wafer throughputs of Litho-X9?",
     "Give both the normal wph throughput and the capped wph throughput under LX-812 haze alarm on Litho-X9.",
     "185 wph nominal, 115 wph capped", ["185", "115"]),
    ("In the Litho-X9 manual, what fault does code LX-771 signify?",
     "Which Litho-X9 failure mode corresponds to diagnostic code LX-771?",
     "Tin Droplet Generator Jitter", ["tin droplet", "jitter"]),
    ("In the Litho-X9 manual, what triggers fault code LX-812?",
     "State the fault name and exact trigger condition for Litho-X9 code LX-812.",
     "Collector Mirror Haze Alarm (laser power > 340 W and mirror haze > 4.5%)", ["haze", "340", "4.5"]),
]

ALL_FACTS = {
    "A_heliospay": FACTS_A,
    "B_nvx418": FACTS_B,
    "C_lithox9": FACTS_C,
}


# ===========================================================================
# Procedural Rule Scenario Generators (Strictly Disjoint Train & Test)
# ===========================================================================

def _solve_doc_a(corridor: str, amount: int, new_ben: bool, after_cut: bool, tier1: bool) -> Dict[str, Any]:
    base_risk = {"APAC-B7": 12, "EMEA-K3": 18, "LATAM-Q9": 31}[corridor]
    base_fee = {"APAC-B7": 38, "EMEA-K3": 52, "LATAM-Q9": 74}[corridor]
    r = base_risk
    r += 15 if amount > 150000 else 0
    r += 25 if new_ben else 0
    r += 20 if after_cut else 0
    r -= 10 if tier1 else 0

    if corridor == "LATAM-Q9" and amount > 420000:
        tier, fee = "FREEZE-HP6205", 0
    elif r >= 75:
        tier, fee = "FREEZE-HP4172", 0
    elif r >= 55:
        tier, fee = "L2-COMMITTEE", base_fee + 25
    elif r >= 30:
        tier, fee = "L1-ANALYST", base_fee + 12
    else:
        tier, fee = "AUTO-CLEAR", base_fee

    q = (
        f"[HeliosPay v4.2 Wire Assessment] Evaluate the following cross-border wire:\n"
        f"- Corridor: {corridor}\n"
        f"- Amount: {amount:,} USD\n"
        f"- Beneficiary status: {'NEW first-time beneficiary' if new_ben else 'Existing beneficiary'}\n"
        f"- Submission time: {'17:15 SGT (after cutoff)' if after_cut else '14:20 SGT (before cutoff)'}\n"
        f"- Sender status: {'Tier-1 Verified' if tier1 else 'Standard (non-Tier-1)'}\n"
        f"Compute the Risk Score R, Approval Tier, and Total Fee (in bps). "
        f"End your response with exactly:\n"
        f"Risk Score: <R>\nTier: <TIER>\nTotal Fee: <FEE> bps"
    )
    a = (
        f"Under HeliosPay v4.2, corridor {corridor} has Base Risk {base_risk} and Base Fee {base_fee} bps.\n"
        f"Adjustments: amount>150k ({'+15' if amount > 150000 else '+0'}), "
        f"new beneficiary ({'+25' if new_ben else '+0'}), "
        f"after 16:40 SGT ({'+20' if after_cut else '+0'}), "
        f"Tier-1 Verified ({'-10' if tier1 else '+0'}).\n"
        f"Therefore R = {r}. Applying the priority decision tree yields Tier = {tier} and Total Fee = {fee} bps.\n"
        f"Risk Score: {r}\nTier: {tier}\nTotal Fee: {fee} bps"
    )
    return {"question": q, "gold_answer": a, "expected": {"R": r, "tier": tier, "fee": fee}}


def _solve_doc_b(weight: int, egfr: int, age: int, cyp: bool, strem: int, alt: float) -> Dict[str, Any]:
    d_base = min(int(2.5 * weight), 240)
    if egfr < 38:
        action, final_dose = "CONTRAINDICATED-STOP", 0
        steps = f"eGFR={egfr} < 38 mL/min triggers immediate renal stopping rule."
    elif strem > 850 or alt > 5.0:
        action, final_dose = "GRADE-3-HOLD", 0
        steps = f"sTREM-42={strem} / ALT={alt}x ULN triggers GRADE-3-HOLD with Thaloxene-7 rescue."
    else:
        d = float(d_base)
        notes = [f"Base dose = min(2.5*{weight}, 240) = {d_base} mg"]
        if egfr < 55:
            d *= 0.8
            notes.append(f"38<=eGFR<55 (-20%) -> {d:.1f} mg")
        if age >= 70:
            d -= 15.0
            notes.append(f"age>=70 (-15 mg) -> {d:.1f} mg")
        if cyp:
            d *= 0.6
            notes.append(f"CYP3A4 inhibitor (-40%) -> {d:.1f} mg")
        if strem >= 500:
            action = "GRADE-2-REDUCE"
            d *= 0.5
            notes.append(f"500<=sTREM-42<=850 (-50%) -> {d:.1f} mg")
        else:
            action = "CONTINUE-STANDARD"
        final_dose = max(0, int(round(d / 5.0) * 5))
        steps = "; ".join(notes) + f"; rounded to nearest 5 mg = {final_dose} mg."

    q = (
        f"[NVX-418 Clinical Dosing] Determine the clinical action and daily dose for this NVX-418 trial patient:\n"
        f"- Weight: {weight} kg\n"
        f"- eGFR: {egfr} mL/min\n"
        f"- Age: {age} years\n"
        f"- Co-administered CYP3A4 inhibitor: {'Yes (Ketoconazole)' if cyp else 'No'}\n"
        f"- sTREM-42: {strem} pg/mL\n"
        f"- ALT: {alt}x ULN\n"
        f"End your response with exactly:\n"
        f"Action: <ACTION>\nFinal Dose: <DOSE> mg"
    )
    a = (
        f"Following the NVX-418 Phase IIb SOP: {steps}\n"
        f"Action: {action}\nFinal Dose: {final_dose} mg"
    )
    return {"question": q, "gold_answer": a, "expected": {"action": action, "dose": final_dose}}


def _solve_doc_c(e_raw: float, t_pell: int, vac_bad: bool) -> Dict[str, Any]:
    e_eff = round(e_raw + 0.4 * (t_pell - 380) + (1.5 if vac_bad else 0.0), 2)
    if t_pell > 415:
        status, offset = "SHUTDOWN-LX639", 0.0
    elif e_eff >= 12.0:
        status, offset = "RECALIBRATE-LX904", 0.0
    elif e_eff >= 6.0:
        status, offset = "OFFSET-COMPENSATE", round(-0.75 * e_eff, 2)
    else:
        status, offset = "PASS-NOMINAL", 0.0

    q = (
        f"[Litho-X9 Lot Diagnostics] Evaluate this Litho-X9 EUV lithography telemetry snapshot:\n"
        f"- Raw Overlay Error E_raw: {e_raw} nm\n"
        f"- Pellicle Temperature T_pellicle: {t_pell} K\n"
        f"- Chamber Vacuum: {'2.4e-6 Pa (above 1.8e-6 Pa floor)' if vac_bad else '1.2e-6 Pa (within floor)'}\n"
        f"Compute Effective Overlay Error E_eff (nm), Lot Status, and Feed-Forward Offset (nm). "
        f"End your response with exactly:\n"
        f"E_eff: <E_EFF> nm\nStatus: <STATUS>\nOffset: <OFFSET> nm"
    )
    a = (
        f"Under the Litho-X9 manual: E_eff = {e_raw} + 0.4*({t_pell}-380) + {'1.5' if vac_bad else '0.0'} = {e_eff} nm.\n"
        f"Applying the disposition tree gives Status = {status} and Offset = {offset} nm.\n"
        f"E_eff: {e_eff} nm\nStatus: {status}\nOffset: {offset} nm"
    )
    return {"question": q, "gold_answer": a, "expected": {"e_eff": e_eff, "status": status, "offset": offset}}


def build_domain_datasets(seed: int = 42) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Build strictly disjoint train / test_rules / test_facts for all 3 vertical domains.
    """
    rng = random.Random(seed)
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # --- Domain A ---
    all_a = []
    for corr in ["APAC-B7", "EMEA-K3", "LATAM-Q9"]:
        for amt in [50000, 120000, 180000, 310000, 480000]:
            for nb in [False, True]:
                for ac in [False, True]:
                    for t1 in [False, True]:
                        item = _solve_doc_a(corr, amt, nb, ac, t1)
                        item["kind"] = "rule"
                        item["domain"] = "A_heliospay"
                        all_a.append(item)
    rng.shuffle(all_a)
    test_rules_a = all_a[:24]
    train_rules_a = all_a[24:104]  # 80 disjoint scenarios

    train_facts_a = []
    test_facts_a = []
    for q_tr, q_te, ans, kws in FACTS_A:
        train_facts_a.append({
            "question": q_tr,
            "gold_answer": f"According to HeliosPay v4.2, the answer is: {ans}.",
            "expected": {"keywords": kws},
            "kind": "fact",
            "domain": "A_heliospay",
        })
        test_facts_a.append({
            "question": q_te,
            "gold_answer": ans,
            "expected": {"keywords": kws},
            "kind": "fact",
            "domain": "A_heliospay",
        })
    # Upsample facts in training so both rules and facts get strong supervision
    out["A_heliospay"] = {
        "train": train_rules_a + train_facts_a * 4,
        "test_rules": test_rules_a,
        "test_facts": test_facts_a,
    }

    # --- Domain B ---
    all_b = []
    for w in [52, 60, 68, 76, 84, 92, 100]:
        for egfr in [30, 45, 72]:
            for age in [45, 74]:
                for cyp in [False, True]:
                    for strem, alt in [(320, 1.5), (640, 2.0), (920, 2.2)]:
                        # Check no exact .5 rounding ambiguity on 5mg grid
                        item = _solve_doc_b(w, egfr, age, cyp, strem, alt)
                        item["kind"] = "rule"
                        item["domain"] = "B_nvx418"
                        all_b.append(item)
    rng.shuffle(all_b)
    test_rules_b = all_b[:24]
    train_rules_b = all_b[24:104]

    train_facts_b, test_facts_b = [], []
    for q_tr, q_te, ans, kws in FACTS_B:
        train_facts_b.append({
            "question": q_tr,
            "gold_answer": f"According to the NVX-418 Phase IIb SOP, the answer is: {ans}.",
            "expected": {"keywords": kws},
            "kind": "fact",
            "domain": "B_nvx418",
        })
        test_facts_b.append({
            "question": q_te,
            "gold_answer": ans,
            "expected": {"keywords": kws},
            "kind": "fact",
            "domain": "B_nvx418",
        })
    out["B_nvx418"] = {
        "train": train_rules_b + train_facts_b * 4,
        "test_rules": test_rules_b,
        "test_facts": test_facts_b,
    }

    # --- Domain C ---
    all_c = []
    for e_raw in [1.0, 2.5, 4.0, 5.5, 7.0, 9.0, 11.0]:
        for t_pell in [375, 385, 390, 395, 400, 405, 410, 420]:
            for vac in [False, True]:
                item = _solve_doc_c(e_raw, t_pell, vac)
                item["kind"] = "rule"
                item["domain"] = "C_lithox9"
                all_c.append(item)
    rng.shuffle(all_c)
    test_rules_c = all_c[:24]
    train_rules_c = all_c[24:104]

    train_facts_c, test_facts_c = [], []
    for q_tr, q_te, ans, kws in FACTS_C:
        train_facts_c.append({
            "question": q_tr,
            "gold_answer": f"According to the Litho-X9 Yield Diagnostics Manual, the answer is: {ans}.",
            "expected": {"keywords": kws},
            "kind": "fact",
            "domain": "C_lithox9",
        })
        test_facts_c.append({
            "question": q_te,
            "gold_answer": ans,
            "expected": {"keywords": kws},
            "kind": "fact",
            "domain": "C_lithox9",
        })
    out["C_lithox9"] = {
        "train": train_rules_c + train_facts_c * 4,
        "test_rules": test_rules_c,
        "test_facts": test_facts_c,
    }

    return out


# ===========================================================================
# Deterministic Graders
# ===========================================================================

def grade_domain_response(response: str, item: Dict[str, Any]) -> bool:
    """Strictly grade rule application or exact factual recall."""
    if "</think>" in response:
        vis = response.split("</think>")[-1].strip()
    else:
        vis = response.strip()
    low = vis.lower()
    exp = item["expected"]

    if item["kind"] == "fact":
        kws = exp["keywords"]
        # For facts with multiple required elements (e.g. ["12", "38"] or ["2.5", "6.0", "45"]),
        # distinguish alternative spellings vs required multi-parts:
        if len(kws) == 2 and (kws[0].replace("-", " ") == kws[1] or kws[0].replace(",", "") == kws[1]):
            return any(k.lower() in low for k in kws)
        if kws == ["1.8e-6", "1.8×10^-6", "1.8 x 10^-6"]:
            return any(k.lower() in low for k in kws) or ("1.8" in low and "-6" in low)
        if kws == ["euv-os 7.3.4", "7.3.4"]:
            return "7.3.4" in low
        if kws == ["nordicbio-oslo", "nordicbio"]:
            return "nordicbio" in low
        if kws == ["dual-key signature expiry", "signature expiry"]:
            return "signature expiry" in low or "dual-key" in low
        return all(k.lower() in low for k in kws)

    # Procedural rule grading (strip markdown bold/code and take the LAST match)
    clean = re.sub(r"[*`]", "", low)
    dom = item["domain"]
    if dom == "A_heliospay":
        m_r = re.findall(r"risk\s*score\s*[:=]\s*(-?\d+)", clean)
        m_t = re.findall(r"tier\s*[:=]\s*([a-z0-9\-]+)", clean)
        m_f = re.findall(r"total\s*fee\s*[:=]\s*(\d+)", clean)
        if not (m_r and m_t and m_f):
            return False
        return (
            int(m_r[-1]) == exp["R"]
            and m_t[-1].upper() == exp["tier"].upper()
            and int(m_f[-1]) == exp["fee"]
        )

    if dom == "B_nvx418":
        m_a = re.findall(r"action\s*[:=]\s*([a-z0-9\-]+)", clean)
        m_d = re.findall(r"final\s*dose\s*[:=]\s*(\d+(?:\.\d+)?)", clean)
        if not (m_a and m_d):
            return False
        return (
            m_a[-1].upper() == exp["action"].upper()
            and abs(float(m_d[-1]) - exp["dose"]) < 0.5
        )

    if dom == "C_lithox9":
        m_e = re.findall(r"e_eff\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*nm", clean)
        if not m_e:
            m_e = re.findall(r"e_eff\s*[:=]\s*(-?\d+(?:\.\d+)?)", clean)
        m_s = re.findall(r"status\s*[:=]\s*([a-z0-9\-]+)", clean)
        m_o = re.findall(r"offset\s*[:=]\s*(-?\d+(?:\.\d+)?)", clean)
        if not (m_e and m_s and m_o):
            return False
        return (
            abs(float(m_e[-1]) - exp["e_eff"]) <= 0.05
            and m_s[-1].upper() == exp["status"].upper()
            and abs(float(m_o[-1]) - exp["offset"]) <= 0.05
        )

    return False
