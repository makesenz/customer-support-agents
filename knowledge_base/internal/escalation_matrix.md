---
doc: escalation_matrix
audience: INTERNAL – used by the Log Agent, never quoted to customers
last_updated: 2026-11-01
version: 1.0
status: DRAFT – team names and routing to confirm with client
---

# Escalation Matrix

A ticket is **Escalated** if **any one** of the rules below is true. Record every rule that matched in `escalation_reason`.

## Rules from the assignment brief
| # | Trigger | Example |
|---|---|---|
| E1 | Urgency is **High** | "I need this today", order 7+ working days late |
| E2 | Customer is **very angry** | Shouting, abuse, "worst company", threatening to post online |
| E3 | **No matching document** in the knowledge base | Question not covered by any KB file |
| E4 | **Legal threats** | "lawyer", "consumer court", "police complaint", "FIR" |
| E5 | **Fraud or payment disputes** | Double charge, chargeback, "I paid but no order", paid to a fake site |
| E6 | Resolution Agent confidence is **Low** | Partial or uncertain answer |

## Extra rules for Maa Wellness (health products)
| # | Trigger | Why |
|---|---|---|
| E7 | **Side effect or health reaction** reported (rash, stomach pain, dizziness, chest pain, allergy, etc.) | Customer safety. A human must follow up. |
| E8 | **Personal medical question** (pregnancy, diabetes, BP, heart, other medicines, surgery) | The bot must not give medical advice. |
| E9 | **Refund request for a product the customer opened and used**, or "product didn't work". **Not** for items that arrived damaged, leaking or wrong: those follow the replacement policy and are Resolved. | Needs human judgement. |
| E10 | **Order value ≥ ₹9,900** (Power Combo or above) **and** another rule already applies. Never a reason on its own. | High-value customer gets priority |
| E11 | **Counterfeit or fake product** suspected, or bought or paid on any website other than www.maawellness.in | Brand protection |
| E12 | **Media, regulator or bulk/distributor enquiry** | Business decision |
| E13 | **Same order complained about 2+ times** | Repeat contact means the first answer failed |

## Routing
| Category | Escalate to |
|---|---|
| Billing / Payment / Fraud | Accounts team |
| Health / Side effects / Medical | Wellness counsellor (Hosur) |
| Shipping delays / Damage | Dispatch team |
| Legal / Media / Counterfeit | Management |
| Anything else | Support lead |

## Escalated reply rules
- Still send the customer a polite holding reply. Do not leave it empty.
- Tell them a support specialist will contact them, and give the phone number +91 90425 12304.
- Never promise a refund amount, compensation or a deadline that is not in the KB.
