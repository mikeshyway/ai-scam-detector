# Phone Module

## Flow

```text
User phone number
  -> normalize to canonical international format
  -> configured live provider checks
  -> Veriphone.io metadata evidence
  -> PenipuMY reputation/report evidence where configured
  -> unknown result when evidence is unavailable
  -> shared normalized record
  -> transparent rules and explanation
  -> Streamlit result/history
```

The final prototype evidence notebook treats phone checks as provider evidence,
not local ML training. Veriphone.io supplies carrier/line metadata. PenipuMY supplies
reputation or report-oriented fields where a valid key is configured. Older
IPQualityScore client files may remain in `src/phone/` for compatibility or
history, but they are not the primary documented final flow.

## Phone Number Normalization

The UI accepts common local and international formats, then converts them into
one canonical E.164-style internal format before lookup.

Accepted examples:

```text
012-345 6789
0123456789
+60 12-345 6789
60123456789
(03) 1234 5678
```

Canonical internal examples:

```text
+60123456789
+60312345678
```

Veriphone.io receives the canonical E.164-style value, such as `+60123456789`. The app
rejects clearly invalid text, repeated plus signs, and alphabetic input.

## Live Provider

### Veriphone.io Carrier Lookup

Veriphone.io is used as the active carrier and number metadata provider. It can
return validity, carrier, line type, E.164 phone number, national/local
formatting, country, country code, calling country code, region, timezone, and
current-carrier metadata when the selected lookup mode supports it.

Documentation: <https://veriphone.io/docs/v3>

API keys: <https://veriphone.io/app>

Free-tier note: Veriphone.io currently gives free accounts 1,000 credits per
month with no credit card required. Standard validation uses 1 credit. Current
carrier lookup uses 10 credits, and the API returns HTTP 402 when credits are
exhausted. See <https://veriphone.io/pricing> and
<https://veriphone.io/docs/v3>.

Carrier metadata is not scam reputation. A valid phone number does not prove a
caller is safe, and a VoIP/mobile/landline classification does not prove fraud
by itself.

### PenipuMY

PenipuMY is used as a reputation/report evidence provider when configured. Its
fields can support the phone concern explanation, but provider reports are still
evidence, not a trained local caller-risk model and not legal proof.

Free-tier note: PenipuMY currently lists a Free API tier at 100 requests per
day, authenticated with the `X-API-Key` header, with the daily limit resetting
at midnight Malaysia time. See <https://penipu.my/api/v1/docs>.

## API Key

For normal dashboard use, configure provider keys directly inside the Phone
Number tab. Each provider card lets the user enable the provider, paste a
session-only key, test the connection, and view diagnostics. This is the
simplest path for capstone demonstrations.

Environment variables and Streamlit secrets are optional. Use them when keys
should be available automatically after a local restart or in a hosted
deployment.

```powershell
$env:VERIPHONE_API_KEY="your-veriphone-key"
$env:PENIPUMY_API_KEY="your-penipumy-key"
```

Streamlit secrets are also supported:

```toml
VERIPHONE_API_KEY = "..."
PENIPUMY_API_KEY = "..."

[veriphone]
api_key = "..."

[penipumy]
api_key = "..."
```

Never commit `.env`, `.streamlit/secrets.toml`, or real API keys. The repository
includes `.env.example` with blank provider placeholders only.

## Archived Omkar Provider

The previous Omkar Carrier Lookup integration is archived at:

```text
archive/deprecated/phone_providers/omkar/
```

Omkar is no longer active in the dashboard because its current free/demo
response requires a paid Carrier Lookup plan. Restore it only if Omkar access
becomes available again.

## Legacy Local And Demo Evidence

Path: `data/processed/phone/phone_dataset.csv`

This file is retained for older helper workflows and traceable historical
records. The active dashboard phone investigation uses live Veriphone.io and
PenipuMY provider evidence, then returns Unknown when no usable provider
evidence is available. Do not place synthetic demo records in this file.

Required columns:

```text
phone
police_report_count
verified_report_count
spam
fraud
business_tier
business_name
spoofing_report_count
source
record_type
is_demo
source_reference
last_verified
```

Rows in this file should use `record_type=real` and `is_demo=false`.

Path: `data/demo/phone_demo_dataset.csv`

This file contains fictional capstone examples only. The Phone Number tab will
not search it unless Demo Mode is explicitly enabled. Demo results are labelled
as demonstration data and excluded from dashboard headline KPIs.

Path: `data/demo/demo_phone_numbers.csv`

This smaller demo file may be used by notebook/demo workflows as repeatable
presentation input. It should not be described as trained phone-risk data.

Active dashboard order:

```text
Configured live providers
  -> unknown result
```

## Output Principles

- `Valid` means number format/routing appears valid.
- `Metadata available` means carrier or line information was returned.
- `Unknown` means no reputation conclusion is available.
- `High Risk` appears only when real reputation evidence or explicit fallback
  records support it.

The UI shows provenance for each result:

```text
Live provider: Veriphone.io / PenipuMY where configured
Provider returned: Carrier or validation metadata / No usable carrier metadata
Scam reputation available: Yes/No
```

## Charts

The Phone Number tab may show:

- Lookup Evidence Coverage
- Caller Claim Consistency
- Provider Response Completeness
- Session Lookup History after multiple phone lookups

These charts summarize available evidence. They are not ML probabilities and do
not change the final lookup result.

## No Additional Phone ML Model

The phone module intentionally remains:

```text
Veriphone.io API
+ PenipuMY API where configured
+ normalization
+ transparent consistency rules
+ explainability
```

Do not add a phone-specific machine-learning model unless a sufficiently large,
labelled, traceable phone-metadata dataset becomes available. Adding a model
without that dataset would create weak or misleading evidence.

## Unknown Result

If neither Veriphone.io nor PenipuMY returns usable evidence, the module returns
an Unknown result. Unknown does not mean safe. The UI should continue to advise
verification and never sharing OTPs, passwords, banking details, or personal
information.

## History Database

Phone rows saved in `data/session_history.db` are report evidence rows. They can
contain masked phone numbers, provider status, provider-derived fields, concern
labels, concern scores, and recommended verification actions.

This database is not a locally trained phone-risk database. Stored phone
outcomes come from provider-pulled evidence plus deterministic rules at lookup
time. A future lookup can differ if provider data, API keys, quota, rate limits,
or network availability change.

## Module Responsibilities

- `veriphone_client.py`: Veriphone.io HTTP communication and response parsing
- `penipumy_client.py`: PenipuMY HTTP communication and response parsing
- `phone_lookup.py`: provider -> local -> demo/unknown orchestration
- `phone_rules.py`: transparent evidence-based reputation/context level
- `phone_explainability.py`: readable evidence and recommendations
- `ipqs_client.py`: deprecated provider client kept out of the active UI
