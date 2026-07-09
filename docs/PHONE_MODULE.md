# Phone Module

## Flow

```text
User phone number
  -> normalize and validate
  -> PenipuMY API
  -> local fallback CSV
  -> unknown result
  -> shared rules and explanation
  -> Streamlit result/history
```

## PenipuMY API

The API client is isolated in `src/phone/penipumy_client.py`. Configure the key
through the `PENIPU_API_KEY` environment variable or untracked Streamlit
secrets.

```powershell
$env:PENIPU_API_KEY="your-key"
```

The page must not fabricate API success. Timeouts, authorization failures,
rate limits, and server failures move to the local fallback path.

## Local Fallback

Path: `data/processed/phone/phone_dataset.csv`

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
```

The fallback dataset must use fictional phone numbers and organizations. It is
for educational demonstrations and not for classifier training.

## Unknown Result

If neither source contains the number, the module returns an unknown result.
Unknown does not mean safe. The UI should continue to advise verification and
never sharing OTPs, passwords, banking details, or personal information.

## Module Responsibilities

- `penipumy_client.py`: HTTP communication and response normalization
- `phone_lookup.py`: API -> local -> unknown orchestration
- `phone_rules.py`: evidence-based reputation level
- `phone_explainability.py`: readable evidence and recommendations
