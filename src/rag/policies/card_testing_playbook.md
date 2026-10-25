# Card Testing, Micro-Transaction Velocity & Brute-Force Playbook
Document ID: SOP-SEC-BIN-2024
Operational Unit: Merchant & Card Network Security

## 1. Card Testing Attack Patterns
- **Velocity Threshold:** 3 or more authorization attempts within 10 minutes for monetary amounts under $5.00, followed by an immediate authorization attempt exceeding $500.00.
- **PIN / CVV Brute-Force:** More than 2 consecutive incorrect CVV2 or PIN entries within a 1-hour window triggers an automated security lock on the payment credential.

## 2. Mitigation Protocol
1. Temporary lock of card PAN (Primary Account Number) across all merchant acquirers.
2. Require 3D Secure 2.0 Strong Customer Authentication (SCA) with step-up one-time password (OTP) verification.
3. Add originating merchant category code (MCC) and IP subnet to active rate-limiting filter.
