# Account Takeover (ATO) & Impossible Travel Incident Playbook
Document ID: SOP-SEC-ATO-2024
Operational Unit: Fraud Operations & Incident Response

## 1. Impossible Travel Velocity Thresholds
- **Physical Infeasibility:** Defined as two or more authenticated events originating from geographical coordinates separated by more than 800 km within a time interval that would require travel speeds in excess of 900 km/h (standard commercial flight velocity).
- **Primary Indicators:** Sudden change in autonomous system number (ASN), foreign IP geolocation, or VPN / proxy exit node routing during an active banking session.

## 2. Immediate Containment Checklist
1. Terminate all active OAuth, web session cookies, and mobile JWT refresh tokens immediately.
2. Downgrade account permissions to read-only status and place a security freeze on external funds movement.
3. Require out-of-band identity verification (e.g., biometric selfie match against driver's license record) prior to token re-issuance.
4. Notify the cardholder via verified secondary communications channel (secondary phone or email).
