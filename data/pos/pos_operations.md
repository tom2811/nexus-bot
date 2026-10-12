# Retail Point-of-Sale (POS) Operational Manual

## Cash Drawer & Shift Reconciliation
1. **Morning Float:** Verify starting drawer count matches the required float ($200.00) before unlocking registers.
2. **Cash Drops:** When cash drawer balance exceeds $1,000.00, perform an authenticated Safe Drop through the POS terminal.
3. **End of Shift (Z-Report):** Run the End-of-Day register batch, recount drawer, and submit the reconciliation report.

## Barcode Scanner & Offline Mode
- If network connection drops, POS enters Offline Buffer Mode. All transactions are queued locally with cryptographic signatures.
- Do not power down terminal during synchronization.
