HR/HRV sex-adjusted rerun package

Updated script:
- 03_HRV_drift_v1_sex_adjusted.py

Key changes from the uploaded 03_HRV_drift_v1.py:
1. Final HR/HRV mixed-effects models are selected only from sex-adjusted candidates.
   Base final model: drift ~ arm * scenario + sex_c
2. AIC/BIC model-selection tables now compare:
   - M0_base_sex: drift ~ arm * scenario + sex_c
   - M1_plus_fat: drift ~ arm * scenario + sex_c + fat_pct_c
3. The script accepts the uploaded sedentary column name is_ok_sedentary and treats 'ok' as sedentary/valid.
4. Paths are portable: by default, the script looks for 02_AllHRV_withSed.csv and 00_Participants_Meta.xlsx in the same folder as the script, and writes outputs to HRV_outputs_sex_adjusted.

Rerun summary:
- HR: final model = drift ~ arm * scenario + sex_c
- lnRMSSD: final model = drift ~ arm * scenario + sex_c
- lnHF: final model = drift ~ arm * scenario + sex_c
- P04 QC: PASS; P04 retained
- Participants retained per outcome: 14 total; FR n=8, CC n=6, with HS1 and HS2 values in both arms.

Package contents:
- Updated Python script
- Final model summaries (.txt)
- AIC/BIC model-selection tables (.csv)
- Participant-level drift tables (.csv)
- Main-text tables (.csv/.txt)
- Exclusion/filtering logs (.csv)
- Paired plots (.png/.pdf)
