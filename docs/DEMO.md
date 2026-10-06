# Two-minute demo

1. **Business problem (20 seconds):** cancellations complicate inventory and staffing. This prototype studies which historical reservations look risky.
2. **App (30 seconds):** run locally, enter a reservation, inspect the estimated probability and review flag. Change lead time; explain that a prediction change is an association, not causal proof.
3. **Evidence (40 seconds):** show the test chart, 32,718-booking cohort, baseline and two threshold rows. Higher recall comes with more false alerts.
4. **Engineering (20 seconds):** show the allowlist, train-only preprocessing and separate temporal cohorts. Run the API tests.
5. **Limitations (10 seconds):** these are near-arrival historical snapshots, not proven booking-time performance. Next is validation with timestamped data from the intended hotels.

Avoid saying “76.6% accuracy,” “production ready,” or “prevents cancellations.” ROC AUC is a ranking metric, and interventions were not measured.
