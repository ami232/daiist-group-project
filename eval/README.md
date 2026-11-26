# Tender Desk evaluation

## What is already prepared?

| File | What it contains | What the team does |
|---|---|---|
| [case-plan.csv](case-plan.csv) | 30 candidate excerpts with real TED source links, split into five cases per person | Check each source, agree on the company scenario, and label your five cases |
| [cases.csv](cases.csv) | The input file used for grading and the evaluation script | Copy the 30 completed rows here after the team checks them |
| [reference.csv](reference.csv) | 30 separate Spanish construction excerpts | Confirm these are typical public works inputs |
| [shifted.csv](shifted.csv) | 10 separate Spanish cleaning notices | Use them to test whether the sector-change alert fires |

The CSV cells labeled **input** contain short excerpts prepared from TED metadata, with contact details removed. The source link in **notes** identifies the notice. Where available, a separate award link gives the recorded offer count. The numbered groups are **c01–c05, c06–c10, c11–c15, c16–c20, c21–c25, and c26–c30**. The mix is 14 normal, six hard, six review, and four out of scope.

## How to do your five cases

1. Open the **Call** link in the notes and check the excerpt against the original. Add or correct wording if useful. Strip individual names, emails, phone numbers, and ID numbers. Retain the line `Main CPV: XXXXXXXX` in normal cases for the drift replay.
2. Before running Tender Desk, write the facts you can verify from the **excerpt**. For example, write the estimated budget without VAT, duration, and CPV. If a needed field is absent, use JSON `null` for a numeric field or leave it out if you are not scoring that field. An intentionally missing budget stays missing even when the original notice includes it.
3. Agree on a company scenario. If the excerpt gives too little evidence to confirm fit, use `Unconfirmed`. If your scenario gives the company verified fit or a specific failed requirement, use `Meets our requirements` or `Fails a mandatory requirement`. The expected decision is what should happen on that scenario. Enter `review`, `bid` (shortlist), or `no_bid`. An out-of-scope case should be reviewed. Write the reasoning in **notes**.
4. Open the **Award** link separately and record the count of offers received as `actual_offers`. Use `null` if there is no observable count. Keep the award notice and actual count out of the input text.
5. Have another person do the usual manual screening on the same excerpt and company scenario without the app. Put that person’s decision in **baseline**, using the same three codes.
6. In [Tender Desk](https://construction-tender-desk.onrender.com/), open **Load an assigned test case** and choose the case ID. If you edited its excerpt, paste the final version over the prefilled text. For a case with a checked company scenario, open **Company eligibility (optional)** and select the matching outcome. Then select **Analyse tender** and record the decision, estimated offers, any field it missed, and any error. The default **Eligibility not checked** corresponds to `Unconfirmed` in evaluation JSON. **Company meets all mandatory requirements** corresponds to `Meets our requirements`. **Company fails a mandatory requirement** corresponds to `Fails a mandatory requirement`.

In the **expected** cell, use a JSON object. This generic formatting example uses invented numbers:

```json
{"facts":{"cpv":"45233000","budget_eur":500000,"duration_months":12},"decision":"review","actual_offers":8,"fit":"Unconfirmed"}
```

The app estimates how many offers a tender will receive. A different offer number is a measured prediction error. For a known outcome of 8 and an estimate of 11, the absolute error is 3 offers. Leave the `fit` field out to use `Unconfirmed`. Save as UTF-8 CSV with the existing headers. Record only team-verified expected answers in `cases.csv`.

## Run the evaluation and drift demo

```bash
uv run python main.py reference
uv run python main.py drift-demo --practice
uv run python main.py evaluate
uv run python main.py drift-demo
uv run python check.py
```

The first two commands can run immediately. They use the explicit primary CPV code printed in each excerpt, with no Gemini calls. The other three use the team's completed `cases.csv`. Evaluation writes `eval/results.csv` with extraction accuracy, decision agreement, offer error, review rate, cost and latency. The outputs are controlled fields and decision codes, so the evaluator scores them directly.

The drift test changes the sector while keeping Spain constant. It compares the share of explicit CPV codes outside construction with the 30-notice reference. Expect **quiet** on the construction cases and **ALERT** on the cleaning notices. The live Monitoring tab needs 20 recent requests with recognizable CPV lines to show the sector alert. The 10 shifted notices are enough for the scripted replay, which is separate from the live request window.
