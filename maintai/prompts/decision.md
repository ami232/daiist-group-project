TASK (step 2 of 3, decide): using the machine data, the risk assessment and the tool results,
decide the work order. Return only the JSON object required by the schema.

Decision rules of the plant:
- priority by risk band: critical -> P1; high -> P2 or P1; medium -> P3 or P2.
- action must fit the failure mode you address:
    TWF -> replace_tool or stop_and_inspect
    OSF -> reduce_load, replace_tool or stop_and_inspect
    HDF -> cooling_check or stop_and_inspect
    PWF -> power_check, reduce_load or stop_and_inspect
    any mode -> schedule_inspection is always acceptable
  Do not choose "monitor": this case is not low risk.
- failure_mode_addressed: one of the two most likely modes in the risk assessment.
- part_ids: zero, one or two part_id values copied exactly from check_spare_parts results. Prefer
  parts that are in stock. If the only fitting part has in_stock = 0, you may still list it, but set
  requires_human_review to true and say why in review_reason.
- window_id: one window_id copied exactly from find_maintenance_window results, or null if none
  was returned.
- requires_human_review: true when the risk band is critical, when a plant policy requires a
  person, when the risk model flags readings outside its training range, or when you are unsure.
  review_reason: one short sentence, or "" when no review is requested.
