TASK (step 1 of 3, gather facts): decide which plant tools you need before a work order can be
planned, and call them. Do not write the work order yet.

- Call check_spare_parts for the failure mode you would address (normally the most likely mode in
  the risk assessment) and the machine type. You may also call it for the second most likely mode.
- Call find_maintenance_window with the urgency that fits the risk band
  (critical -> "now", high -> "shift", medium -> "planned") and a realistic duration in hours.
- Call get_maintenance_policy for the failure mode and the risk band.
You may call several tools at once. Call each tool at most twice.
