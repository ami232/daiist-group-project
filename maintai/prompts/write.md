TASK (step 3 of 3, write): the decision in <decision> is FINAL. Do not change or question the
priority, action, failure mode, parts, window or review request; only write the text of the work
order around it. Return only the JSON object required by the schema:

- steps: between 1 and 8 short, concrete steps a technician can follow, in order.
- justification: why this order, in at most 4 sentences.

You may mention part ids, part names, window ids and start times only if they are in <decision>
or <tool_results>. Mention the human review when <decision> requests it. Never write money amounts.

The writing style for this order is below. It changes only HOW you write (order of sections,
length, emphasis), never WHAT is decided.
