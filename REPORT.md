# Group Assignment Report

- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render): https://aisle-run-daiist.onrender.com/
- **Final commit** (tag `final`): pending the team evaluation

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

## Business case

Aisle Run helps shoppers find the products on a list and decide which shelf
to visit next. The proposed baseline is visiting each item's shelf in the
order it appears on the list. Both routes use the same entrance, checkout,
and walkable map. Compare distances on cases with a valid route, and measure
correct route completion and correct review decisions across the full set.

**Worked example (one simulated shopping trip).** The list reads pasta,
tomatoes, eggs, and cereal. Following list order takes 95.75 m on the sample
map. The planner groups stops and returns a 63.41 m walk, a reduction of
32.34 m (33.8%). At an assumed walking speed of 1 m/s, this is about 32
seconds of walking. If 1,000 monthly trips all saved that much walking and
shoppers valued their time at €12 per hour, the scenario would be worth
about €108 per month in shopper time. This scenario assumes the same saving
on each trip and excludes photo processing, time at the shelves, and
retailer revenue. The team will estimate real-world value from its evidence.

## Architecture

`list photo → Gemini vision → grocery lines → Gemini catalog agent → verified product IDs and shelf sides → walking route`

The vision call reads the image into ordered grocery lines and flags unclear
text. The agent receives those lines, calls the local catalog tool and returns
a match decision for each line. The application verifies each product ID
against the store catalog. The route planner uses the verified shelf stops and
the store's walkable paths to compute the route from Entrance to Checkout.

Both AI components currently use Gemini 3.5 Flash-Lite with separate prompts
and inputs. The agent uses Gemini function calling for the catalog lookup and
structured match decisions. The route planner is deterministic.

*Team to document model choices and any code reused from Assignments 1–3.*

## Data

The app bundles a synthetic supermarket catalog, floor plan, and walking
graph in `store.json`. They were prepared for the demonstration and are
independent of the evaluation photos. The application logs a generic image
summary, numerical photo brightness, redacted vision output, and verified
catalog results. Request records exclude raw photos, upload paths, and
unrestricted transcriptions.

*Team to document the evaluation images, sources and permissions, dataset
sizes, and verify that the images contain no personal details.*

## Evaluation

### Components

Report the share of labeled lines correctly read by the vision stage and
the share of catalog decisions that match the team labels. Record the
denominator for each stage and test the catalog agent with clean,
team-transcribed input so upstream reading errors can be distinguished
from matching errors. Team results are pending.

### End to end

The team will write at least 30 cases with expected product and review
decisions. One proposed mix is 17 normal, five hard, five review, and
three out-of-scope cases. Structured outputs are scored directly.

Deployment smoke checks used a grocery list containing tomatoes, pasta,
cereal, and eggs and a review list containing tomatoes and fresh salmon.
The public app returned an estimated 63 m route for the first list and
asked the shopper to check unresolved items for the second. These are
functionality checks. Team accuracy and baseline comparisons are pending.

### Cost and latency

The app logs input and output tokens, elapsed seconds, and an estimated
paid-tier cost for each AI stage and the whole request. The estimate uses
$0.30 per million input tokens, $2.50 per million output tokens, and the
illustrative conversion of €0.90 per US dollar. Actual charges depend on
the account's plan. Team-run averages are pending.

## Operations

The Monitoring tab shows the share of requests sent to review and the
share of rated requests that received a thumbs down. It compares grayscale
image brightness in recent requests with a reference set using population
stability index (PSI). A template alert rule combines a minimum PSI and a
sample-size adjustment. The deployment blueprint provisions Postgres for
request and feedback logs.

The dashboard waits for at least 20 requests and uses up to 100 recent
requests for the drift check. The team still needs 30 separate normal
reference photos, at least 10 intentionally shifted photos, and results
from the normal and shifted replay before justifying a final threshold.

## Failures

A temporary Gemini server error occurred during a deployment smoke
check. The app displayed a retry message. Report team-run failure counts
by stage once the labeled evaluation is complete.

## Decision

Ask the shopper to check the photo or ask staff when the list is unreadable
or a catalog match remains unresolved. Estimate daily reviews by multiplying
expected daily trips by the review rate measured on team cases.

**Current recommendation:** use the public app for group testing. A
real-store pilot remains a team decision informed by correct shelf
matches, route improvements over list order, staff review workload, and
an up-to-date store map and catalog.

## How to run it

Follow `SETUP.md` and provide `GEMINI_API_KEY`. The demo store file is
bundled locally. See `eval/README.md` for the offline preflight, scoring
command, and drift replay.

## AI use

Codex was used to develop and test the Aisle Run demo and Gradio code,
including the two AI components, the route planner, evaluation tooling,
this technical description, and the presentation draft and worked example.

*Team to verify and complete this disclosure before submission.*
