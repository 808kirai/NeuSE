AI safety and security

NeuSE helps decide where rescuers look for people. A wrong answer can send a team the wrong way or make them stop searching ground where someone is waiting, and the data it holds shows where people are. This file sets out what could go wrong, what NeuSE does about it, and what is left for people to handle. Each measure has tests, named below, that run on every push.

The measures follow published guidance

- OWASP Top 10 for LLM Applications 2025, for the language model that writes reports. Prompt injection is LLM01, sensitive information disclosure LLM02, supply chain LLM03 and improper output handling LLM05
- OWASP Machine Learning Security Top 10, for the detectors. ML06 is AI supply chain attacks, where a model file is swapped or changed
- OWASP API Security Top 10, for the command center's API
- The NIST AI Risk Management Framework, whose four functions, govern, map, measure and manage, shape this file. Map is the threat list below, measure is the tests and the scores in the readme, manage is the measures, and govern is the rules for people at the end
- Kenya's Data Protection Act 2019. Section 25 asks that personal data is limited to what is needed and kept no longer than needed. A sighting's position is about a person, so it is treated as personal data

Who and what to protect against

- A field unit that is lost, stolen or left unattended, whose files can be read or changed
- Someone on the network between a unit and the command center
- Someone on the internet who finds the command center's address and tries to read it or guess a token
- A unit, or someone holding a unit's token, sending false or harmful data
- A language model that makes things up, or is steered by text hidden in the data it is given
- Faults that are not attacks at all, a camera that goes blind, a model that quietly stops working, a clock that is wrong
- People deliberately hiding from thermal cameras. Published work shows patches of warming and cooling material can hide a person from infrared detectors, with over 90% success reported in some tests

Safety of the detection itself

- A blind camera never makes ground look searched. A missing, unreadable, flat or saturated frame does not mark any ground as searched, and the unit shows as not searching while it lasts. Frames count as blind when their brightness spreads less than 2 grey levels, `NEUSE_MIN_FRAME_SPREAD`. The flattest of HIT-UAV's 2,898 real frames spreads 11. Tests in `tests/test_coverage.py`
- A unit that stops detecting is shown. The map marks a unit that is in contact but whose detector has been quiet for 30 seconds as not searching, so a coordinator never mistakes a broken unit for empty ground
- Confidence means what it says. Scores are calibrated on the validation images, so a hit shown as 0.9 is a person about 9 times in 10. On the test images, hits at 0.9 or more were people 92% of the time
- Sightings are never turned into a head count. One person can appear in many frames, so the map, reports and alerts count sightings. A model report that counts people is refused, see below
- Every number is measured on held out data and reported with its limits, including how the detector does on flights it has never seen and where it is weakest, by day and at 120 to 130 m
- Searched ground is not proof nobody is there. The detector finds about two thirds of people on the test images, so a searched square can still hold someone. The readme, reports and the technical paper say so, and planned future work turns the searched squares into a probability of detection

Model files, OWASP ML06 and LLM03

- Each trained model's SHA-256 is kept in `models/checksums.txt`, written by the training scripts, and checked before the unit loads the model. A changed or damaged file is refused and the unit falls back to a detector it can trust, from the MindYOLO model or the person CNN to the person check, and from the person check to warm spots. Damaged person check weights used to crash the unit. Tests in `tests/test_integrity.py`
- The MindYOLO model, which is not kept in git, can be pinned with `NEUSE_MINDYOLO_SHA256` or by adding it to `models/checksums.txt`
- Weights are loaded with NumPy without pickle, so a weights file cannot run code

The language model for reports, OWASP LLM01, LLM02 and LLM05

- Only measured facts go to the model, as JSON, and every text value in them is a plain name of letters, digits, dots, dashes and underscores. Units cannot send free text, such as a detector name with instructions in it. Tests in `tests/test_security.py`
- The model is told the JSON is data and never instructions
- Every number in the model's report must come from the facts, allowing rounding. A report with any other number, one that counts people, or one over 3,000 characters is not shown, and the template report is used with a note saying why. Control characters are removed. The facts always come back with the text so a coordinator can check it
- The facts only go to a model over HTTPS, or to one on the same computer, since they show where people are
- If the model is unreachable or misbehaves, the template writes the report, which needs no network

The command center, OWASP API Security

- Nothing is given out without a token. A unit token can only send, and only for its own unit, so a lost unit cannot read where people are or pretend to be another unit. Tokens are compared in constant time
- The command center will not open to a network with a token shorter than 16 characters or one still holding the example value from `.env.example`
- A caller that gives 20 wrong tokens in 5 minutes is turned away for the rest of that window, even with a right token, so guessing is slow and a lucky guess tells it nothing. Behind Caddy, `NEUSE_CC_TRUST_PROXY=1` makes this count each caller's own address. Units treat the refusal like a dropped link and keep their sightings
- Every page and reply carries headers that stop it being framed by another site or read as another file type, and other sites are told only the server's own address, never a page's path. OpenStreetMap's tile servers require that address and refuse map tiles without it. Replies holding positions are never cached. The map page has a content security policy that only runs its own scripts, with a new nonce for each request, and everything a unit sends is put on the page as text, never as HTML
- Every API request is logged with who made it, the coordinator, a unit's name or refused, but never the token. Everything taken from the request is escaped and cut to 200 characters, so a crafted address cannot forge a line in the log. CodeQL found this, see below
- Request bodies are capped at 2 MB, and 16 MB once unzipped, so a crafted upload cannot fill the memory. Every field of every sighting is checked, and one bad row is refused on its own without blocking the rest
- HTTPS comes from Caddy, which also tells browsers to always use HTTPS for the address. A unit refuses to send its token over plain HTTP to another computer
- The command center runs in Docker as an ordinary user, not as root

Personal data, Kenya Data Protection Act 2019

- Positions of people are treated as personal data
- `NEUSE_CC_KEEP_DAYS` removes sightings, searched ground and alerts older than that many days from the command center, checked at most once an hour. `NEUSE_KEEP_DAYS` does the same for sightings the command center has confirmed on each unit. Both keep everything by default, since how long a search needs its records is for the team to decide. Set them once a search is over
- Only what the search needs is kept. No images leave the unit, only boxes, positions, times and scores
- Alert texts carry an area's position and the number of sightings, nothing about who anyone is

The code and its packages, OWASP LLM03 and ML06

- Every package is pinned to an exact version
- CI audits every installed package for known vulnerabilities with pip-audit, installed exactly as the Docker image installs them
- The audit found that Huawei Cloud's SMN SDK pins pymongo 4.18.2 and pyasn1 0.6.3, which have known vulnerabilities, even in its newest release. The SDK is now installed without its pins, from `command_center/requirements-smn.txt`, with pymongo 4.18.3 and pyasn1 0.6.5 from `command_center/requirements.txt`. Publishing through the SDK to a test server was checked with these versions, and the request was signed and sent as SMN expects
- GitHub's CodeQL code scanning checks the Python code on every push to main, every pull request and once a week. Its first scan found 7 issues. Three were real, the access log wrote the request path as given, so a path with an encoded line break could forge a log line. Four were in the tests, which checked HTML with a pattern and addresses by matching text. All 7 were fixed, the tests now read the page with an HTML parser and the policy by its parts
- Dependabot proposes updates for Python packages, GitHub Actions and the Docker base image, and each one must pass the tests
- The CI jobs get a read only token

Risks that remain, and what people must do

- Hiding from thermal cameras on purpose, with adversarial patches or heat blocking clothing, can defeat any thermal detector. NeuSE does not defend against it. Searches for people who want to be found are its purpose, and the alert rule needs several sightings, which helps against single false hits but not against hiding
- The detector misses about a third of people on the test images, more by day and from high up. Teams must not treat searched ground as cleared ground
- A unit's own files are not encrypted. A lost unit gives away the sightings it holds and its token. Revoke a lost unit's token by removing it from `NEUSE_CC_UNIT_TOKENS` and restarting the command center
- The coordinator token is one shared secret. Anyone holding it can read everything. Share it only with coordinators and change it after a search
- Ground positions assume flat ground and the camera's view angles, which are a guess until set for the real camera
- NeuSE has not been used in a real search. It supports human searchers and never replaces their judgement

Reporting a security problem

See `SECURITY.md`.
