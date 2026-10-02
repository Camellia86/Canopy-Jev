# SPDX-License-Identifier: MIT
"""Direct Python usage; all paths and repository choices are supplied by the user."""
import argparse
import json
from pathlib import Path
from canopy_jev import CanopyJev

p = argparse.ArgumentParser()
p.add_argument('--artifacts', required=True)
p.add_argument('--base-model', required=True)
p.add_argument('--revision')
p.add_argument('--request', type=Path, default=Path(__file__).with_name('request.json'))
a = p.parse_args()
model = CanopyJev.from_pretrained(a.artifacts, base_model=a.base_model, revision=a.revision)
request = json.loads(a.request.read_text())
print(json.dumps(model(request['state'], request['questions']), indent=2))
model.assert_unchanged()
