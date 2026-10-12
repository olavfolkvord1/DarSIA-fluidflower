"""Seal actual complete Policy A predictions for review, with Claude scoring pending."""
from __future__ import annotations

import argparse
import json

from darsia.fluidflower import phase_ffac_common_mass_inputs as assets
from darsia.fluidflower.phase_ffac_selected_policy_a_completion import inspect_completed_review


def seal(plan, candidate, completion, report, destination):
    result=inspect_completed_review(assets.asset(plan),candidate,assets.asset(completion),assets.asset(report))
    assets.write_new_json(destination,result)
    return assets.asset(destination)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','candidate','completion','report','destination'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(seal(args.plan,args.candidate,args.completion,args.report,args.destination)))


if __name__=='__main__':
    main()
