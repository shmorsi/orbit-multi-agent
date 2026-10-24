"""Command line for doctor, the local server, and trace export."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    """Dispatch doctor, serve, traces, and score."""
    parser = argparse.ArgumentParser(prog="orbit", description="Orbit: Multi-Agent Mission Control")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check Python, device, and one tiny task")
    doctor.add_argument("--live", action="store_true")
    doctor.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    serve = sub.add_parser("serve", help="Mission control plus live runs")
    serve.add_argument("--port", type=int, default=8741)
    traces = sub.add_parser("traces", help="Rewrite the bundled mission-control traces from the engine")
    traces.add_argument("--team", choices=("v1", "v2"), default="v1")
    sub.add_parser("score", help="Score one team once on the frozen test (see --help)", add_help=False)
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] == "score":
        from orbit_agents.eval.score_team import main as score_main

        score_main(argv[1:])
        return
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from orbit_agents.doctor import main as doctor_main

        argv = ["--live", "--model", args.model] if args.live else []
        doctor_main(argv)
        return
    if args.command == "serve":
        from orbit_agents.serve import serve as serve_forever

        serve_forever(args.port)
        return
    from orbit_agents.trace import main as traces_main

    traces_main(["--team", args.team])


if __name__ == "__main__":
    main()
