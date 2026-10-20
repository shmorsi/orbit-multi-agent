"""Command line for doctor, the local server, and trace export."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="orbit", description="Orbit: Multi-Agent Mission Control")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check Python, device, and one tiny task")
    doctor.add_argument("--live", action="store_true")
    doctor.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    serve = sub.add_parser("serve", help="Mission control plus live runs")
    serve.add_argument("--port", type=int, default=8741)
    sub.add_parser("traces", help="Rewrite the bundled mission-control traces from the engine")
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

    traces_main()


if __name__ == "__main__":
    main()
