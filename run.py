import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="RhythmFall Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Shut down automatically after N seconds of no requests (0 = disabled)",
    )
    args = parser.parse_args()

    from app import create_app
    app = create_app()

    if args.idle_timeout > 0:
        from app.shutdown import start_idle_timer
        start_idle_timer(args.idle_timeout)

    app.run(debug=False, port=args.port, host=args.host, use_reloader=False)


if __name__ == "__main__":
    main()
