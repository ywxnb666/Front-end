from __future__ import annotations

import argparse


def enable_high_dpi_awareness() -> None:
    try:
        import ctypes
    except Exception:
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote project console")
    parser.add_argument("--debug", action="store_true", help="use simulated pipeline progress for frontend debugging")
    parser.add_argument("--tk", action="store_true", help="launch the original Tk frontend")
    parser.add_argument("--host", default="127.0.0.1", help="web frontend bind host")
    parser.add_argument("--port", type=int, default=8011, help="web frontend bind port")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser for the web frontend")
    args = parser.parse_args()

    if args.tk:
        enable_high_dpi_awareness()
        from app.ui.main_window import launch_app

        launch_app(debug=args.debug)
    else:
        from app.web import launch_web_app

        launch_web_app(
            debug=args.debug,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
