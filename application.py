#!/usr/bin/env python

from ui import research_ui

# Main entry point of the application.
# Simply creates an instance of the research ui and serves it.
if __name__ == "__main__":
    ui = research_ui.ResearchUI()
    ui.serve()
