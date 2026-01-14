import time
from rich.markdown import Markdown
from rich.live import Live


def render_streaming_response(stream_generator, first_chunk=None):
    full_response = ""
    sources = []
    last_refresh_time = 0
    REFRESH_INTERVAL = 0.1  # Update UI max every 0.1 seconds

    with Live(Markdown(""), auto_refresh=False, vertical_overflow="visible") as live:

        def update_ui(force=False):
            nonlocal last_refresh_time
            current_time = time.time()
            if force or (current_time - last_refresh_time > REFRESH_INTERVAL):
                live.update(Markdown(full_response))
                live.refresh()
                last_refresh_time = current_time

        def process_single_chunk(chunk):
            nonlocal full_response, sources
            # Capture Sources
            if isinstance(chunk, dict) and "__sources__" in chunk:
                sources = chunk["__sources__"]
                return
            # Accumulate Text
            if isinstance(chunk, str):
                full_response += chunk
                update_ui(force=False)

        # Process the first chunk if it exists
        if first_chunk:
            process_single_chunk(first_chunk)

        # Process the rest of the stream
        for chunk in stream_generator:
            process_single_chunk(chunk)

        # Ensure final state is rendered
        update_ui(force=True)

    return full_response, sources
