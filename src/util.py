import streamlit as st

class StreamlitProgressHook:
    """Streamlit-compatible progress hook for PyAnnote pipeline."""

    def __init__(self):
        self.progress_bar = st.progress(0)
        self.task_progress = {}
        self.step_name = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.progress_bar.empty()

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None:
            completed = total = 1

        # Track progress for each step
        if step_name != self.step_name:
            self.step_name = step_name
            self.task_progress[self.step_name] = 0

        # Calculate overall progress percentage
        self.task_progress[self.step_name] = (completed / total) if total else 1
        overall_progress = sum(self.task_progress.values()) / len(self.task_progress)

        # Update the Streamlit progress bar
        self.progress_bar.progress(int(overall_progress * 100))
