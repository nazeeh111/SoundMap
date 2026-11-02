from .plotting import AcousticCameraPlot
from .dashboard import Dashboard


def __getattr__(name):
    # Camera packages are optional for the classical dashboard.
    if name == "VideoStream":
        from .video import VideoStream
        return VideoStream
    raise AttributeError(name)
