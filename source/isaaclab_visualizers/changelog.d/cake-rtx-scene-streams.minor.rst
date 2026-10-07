Added
^^^^^

* Exposed task UI callbacks on both Newton GL and RTX visualizers through their shared base class.
* Added deferred RTX history reset for geometry edits during UI drawing, applied at the next frame boundary.

* Added task-owned GPU scene streams, frame completion, render-history reset and RGB capture to the standard Newton RTX visualizer.
* Added configurable asynchronous RTX rendering, distant-light rotation and static collision geometry visibility.

Fixed
^^^^^

* Load visual meshes when selecting a headless Newton RTX visualizer.
* Resolve OVRTX 0.6 render-variable paths for native window presentation, RGB capture and screenshots.
