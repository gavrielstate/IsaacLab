# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image presentation fallback; physics and OVRTX rendering remain on GPU."""


def initialize(viewer):
    """Use a standard pyglet window and Newton's existing GUI/camera controls."""
    import pyglet
    from newton._src.viewer.viewer_gui import ViewerGui

    window = pyglet.window.Window(
        width=viewer._window_width,
        height=viewer._window_height,
        caption="Newton RTX Viewer (CPU image presentation)",
        resizable=True,
        vsync=viewer._vsync,
        config=pyglet.gl.Config(double_buffer=True, red_size=8, green_size=8, blue_size=8),
    )
    viewer._window = window
    viewer._pyglet = pyglet
    viewer._pyglet_gl = pyglet.gl
    viewer._pyglet_app = pyglet.app
    viewer._cpu_color_texture = pyglet.image.Texture.create(viewer.camera.width, viewer.camera.height)

    def key_press(symbol, modifiers):
        if not viewer.gui.should_ignore_keyboard_input():
            viewer._keys_down.add(symbol)
        viewer.gui.handle_key_press(symbol, close_fn=window.close)

    def resize(width, height):
        viewer._window_width, viewer._window_height = width, height

    def close():
        viewer._should_close = True

    window.push_handlers(
        on_key_press=key_press,
        on_key_release=lambda symbol, modifiers: viewer._keys_down.discard(symbol),
        on_mouse_drag=lambda x, y, dx, dy, buttons, modifiers: viewer.gui.handle_mouse_drag(
            x, y, dx, dy, buttons, viewer._to_framebuffer_coords, modifiers
        ),
        on_mouse_press=lambda x, y, button, modifiers: viewer.gui.handle_mouse_press(
            x, y, button, viewer._to_framebuffer_coords
        ),
        on_mouse_release=lambda x, y, button, modifiers: viewer.gui.handle_mouse_release(x, y, button),
        on_mouse_scroll=lambda x, y, scroll_x, scroll_y: viewer.gui.handle_mouse_scroll(scroll_y),
        on_resize=resize,
        on_close=close,
    )
    viewer.gui = ViewerGui(viewer, window)
    viewer.gui.register_ui_callback(viewer._ui_populate_rendering_panel, position="rendering")
    for callback, position in viewer._pending_ui_callbacks:
        viewer.gui.register_ui_callback(callback, position=position)
    viewer._pending_ui_callbacks = []
    if viewer._pending_splash is not None:
        active, text = viewer._pending_splash
        if active:
            viewer.gui.show_loading_splash(text)
        else:
            viewer.gui.hide_loading_splash()
        viewer._pending_splash = None


def present(viewer, rgba):
    """Upload only the completed color image, retaining native GUI and aspect ratio."""
    window, pyglet = viewer._window, viewer._pyglet
    window.switch_to()
    height, width = rgba.shape[:2]
    image = pyglet.image.ImageData(width, height, "RGBA", rgba.tobytes(), pitch=-4 * width)
    viewer._cpu_color_texture.blit_into(image, 0, 0, 0)
    window.clear()
    ww, wh = window.get_size()
    scale = min(ww / width, wh / height)
    drawn_width, drawn_height = width * scale, height * scale
    viewer._cpu_color_texture.blit(
        (ww - drawn_width) / 2, (wh - drawn_height) / 2, width=drawn_width, height=drawn_height
    )
    viewer.gui.render_frame(update_fps=True)
    window.flip()
