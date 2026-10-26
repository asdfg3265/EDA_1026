"""Rasterise SVG documents with headless Chromium (same engine family as the EasyEDA editor)."""
import os


class Rasterizer:
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        exe = '/opt/pw-browsers/chromium' if os.path.exists('/opt/pw-browsers/chromium') else None
        try:
            args = ['--disable-lcd-text', '--disable-font-subpixel-positioning', '--force-color-profile=srgb', '--font-render-hinting=none']
            self.browser = (self._pw.chromium.launch(executable_path=exe, args=args) if exe and os.path.isfile(exe)
                            else self._pw.chromium.launch(args=args))
        except Exception:
            self.browser = self._pw.chromium.launch(args=['--disable-lcd-text'])
        self.page = self.browser.new_page()

    def render(self, svg, W, H, out_png):
        html = ('<html><head><style>html,body{margin:0;padding:0;background:#FFFCF8;-webkit-font-smoothing:antialiased} svg text{text-rendering:geometricPrecision}</style></head>'
                f'<body>{svg}</body></html>')
        self.page.set_viewport_size({'width': int(W), 'height': int(H)})
        self.page.set_content(html)
        self.page.screenshot(path=out_png, clip={'x': 0, 'y': 0, 'width': int(W), 'height': int(H)})

    def close(self):
        try:
            self.browser.close()
        finally:
            self._pw.stop()
