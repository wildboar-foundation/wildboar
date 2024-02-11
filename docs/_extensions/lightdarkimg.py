from pathlib import Path

from docutils.parsers.rst.directives.images import Image


def light_dark_paths(path: Path):
    file = Path(path.parts[-1])
    path = Path(*path.parts[:-1])
    ext = file.suffix
    name = file.stem

    return path.joinpath(f"{name}-light{ext}"), path.joinpath(f"{name}-dark{ext}")


class LightDarkImage(Image):
    def run(self):
        image_path = self.arguments[0]
        light, dark = light_dark_paths(Path(image_path))
        self.arguments[0] = str(light)
        classes = self.options.get("classes", [])
        self.options["classes"] = classes + ["only-light"]

        ret = super().run()
        self.arguments[0] = str(dark)
        self.options["classes"] = classes + ["only-dark"]
        return ret + super().run()


def setup(app):
    setup.app = app
    app.add_directive("ldimage", LightDarkImage)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
