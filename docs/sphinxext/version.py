from pkg_resources import parse_version


class SimpleVersion:
    def __init__(self, branch, public=None):
        self.is_released = branch != "master"
        self.version = branch.replace(".X", "") if self.is_released else public
        self.branch = branch
        self.url = f"/{branch}/index.html"

    def __repr__(self) -> str:
        return f"Version({self.version}, {self.url})"

    def __lt__(self, other):
        return parse_version(self.version) < parse_version(other.version)


def find_version_by_name(name, versions):
    for version in versions:
        if version.version == name:
            return version

    raise ValueError(f"{name} is not a defined version")
