from pkg_resources import parse_version


class SimpleVersion:
    def __init__(self, tag, dev_version=None):
        self.is_released = tag != "master"
        self.version = parse_version(tag) if self.is_released else dev_version
        self.name = (
            f"{self.version.major}.{self.version.minor}.{self.version.micro}"
            if self.is_released
            else dev_version.public
        )
        self.url_base = (
            f"{self.version.major}.{self.version.minor}.X"
            if self.is_released
            else "master"
        )
        self.url = f"/{self.url_base}/index.html"

    def __repr__(self) -> str:
        return f"Version({self.version}, {self.url})"

    def __lt__(self, other):
        return self.version < other.version


def find_version_by_name(name, versions):
    for version in versions:
        if version.name == name:
            return version

    raise ValueError(f"{name} is not a defined version")
