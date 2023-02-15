import subprocess
import re

from pkg_resources import parse_version


class Version:
    def __init__(self, tag):
        self.is_released = tag != "master"
        self.version = (
            parse_version(tag) if self.is_released else parse_version("1000000000.0.0")
        )
        self.name = (
            f"{self.version.major}.{self.version.minor}.{self.version.micro}"
            if self.is_released
            else "main (dev)"
        )
        self.url_base = (
            f"{self.version.major}.{self.version.minor}.X"
            if self.is_released
            else "master"
        )
        self.url = f"/{self.url_base}/index.html"

    def __repr__(self) -> str:
        return f"Version({self.name}, {self.url})"

    def __lt__(self, other):
        return self.version < other.version


def find_version_by_major_minor(major, minor, versions):
    for version in versions:
        if version.version.major == major and version.version.minor == minor:
            return version

    raise ValueError(f"{major}.{minor} is not a defined version")


def is_tag_version(tag):
    # Excluding major version of 0
    SEMVER = (
        r"^v?([1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    return re.match(SEMVER, tag)


def get_versions_from_git():
    with subprocess.Popen(["git tag"], stdout=subprocess.PIPE, shell=True) as cmd:
        tags, _ = cmd.communicate()
        tags = [tag.strip() for tag in tags.decode().splitlines()]
        return [Version(tag) for tag in tags if is_tag_version(tag)]


def get_current_branch_from_git():
    return (
        subprocess.run("git branch --show-current", stdout=subprocess.PIPE, shell=True)
        .stdout.decode()
        .strip()
    )


def get_latest_version_major_minor():
    versions = {}

    for version in get_versions_from_git():
        major_minor = f"{version.version.major}.{version.version.minor}"
        if major_minor not in versions:
            versions[major_minor] = version
        else:
            old_version = versions[major_minor]
            if version > old_version:
                versions[major_minor] = version

    return [value for _, value in sorted(versions.items(), reverse=True)]


def load_version_html_context(release):
    versions = get_latest_version_major_minor()
    latest_stable_version = versions[0]
    develop_version = Version("master")

    # Render the development version first
    versions.insert(0, develop_version)
    return {
        "versions": versions,
        "stable_version": latest_stable_version,
        "develop_version": develop_version,
        "current_version": (
            find_version_by_major_minor(release.major, release.minor, versions)
            if get_current_branch_from_git() != "master"
            else develop_version
        ),
    }
