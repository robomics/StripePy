# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT


from a11y_pygments.github_dark.style import Style as GitHubDarkStyle
from a11y_pygments.github_light.style import Style as GitHubLightStyle
from pygments.token import Generic


class StripePyDarkStyle(GitHubDarkStyle):
    """
    Custom Pygments style based on github-dark with a distinct console prompt.
    """

    name = "stripepy-dark"
    styles = GitHubDarkStyle.styles.copy()
    styles.update({Generic.Prompt: "#D6D8F5"})


class StripePyLightStyle(GitHubLightStyle):
    """
    Custom Pygments style based on github-light with a distinct console prompt.
    """

    name = "stripepy-light"
    styles = GitHubLightStyle.styles.copy()
    styles.update({Generic.Prompt: "#4CAF50"})
