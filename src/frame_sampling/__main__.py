"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Frame Sampling."""


if __name__ == "__main__":
    main(prog_name="frame-sampling")  # pragma: no cover
