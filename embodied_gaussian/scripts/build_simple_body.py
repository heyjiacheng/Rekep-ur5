from dataclasses import dataclass, field
from pathlib import Path

import tyro

from embodied_gaussians.scene_builders.domain import (
    save_posed_images,
    load_posed_images,
)
from embodied_gaussians import SimpleBodyBuilder, SimpleBodyBuilderSettings
from embodied_gaussians.utils.utils import read_extrinsics, read_ground
from utils import get_datapoints_from_live_cameras


@dataclass
class Params:
    save_path: tyro.conf.PositionalRequiredArgs[Path]
    """
    Path to a JSON file this will be saved to.
    """
    extrinsics: Path
    """
    Path to a JSON file containing the extrinsics of the cameras.
    """
    ground: Path | None = None
    """
    The method makes sure that no particles moves below this ground plane.
    """
    max_depth: float = 2.0
    """
    Maximum depth to consider for building the body.
    """
    visualize: bool = False
    """
    Visualize the body building process
    """
    offline: bool = False
    """
    Whether to use the saved posed images instead of running the cameras.
    """
    save_posed_images: bool = True
    """
    Save the posed images to a file so that the cameras don't need to be run again.
    """
    builder: SimpleBodyBuilderSettings = field(
        default_factory=SimpleBodyBuilderSettings
    )


def main(params: Params):
    assert params.save_path.suffix == ".json"
    if params.save_path.exists():
        overwrite = input(f"File {params.save_path} already exists. Overwrite? (y/n)")
        if overwrite != "y":
            print("Aborting.")
            return
    else:
        params.save_path.parent.mkdir(parents=True, exist_ok=True)

    name = params.save_path.stem

    settings = params.builder

    if params.ground is not None:
        ground_data = read_ground(params.ground)
        settings.ground.plane = ground_data

    extrinsics = read_extrinsics(params.extrinsics)
    if not params.offline:
        datapoints = get_datapoints_from_live_cameras(
            extrinsics
        )
        if params.save_posed_images:
            save_posed_images(f"temp/posed_images/{name}.npz", datapoints)
    else:
        try:
            datapoints = load_posed_images(f"temp/posed_images/{name}.npz")
        except FileNotFoundError:
            print("Posed images not found. Run with offline=False to generate them")
            return

    result = SimpleBodyBuilder.build(
        name=name, settings=settings, datapoints=datapoints, visualize=params.visualize
    )

    if result is None:
        print("Body builder failed")
        return

    with open(params.save_path, "w") as f:
        f.write(result.model_dump_json(indent=4))

    print(f"Saved body to {params.save_path}")


if __name__ == "__main__":
    params = tyro.cli(Params)
    main(params)
