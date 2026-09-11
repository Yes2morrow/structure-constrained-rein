import matplotlib.pyplot as plt
from core.envs.env import HouseEnv

__all__ = ["save_layout_image"]


def save_layout_image(
    env:HouseEnv,
    filename,
    size=300,
    init_livingroom=False,
    show_room_metrics:bool|None=None,
    show_total_area:bool|None=None,
    show_summary_panel:bool|None=None,
):
    """ Save layout image of the house environment """
    env.render(
        render=False,
        show_init_livingroom=init_livingroom,
        show_room_metrics=show_room_metrics,
        show_total_area=show_total_area,
        show_summary_panel=show_summary_panel,
    )
    plt.savefig(filename, dpi=size)
