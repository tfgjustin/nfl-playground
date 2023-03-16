#!/usr/bin/python

import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import sys

from space import analyze_frames, contour_levels
from tracking import append_ball_snap_position, decompose_speed_vectors, is_football, rotate_field


def get_start_frame_id(merged_df):
    snap_frame = merged_df[(merged_df.event == 'ball_snap') & (merged_df.team == 'football')]
    if snap_frame.empty:
        return 0
    snap_frame_id = snap_frame['frameId'].unique()[0]
    return max(0, snap_frame_id - 10)


def select_and_merge_data(players_df, plays_df, tracking_df, pff_df, game_id, play_id):
    merged_df = plays_df[(plays_df.gameId == game_id) & (plays_df.playId == play_id)]
    if merged_df.empty:
        return merged_df
    merged_df = pd.merge(merged_df, tracking_df, on=['gameId', 'playId'], how='left')
    # Now that we've merged in tracking data, grab only one second of pre-snap data.
    start_frame_id = get_start_frame_id(merged_df)
    merged_df = merged_df[merged_df.frameId >= start_frame_id]
    print('Play starts at frame %d' % start_frame_id)
    merged_df = pd.merge(merged_df, players_df, on=['nflId'], how='left')
    merged_df = pd.merge(merged_df, pff_df, on=['gameId', 'playId', 'nflId'], how='left')
    # Rotate all the information so all plays go bottom-to-top
    merged_df = rotate_field(merged_df)
    merged_df = decompose_speed_vectors(merged_df)
    merged_df = append_ball_snap_position(merged_df)
    return merged_df


def generate_field():
    """Generates a realistic american football field with line numbers and hash marks.

    Returns:
        [tuple]: (figure, axis)
    """
    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1,
                             edgecolor='black', facecolor='green', zorder=0)
    fig, ax = plt.subplots(1, figsize=(5.33, 12))
    ax.add_patch(rect)

    # line numbers
    plt.plot([0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             [10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             color='white')
    for y in range(20, 110, 10):
        numb = y
        if y > 50:
            numb = 120-y
        plt.text(5, y-1, str(numb - 10), horizontalalignment='center', fontsize=20, color='white', rotation=270)
        plt.text(53.3-5, y-0.75, str(numb-10),
                 horizontalalignment='center', fontsize=20, color='white', rotation=90)

    # hash marks
    for y in range(11, 110):
        ax.plot([0.4, 0.7], [y, y], color='white')
        ax.plot([53.0, 52.5], [y, y], color='white')
        ax.plot([22.91, 23.57], [y, y], color='white')
        ax.plot([29.73, 30.39], [y, y], color='white')

    # set limits and hide axis
    plt.xlim(0, 53.3)
    plt.ylim(0, 120)
    plt.axis('off')
    return fig, ax


# Color of the circle we're drawing on the image
def get_circle_color(row):
    if is_football(row):
        return 'darkgoldenrod'
    elif row['pff_positionLinedUp'] == 'QB':
        return 'red'
    elif row['team'] == row['possessionTeam']:
        return 'indianred'
    else:
        return 'skyblue'


# The football is small, and everything else is big
def get_circle_size(row):
    return 50 if is_football(row) else 150


def add_object_to_chart(row, added_objects, labelNumbers=False, showArrow=True):
    _ARROW_MULT = 0.5
    # TODO: Better handle when we're not tracking the objects we add.
    if added_objects is None:
        added_objects = list()
    added_objects.append(
        plt.scatter(row['normX'], row['normY'], color=get_circle_color(row), s=get_circle_size(row), zorder=10)
    )
    if is_football(row):
        # Footballs do not have jerseys or arrows
        return
    if showArrow:
        added_objects.append(
            plt.arrow(row['normX'], row['normY'], row['xSpeed'] * _ARROW_MULT, row['ySpeed'] * _ARROW_MULT,
                      color='black', width=0.15, zorder=5)
        )
    if labelNumbers:
        added_objects.append(
            plt.annotate(int(row['jerseyNumber']), (row['normX'], row['normY']),
                         xytext=(row['normX'] - 0.4, row['normY'] - 0.3), fontsize=6, color='white', zorder=15)
        )


def draw_objects(object_rows, all_objects, labelNumbers=False, showArrow=True):
    for _, row in object_rows.iterrows():
        add_object_to_chart(row, all_objects, labelNumbers=labelNumbers, showArrow=showArrow)


def cleanup_objects(objects):
    for obj in objects:
        obj.remove()
        del obj
    objects.clear()


def create_frame_filename(base_directory, game_id, play_id, frame_id):
    # TODO: Use path joining
    return '%s/g%s_p%05d_f%03d.png' % (base_directory, game_id, play_id, frame_id)


def draw_influence(x, y, influence, ax):
    ax.contourf(x, y, influence, contour_levels(), alpha=0.75, cmap='bwr', extend='both', zorder=3)
    min_i = np.min(influence)
    max_i = np.max(influence)
    sum_i = float(np.sum(influence, axis=None))
    ax.text(8, 3, 'Total: %6.2f [%5.3f - %5.3f]' % (sum_i, min_i, max_i), fontsize=20, color='white', zorder=10)


def debug_ax(filename, ax):
    print(filename, 'AxChildren')
    for child in ax.get_children():
        print(child)
    print(filename, 'AxObj')
    for obj in ax.findobj():
        print(obj)


def create_one_frame(game_id, play_id, frame_id, df, x, y, influence, base_directory, objects=None):
    # Now create the figure and draw the objects
    fig, ax = generate_field()
    draw_objects(df, objects, labelNumbers=True, showArrow=True)
    draw_influence(x, y, influence, ax)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # Write the plot to a file and save the filename
    filename = create_frame_filename(base_directory, game_id, play_id, frame_id)
    # debug_ax(filename, ax)
    plt.savefig(filename)
    plt.close()
    return filename


def create_frames(merged_df, base_directory, frame_files):
    fq_frame_ids = merged_df.groupby(['gameId', 'playId', 'frameId']).groups
    print('Found %d total frames' % len(fq_frame_ids))
    # TODO: Flip this back on to keep tracking the objects in the graph.
    objects = None
    x, y = np.mgrid[0:53.3:.1, 0:120:.1]
    locations = np.dstack((x, y))
    fq_frame_id_to_influence = analyze_frames(merged_df, fq_frame_ids, locations)
    print('Produced analysis for %d frames' % len(fq_frame_id_to_influence))
    for fq_frame_id, influence in sorted(fq_frame_id_to_influence.items()):
        frame_data = merged_df[(merged_df.gameId == fq_frame_id[0]) & (merged_df.playId == fq_frame_id[1]) &
                               (merged_df.frameId == fq_frame_id[2])]
        filename = create_one_frame(fq_frame_id[0], fq_frame_id[1], fq_frame_id[2], frame_data, x, y, influence,
                                    base_directory, objects=objects)
        frame_files.append(filename)


def create_gif_filename(base_directory, game_id, play_id):
    # TODO: Use path joining
    return '%s/g%s_p%05d.gif' % (base_directory, game_id, play_id)


def create_animated_gif(game_id, play_id, base_directory, frame_files):
    last_data = None
    gif_filename = create_gif_filename(base_directory, game_id, play_id)
    with imageio.get_writer(gif_filename, mode='I') as writer:
        for filename in frame_files:
            last_data = imageio.imread(filename)
            writer.append_data(last_data)
        if last_data is None:
            return
        for _ in range(15):
            writer.append_data(last_data)


def visualize_play(game_id, play_id, merged_df, base_directory):
    frame_files = list()
    create_frames(merged_df, base_directory, frame_files)
    create_animated_gif(game_id, play_id, base_directory, frame_files)


def main(argv):
    if len(argv) != 9:
        print('Usage: %s <game_id> <play_id> <games_csv> <players_csv> <plays_csv> <tracking_csv> <pff_csv> <out_dir>' %
              argv[0])
        return 1
    game_id = int(argv[1])
    play_id = int(argv[2])
    # games_df = pd.read_csv(argv[3])
    players_df = pd.read_csv(argv[4])
    plays_df = pd.read_csv(argv[5])
    tracking_df = pd.read_csv(argv[6])
    pff_df = pd.read_csv(argv[7])
    merged_df = select_and_merge_data(players_df, plays_df, tracking_df, pff_df, game_id, play_id)
    print('Found %d points for (%d, %d)' % (len(merged_df), game_id, play_id))
    visualize_play(game_id, play_id, merged_df, argv[8])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
