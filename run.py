import os
import numpy as np
from PIL import Image
from datetime import datetime
from argparse import ArgumentParser, SUPPRESS
from GA import ImageEvolutionTask, GeneticAlgorithm
from utils import resize_image, save_evolution_progress_as_gif
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_FILENAME_SUFFIX,
    ANIMATION_IMAGE_WIDTH,
    ANIMATION_FILENAME_SUFFIX,
    ANIMATION_DURATION,
    DEFAULT_RUN_PARAMS,
    TASK_PARAMS,
    GA_PARAMS
)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False, description='Run image evolution using a genetic algorithm.')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )
    required.add_argument('-i', '--image_filename', required=True, type=str,
                          help=f'input image filename (must be placed in {INPUT_DIR}) directory')
    optional.add_argument('--n_iter', default=None, type=float,
                          help='number of algorithm iterations (generations)')
    optional.add_argument('--n_seconds', default=None, type=float,
                          help='duration of algorithm execution (in seconds)')
    optional.add_argument('--no_animation', action='store_false', dest='animation',
                          help='do not save an evolution progress as a GIF animation')
    optional.add_argument('--img_history_step', default=DEFAULT_RUN_PARAMS['img_history_step'], type=int,
                          help='frequency at which algorithm outputs are included in the GIF animation'
                               '(e.g. 10 for every 10th iteration)')
    args = parser.parse_args()

    n_iter = args.n_iter
    n_seconds = args.n_seconds
    animation = args.animation
    img_history_step = args.img_history_step

    if args.n_iter is None and args.n_seconds is None:
        n_iter = DEFAULT_RUN_PARAMS['n_iter']

    if not animation:
        img_history_step = None

    input_img = Image.open(os.path.join(INPUT_DIR, args.image_filename))
    input_img_array = np.array(input_img)

    task = ImageEvolutionTask(input_img_array, **TASK_PARAMS)
    ga = GeneticAlgorithm(**GA_PARAMS)

    best_img, _, current_best_img_history, _, _, _ = ga.fit(
        task=task,
        n_iter=n_iter,
        n_seconds=n_seconds,
        img_history_step=img_history_step
    )

    current_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    output_filename = f'{args.image_filename.split(".")[0]}_{OUTPUT_FILENAME_SUFFIX}_{current_time}.jpg'
    Image.fromarray(best_img).save(os.path.join(OUTPUT_DIR, output_filename))

    if animation:
        input_img_array_resized = resize_image(input_img_array, width=ANIMATION_IMAGE_WIDTH)

        animation_filename = f'{args.image_filename.split(".")[0]}_{ANIMATION_FILENAME_SUFFIX}_{current_time}.gif'
        save_evolution_progress_as_gif(
            img_history=current_best_img_history,
            original_img=input_img_array_resized,
            duration=ANIMATION_DURATION,
            output_path=os.path.join(OUTPUT_DIR, animation_filename)
        )
