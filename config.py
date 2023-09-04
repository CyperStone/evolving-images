INPUT_DIR = './inputs'
OUTPUT_DIR = './outputs'
OUTPUT_FILENAME_SUFFIX = 'evolved'
ANIMATION_IMAGE_WIDTH = 240
ANIMATION_FILENAME_SUFFIX = 'evolution_progress'
ANIMATION_DURATION = 15

DEFAULT_RUN_PARAMS = {
    'n_iter': 1000,
    'img_history_step': 10
}

TASK_PARAMS = {
    'n_triangles': 150,
    'internal_pixels': 10000,
    'max_triangle_field': 0.02,
    'triangle_fields_penalty_strength': 5
}

GA_PARAMS = {
    'population_size': 50,
    'selection_cutoff': 0.15,
    'crossover_rate': 1.,
    'mutation_rate': 0.01,
    'mutation_amount': 0.1
}
