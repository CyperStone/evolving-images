from __future__ import annotations
import time
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from colour.difference.delta_e import delta_E_CIE1976
from utils import (
    resize_image,
    generate_vertices,
    generate_color,
    max_possible_resolution,
    triangle_fields
)


class ImageEvolutionTask:
    """
    Class representing the image evolution task.

    Parameters
    ----------
    img : np.ndarray
        Original image.
    n_triangles : int
        Number of triangles to use for evolution.
    internal_pixels : int
        Equivalent to a resolution of internally processed images (used to keep the aspect ratio of the original image).
        Higher values improve the accuracy but decrease the performance.
    max_triangle_field : float, optional
        Maximum field of a single triangle as a fraction of the original image's area.
    triangle_fields_penalty_strength : float, optional
        The penalty strength for triangle maximum field violations.
        If specified, an extra component is added to the fitness function.
    """

    def __init__(
            self,
            img: np.ndarray,
            n_triangles: int,
            internal_pixels: int,
            max_triangle_field: float = 0.5,
            triangle_fields_penalty_strength: float = None
    ):
        self.img = img
        self.n_triangles = n_triangles
        self.internal_pixels = internal_pixels
        self.max_triangle_field = max_triangle_field
        self.triangle_fields_penalty_strength = triangle_fields_penalty_strength

        resized_height, resized_width = max_possible_resolution(self.img, self.internal_pixels)
        self.resized_img = resize_image(self.img, height=resized_height)
        self.resized_img_field = resized_height * resized_width

        self._init_triangle_max_side_length = None
        self._worst_possible_image_matching = None
        self._max_possible_triangle_fields_cost = None

    def get_original_shape(self) -> tuple[int, ...]:
        """Get the shape of the original image."""
        return self.img.shape

    def get_resized_shape(self) -> tuple[int, ...]:
        """Get the shape of the resized image."""
        return self.resized_img.shape

    def get_init_triangle_max_side_length(self):
        """Get the initial maximum side length of a triangle."""
        if self._init_triangle_max_side_length is None:
            self._init_triangle_max_side_length = np.sqrt(self.max_triangle_field)

        return self._init_triangle_max_side_length

    def get_worst_possible_image_matching(self) -> float:
        """Get the worst possible image matching score."""
        if self._worst_possible_image_matching is None:
            get_opposite_img = np.vectorize(lambda v: 255 if (255 - v) > v else 0)
            opposite_img = get_opposite_img(self.resized_img).astype(np.uint8)
            self._worst_possible_image_matching = np.sum(delta_E_CIE1976(self.resized_img, opposite_img))

        return self._worst_possible_image_matching

    def get_max_possible_triangle_fields_cost(self) -> float:
        """Get the maximum possible cost of triangles' field violation."""
        if self._max_possible_triangle_fields_cost is None:
            self._max_possible_triangle_fields_cost = 0.5 - self.max_triangle_field

        return self._max_possible_triangle_fields_cost

    def triangle_fields_cost(self, genome: np.ndarray) -> float:
        """Calculate the triangles' field violation cost for a given individual."""
        fields = triangle_fields(genome, self.get_resized_shape())
        fields /= self.resized_img_field

        mean_error = np.sqrt(
            np.mean(np.where(fields <= self.max_triangle_field, 0, (fields - self.max_triangle_field) ** 2)))
        return mean_error / self.get_max_possible_triangle_fields_cost()

    def evaluate(self, img: np.ndarray, genome: np.ndarray) -> float:
        """Calculate the fitness of a given individual."""
        fitness = 1 - np.sum(delta_E_CIE1976(self.resized_img, img)) / self.get_worst_possible_image_matching()
        if self.triangle_fields_penalty_strength is None:
            return fitness

        else:
            return np.clip(fitness - self.triangle_fields_penalty_strength * self.triangle_fields_cost(genome), 0, 1)


class Individual:
    """
    Class representing an individual in a genetic algorithm.

    Parameters:
    ----------
    task : ImageEvolutionTask
        Image evolution task instance.
    genome : np.ndarray, optional
        Genetic representation of the individual. By default, it's initialized randomly.
    """

    def __init__(self, task: ImageEvolutionTask, genome: np.ndarray = None):
        self.task = task
        self._evaluation = None
        self._resized_img = None

        if genome is None:
            self.genome = np.array([generate_vertices(self.task.get_init_triangle_max_side_length()) + generate_color()
                                    for _ in range(self.task.n_triangles)])
        else:
            self.genome = genome

    def get_img(self, full_size: bool = False) -> np.ndarray:
        """Get the image representation of the individual."""
        img_shape = self.task.get_original_shape() if full_size else self.task.get_resized_shape()

        if self._resized_img is None or full_size:
            img = Image.fromarray(np.zeros(img_shape, dtype=np.uint8))
            draw = ImageDraw.Draw(img, 'RGBA')

            for gene in self.genome:
                vertices = [
                    (int(gene[0] * img_shape[1]), int(gene[1] * img_shape[0])),
                    (int(gene[2] * img_shape[1]), int(gene[3] * img_shape[0])),
                    (int(gene[4] * img_shape[1]), int(gene[5] * img_shape[0])),
                ]
                color = (int(gene[6] * 255), int(gene[7] * 255), int(gene[8] * 255), int(gene[9] * 255))
                draw.polygon(vertices, fill=color)

            if full_size:
                return np.array(img)

            self._resized_img = np.array(img)

        return self._resized_img

    def evaluate(self) -> float:
        """Calculate the fitness of the individual."""
        if self._evaluation is None:
            self._evaluation = self.task.evaluate(self.get_img(full_size=False), self.genome)
        return self._evaluation

    def mutate(self, mutation_rate: float, mutation_amount: float) -> None:
        """Mutate the individual's genome."""
        self._evaluation = None
        self._resized_img = None

        mutation_func = np.vectorize(
            lambda v: np.clip(v + np.random.rand() * mutation_amount * 2 - mutation_amount, 0, 1)
            if np.random.rand() < mutation_rate else v)

        self.genome = mutation_func(self.genome)


class Population:
    """
    Class representing a population of individuals in a genetic algorithm.

    Parameters:
    ----------
    task : ImageEvolutionTask
        Image evolution task instance.
    population_size : int, optional
        Size of the initial population.
    """

    def __init__(self, task: ImageEvolutionTask, population_size: int = None):
        self.task = task
        self.size = population_size
        self.population = []

        if population_size is not None:
            self.population = [Individual(task=self.task) for _ in range(self.size)]

    def breed_offspring(
            self,
            selection_cutoff: float,
            crossover_rate: float,
            mutation_rate: float,
            mutation_amount: float
    ) -> Population:
        """Create a next population from the current one."""
        self.population.sort(reverse=True, key=lambda individual: individual.evaluate())

        n_selected = int(np.floor(len(self.population) * selection_cutoff))
        n_random_per_selected = int(np.ceil(1 / selection_cutoff))
        new_population = Population(task=self.task)

        for selected_idx in range(n_selected):
            for _ in range(n_random_per_selected):
                random_individual_idx = np.random.randint(0, n_selected)

                if random_individual_idx == selected_idx:
                    if random_individual_idx == n_selected - 1:
                        random_individual_idx -= 1

                    else:
                        random_individual_idx += 1

                parent_1 = self.population[selected_idx]
                parent_2 = self.population[random_individual_idx]
                child = self.crossover(crossover_rate, parent_1, parent_2)
                child.mutate(mutation_rate, mutation_amount)
                new_population.add_child(child)

        return new_population

    def crossover(self, crossover_rate: float, parent_1: Individual, parent_2: Individual) -> Individual:
        """Create a child individual by performing a crossover operation between two parent individuals."""
        if np.random.rand() < crossover_rate:
            genome_size = len(parent_1.genome)
            mask = (np.random.choice(2, genome_size) == 0)
            new_genome = np.where(mask.reshape((-1, 1)), parent_1.genome, parent_2.genome)
            return Individual(task=self.task, genome=new_genome)

        else:
            return parent_1

    def add_child(self, child: Individual) -> None:
        """Add an individual to the population."""
        self.population.append(child)
        self.size = len(self.population)

    def evaluate(self) -> tuple[np.ndarray, float, float]:
        """Evaluate the population and get the best img, the best fitness, and mean fitness of all individuals."""
        fitnesses = [individual.evaluate() for individual in self.population]
        mean_fitness = np.mean(fitnesses)

        best_individual_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_individual_idx]
        best_img = self.population[best_individual_idx].get_img(full_size=True)

        return best_img, best_fitness, mean_fitness


class GeneticAlgorithm:
    """
    Genetic algorithm class.

    Parameters:
    ----------
    population_size : int
        Size of the initial population.
    selection_cutoff : float
        Fraction of individuals selected for breeding in each generation.
    crossover_rate : float
        Probability of crossover between two selected individuals.
    mutation_rate : float
        Probability of mutation occurring for each gene in an individual's genome.
    mutation_amount : float
        Amount by which genes can be mutated.
    """

    def __init__(
            self,
            population_size: int,
            selection_cutoff: float,
            crossover_rate: float,
            mutation_rate: float,
            mutation_amount: float
    ):
        self.population_size = population_size
        self.selection_cutoff = selection_cutoff
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount

    def fit(
            self,
            task: ImageEvolutionTask,
            n_iter: int = None,
            n_seconds: int = None,
            img_history_step: int = None
    ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]], list[float], list[float], list[float]]:
        """Execute the genetic algorithm on a specified image evolution task."""
        if n_iter is None and n_seconds is None:
            raise ValueError('At least one from (n_iter, n_seconds) params must be specified.')

        progress_bar = tqdm(desc='Image evolution', total=n_iter)

        if n_iter is None:
            n_iter = np.inf

        if n_seconds is None:
            n_seconds = np.inf

        fixed_img_history_checkpoints = []
        if img_history_step is not None:
            if 1 < img_history_step <= 2:
                fixed_img_history_checkpoints = [i for i in range(1, 21)]

            elif 2 < img_history_step <= 5:
                fixed_img_history_checkpoints = [i for i in range(1, 21)] + [i for i in range(22, 52, 2)]

            elif img_history_step > 5:
                fixed_img_history_checkpoints = [i for i in range(1, 21)] + [i for i in range(22, 52, 2)] + \
                                                [i for i in range(55, 105, 5)]

        best_fitness_history = []
        current_best_fitness_history = []
        mean_fitness_history = []
        current_best_img_history = []
        best_img_history = []
        best_img = None
        best_fitness = -np.inf

        population = Population(task=task, population_size=self.population_size)

        i = 1
        start_time = time.time()

        while i <= n_iter and (time.time() - start_time) < n_seconds:
            new_population = population.breed_offspring(
                self.selection_cutoff, self.crossover_rate, self.mutation_rate, self.mutation_amount
            )

            current_best_img, current_best_fitness, current_mean_fitness = population.evaluate()
            current_best_fitness_history.append(current_best_fitness)
            mean_fitness_history.append(current_mean_fitness)

            if current_best_fitness > best_fitness:
                best_img = current_best_img
                best_fitness = current_best_fitness

            if (img_history_step is not None) and (i in fixed_img_history_checkpoints or i % img_history_step == 0):
                best_img_history.append((i, best_img))
                current_best_img_history.append((i, current_best_img))

            best_fitness_history.append(best_fitness)
            population = new_population

            progress_bar.update(1)
            i += 1

        return (
            best_img,
            best_img_history,
            current_best_img_history,
            current_best_fitness_history,
            best_fitness_history,
            mean_fitness_history
        )
