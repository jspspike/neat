use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use super::genome::Genome;
use super::innovation::InnovationCounter;
use super::network::Network;
use super::network::Task;

/// Settings on how `Neat` should operate, important for getting good performance
#[derive(Serialize, Deserialize)]
pub struct NeatSettings {
    /// Range for connection weights from -weight to +weight
    pub weight: f32,
    /// Range for how much connection weights should mutate from -weight_mutate to +weight_mutate
    pub weight_mutate: f32,
    /// Cap for how large or small connection weights can get
    pub weight_max: f32,
    /// Chance connection weight will be mutated [0.0 - 1.0]
    pub weight_mutate_rate: f32,
    /// Chance connection will be created between two nodes [0.0 - 1.0]
    pub add_connection_rate: f32,
    /// Chance node will be created in a connection [0.0 - 1.0]
    pub add_node_rate: f32,
    /// Range for node activation to be mutated (activation is sigmoid function applied to every
    /// node)
    pub activation_mutate: f32,
    /// Chance node will have its activation mutated
    pub activation_mutate_rate: f32,
    /// Weight given to different connections when determining if two genomes are of the same
    /// species
    pub connections_diff: f32,
    /// Weight given to different connection weights when determining if two genomes are of the same
    /// species
    pub weight_diff: f32,
    /// Threshold when determining if two genomes are of the same species
    pub species_threshold: f32,
    /// Sets genomes to be feedforward, (no connections going in reverse of an aleady existing
    /// connection between two nodes)
    pub feedforward: bool,
    /// Whether to recalculate fitness if genome was from a previous generation (useful if task
    /// has some amount of randomness causing fitness to change)
    pub reset_fitness: bool,
}

impl NeatSettings {
    /// Returns `NeatSettings` with the following settings
    /// `weight`: 1.0,
    /// `weight_mutate`: 2.0,
    /// `weight_max`: 10.0,
    /// `weight_mutate_rate`: 0.8,
    /// `add_connection_rate`: 0.35,
    /// `add_node_rate`: 0.15,
    /// `activation_mutate`: 0.05,
    /// `activation_mutate_rate`: 0.1,
    /// `connections_diff`: 0.5,
    /// `weight_diff`: 0.1,
    /// `species_threshold`: 0.7,
    /// `feedforward`: true,
    /// `reset_fitness`: false
    pub fn default() -> NeatSettings {
        NeatSettings {
            weight: 1.0,
            weight_mutate: 2.0,
            weight_max: 10.0,
            weight_mutate_rate: 0.8,
            add_connection_rate: 0.35,
            add_node_rate: 0.15,
            activation_mutate: 0.05,
            activation_mutate_rate: 0.1,
            connections_diff: 0.5,
            weight_diff: 0.1,
            species_threshold: 0.7,
            feedforward: true,
            reset_fitness: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Organism {
    genome: Genome,
    fitness: Option<f32>,
}

impl Organism {
    pub fn new(genome: Genome) -> Organism {
        Organism {
            genome,
            fitness: None,
        }
    }
}

/// Struct to run genetic learning algorithm on provided Task
#[derive(Serialize, Deserialize)]
pub struct Neat<T: Task> {
    size: usize,
    population: Vec<Organism>,
    species_count: usize,
    innovations: InnovationCounter,
    settings: NeatSettings,
    best: Organism,
    phantom: PhantomData<T>,
}

impl<T: Task + std::marker::Sync> Neat<T> {
    /// Create new `Neat` with default `NeatSettings`
    ///
    /// # Arguments
    ///
    /// * `size` - Size of population, number of genomes to test per generation
    /// * `inputs` - Number of inputs into Network, should match number of outputs of given `Task`
    /// * `outputs` - Number of outputs from Network, should match number of inputs of given `Task`
    ///
    /// # Example
    ///
    /// ```ignore
    /// use neat::Neat;
    ///
    /// let mut neat = Neat::<TaskImplementation>::default(100, 4, 4);
    /// ```
    pub fn default(size: usize, inputs: u16, outputs: u16) -> Neat<T> {
        Neat::new(size, inputs, outputs, NeatSettings::default())
    }

    /// Create new `Neat`
    ///
    /// # Arguments
    ///
    /// * `size` - Size of population, number of genomes to test per generation
    /// * `inputs` - Number of inputs into Network, should match number of outputs of given `Task`
    /// * `outputs` - Number of outputs from Network, should match number of inputs of given `Task`
    /// * `settings` - Settings on how `Neat` should operate
    ///
    /// # Example
    ///
    /// ```ignore
    /// use neat::{Neat, NeatSettings};
    ///
    /// let settings = NeatSettings::default();
    /// let mut neat = Neat::<TaskImplementation>::new::(100, 4, 4);
    /// ```
    pub fn new(size: usize, inputs: u16, outputs: u16, settings: NeatSettings) -> Neat<T> {
        let mut innovations = InnovationCounter::new(inputs + outputs);

        let mut population = vec![];

        for _ in 0..size {
            let mut genome = Genome::new(inputs, outputs);
            genome.add_connection(&mut innovations, &settings);
            genome.mutate(&mut innovations, &settings);
            population.push(Organism::new(genome));
        }

        let mut best = population[0].clone();
        best.fitness = Some(f32::MIN);

        Neat {
            size,
            population,
            species_count: 0,
            innovations,
            settings,
            best,
            phantom: PhantomData,
        }
    }

    fn speciate(&mut self) -> Vec<Vec<Organism>> {
        let mut species: Vec<Vec<Organism>> = vec![];

        if self.settings.reset_fitness {
            let fitness = Network::new(self.best.genome.clone()).run::<T>();
            self.best.fitness = Some(fitness);
        }

        'population: for org in self.population.iter() {
            if org.fitness.unwrap() > self.best.fitness.unwrap() {
                self.best = org.clone();
            }

            for group in species.iter_mut() {
                if Genome::same_species(&group[0].genome, &org.genome, &self.settings) {
                    group.push(org.clone());
                    continue 'population;
                }
            }
            species.push(vec![org.clone()]);
        }

        self.species_count = species.len();

        species
    }

    fn kill(&mut self) {
        let mut species = self.speciate();

        self.population = vec![];

        for group in species.iter_mut() {
            if group.len() == 1 {
                let mut rng = rand::thread_rng();
                if rng.gen::<f32>() > 0.5 {
                    self.population.append(group)
                }
                continue;
            }

            group.sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            group.drain(group.len() / 2..);

            self.population.append(group);
        }
    }

    fn execute(&mut self) {
        let reset_fitness = self.settings.reset_fitness;
        self.population
            .par_iter_mut()
            .filter(|org| org.fitness.is_none() || reset_fitness)
            .for_each(|mut org| {
                let mut net = Network::new(org.genome.clone());
                org.fitness = Some(net.run::<T>());
            });
    }

    fn generate(&mut self) {
        self.population.shuffle(&mut rand::thread_rng());

        let cross_cap = self.size * 3 / 4;
        let length = self.population.len();

        if cross_cap > length {
            for i in 0..cross_cap - length {
                let new = Genome::cross(&self.population[i].genome, &self.population[i + 1].genome);
                self.population.push(Organism::new(new));
            }
        }

        let length = self.population.len();

        for i in 0..self.size - length {
            let mut new = self.population[i].genome.clone();
            new.mutate(&mut self.innovations, &self.settings);
            self.population.push(Organism::new(new));
        }
    }

    /// Goes through one step of progressing a generation. First it executes the task for the
    /// entire population to find their fitness, removes less fit genomes, and finally generates
    /// new genomes and modifies surviving ones. Returns the `Network` and fitness of most fit
    /// genome from that step.
    pub fn step(&mut self) -> (Network, f32) {
        self.execute();
        self.kill();
        self.generate();

        (
            Network::new(self.best.genome.clone()),
            self.best.fitness.unwrap(),
        )
    }

    /// Returns the number of species that existed in the last step. Useful for determining
    /// what to modify in `NeatSettings`
    pub fn species(&self) -> usize {
        self.species_count
    }
}
