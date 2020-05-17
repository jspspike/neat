use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::marker::PhantomData;

use super::genome::Genome;
use super::innovation::InnovationCounter;
use super::network::Network;
use super::network::Task;

pub struct NeatSettings {
    pub weight: f32,
    pub weight_mutate: f32,
    pub weight_max: f32,
    pub weight_mutate_rate: f32,
    pub add_connection_rate: f32,
    pub add_node_rate: f32,
    pub activation_mutate: f32,
    pub activation_mutate_rate: f32,
    pub connections_diff: f32,
    pub weight_diff: f32,
    pub species_threshold: f32,
    pub feedforward: bool,
}

impl NeatSettings {
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
        }
    }
}

#[derive(Clone, Debug)]
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

pub struct Neat<T: Task> {
    size: usize,
    population: Vec<Organism>,
    innovations: InnovationCounter,
    settings: NeatSettings,
    best: Organism,
    phantom: PhantomData<T>,
}

impl<T: Task> Neat<T> {
    pub fn default(size: usize, inputs: u16, outputs: u16) -> Neat<T> {
        Neat::new(size, inputs, outputs, NeatSettings::default())
    }

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
            innovations,
            settings,
            best,
            phantom: PhantomData,
        }
    }

    fn speciate(&mut self) -> Vec<Vec<Organism>> {
        let mut species: Vec<Vec<Organism>> = vec![];

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

        species
    }

    fn kill(&mut self) {
        let mut species = self.speciate();

        self.population = vec![];

        for group in species.iter_mut() {
            if group.len() <= 1 {
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
        self.population.par_iter_mut().for_each(|mut org| {
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

    pub fn step(&mut self) -> (Network, f32) {
        self.execute();
        self.kill();
        self.generate();

        (
            Network::new(self.best.genome.clone()),
            self.best.fitness.unwrap(),
        )
    }
}
