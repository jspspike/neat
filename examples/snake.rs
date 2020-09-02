use neat::{Neat, NeatSettings, Task};
use snake::{Direction, Snake};
use std::fs;

struct SnakeTask {
    game: Snake,
    score: f32,
    running: bool,
    repeat: u8,
}

impl Task for SnakeTask {
    fn new(_: u64) -> SnakeTask {
        // Create new Task and do all initialization
        SnakeTask {
            game: Snake::new(0, 10),
            score: 0.0,
            running: true,
            repeat: 0,
        }
    }

    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        // Make sure inputs length matches Network outputs length
        assert_eq!(inputs.len(), 1);

        let length = self.game.length();

        // Determine snake direction from input
        self.running = self.game.turn(match inputs[0] {
            v if v < 0.25 => Direction::Left,
            v if v < 0.5 => Direction::Up,
            v if v < 0.75 => Direction::Down,
            _ => Direction::Right,
        });

        // Check if the length hasn't changed (ie. the snake ate food this turn)
        if length != self.game.length() {
            assert_eq!(length, self.game.length() - 1);
            self.repeat = 0;
            // If food eaten update score to reflect that
            self.score = self.game.length() as f32;
        }

        // Update repeat and exit out at 100 to prevent infinte loop
        self.repeat += 1;
        if self.repeat == 100 {
            self.running = false;
        }

        self.score += 0.0001;

        // Get output and return it
        let mut outputs = self.game.walls();
        outputs.append(&mut self.game.snake());
        outputs.append(&mut self.game.food());

        outputs
    }

    fn score(&self) -> Option<f32> {
        // Return Some(score) when task is finished
        match self.running {
            true => None,
            false => Some(self.score),
        }
    }
}

const NEAT_FILE: &str = "examples/snake.data";

fn neat_from_file() -> Result<Neat<SnakeTask>, bincode::Error> {
    let neat_bytes = fs::read(NEAT_FILE)?;

    bincode::deserialize(&neat_bytes)
}

fn main() {
    let mut neat = match neat_from_file() {
        Ok(neat) => neat,
        _ => {
            let mut settings = NeatSettings::default();
            settings.weight_mutate = 3.7;
            settings.species_threshold = 1.45;
            settings.add_connection_rate = 0.44;
            settings.reset_fitness = false;
            settings.connections_diff = 1.0;

            // Create Neat using Task implemented above
            Neat::<SnakeTask>::new(3000, 24, 1, settings)
        }
    };

    let mut best = neat.step();

    for i in 0..100000 {
        let (network, fitness) = best;

        println!(
            "Count: {} Species Count: {} Fitness: {}",
            i,
            neat.species(),
            fitness
        );
        if i % 100 == 0 {
            let neat_bytes = bincode::serialize(&neat).unwrap();
            fs::write(NEAT_FILE, neat_bytes).unwrap();

            let network_bytes = bincode::serialize(&network).unwrap();
            fs::write("examples/snake.network", &network_bytes).unwrap();
            dbg!("Wrote to files");
        }

        // Run step of Neat returning tuple of network and fitness of most fit genome
        best = neat.step();
    }

    let neat_bytes = bincode::serialize(&neat).unwrap();
    fs::write(NEAT_FILE, neat_bytes).unwrap();

    let (network, fitness) = best;
    dbg!(fitness);
    dbg!(&network);

    let network_bytes = bincode::serialize(&network).unwrap();
    fs::write("examples/snake.network", &network_bytes).unwrap();
}
