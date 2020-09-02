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
        SnakeTask {
            game: Snake::new(0, 10),
            score: 0.0,
            running: true,
            repeat: 0,
        }
    }

    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), 1);

        let length = self.game.length();

        self.running = self.game.turn(match inputs[0] {
            v if v < 0.25 => Direction::Left,
            v if v < 0.5 => Direction::Up,
            v if v < 0.75 => Direction::Down,
            _ => Direction::Right,
        });

        if length != self.game.length() {
            assert_eq!(length, self.game.length() - 1);
            self.repeat = 0;
            self.score = self.game.length() as f32;
        }

        self.repeat += 1;
        if self.repeat == 100 {
            self.running = false;
        }

        self.score += 0.0001;

        let mut outputs = self.game.walls();
        outputs.append(&mut self.game.snake());
        outputs.append(&mut self.game.food());

        outputs
    }

    fn score(&self) -> Option<f32> {
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
            settings.weight_mutate = 4.0;
            settings.species_threshold = 1.4;
            settings.add_connection_rate = 0.44;
            settings.reset_fitness = false;
            settings.connections_diff = 1.05;

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
