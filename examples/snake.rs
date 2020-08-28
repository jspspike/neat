use neat::{Neat, Task};
use snake::{Snake, Direction};

struct SnakeTask {
    game: Snake,
    score: f32,
    finished: bool,
}

impl Task for SnakeTask {
    fn new(seed: u64) -> SnakeTask {
        SnakeTask {
            game: Snake::new(seed, 10),
            score: 0.0,
            finished: false,
        }
    }

    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), 1);
        self.finished = self.game.turn(match inputs[0] {
            v if v < 0.25 => Direction::Left,
            v if v < 0.5 => Direction::Up,
            v if v < 0.75 => Direction::Down,
            _ => Direction::Right
        });

        self.score += 0.01;

        let mut outputs = self.game.walls();
        outputs.append(&mut self.game.snake());
        outputs.append(&mut self.game.food());

        outputs
    }

    fn score(&self) -> Option<f32> {
        match self.finished {
            true => Some(self.score),
            false => None,
        }
    }
}

fn main() {

}

