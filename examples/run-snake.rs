use neat::Network;
use snake::{Direction, RenderWindow, Snake, Style};

use std::{fs, thread, time};

fn load_network() -> Result<Network, bincode::Error> {
    let network_bytes = fs::read("examples/snake.network")?;

    bincode::deserialize(&network_bytes)
}

fn main() {
    let window = RenderWindow::new((1000, 1000), "Snake", Style::CLOSE, &Default::default());
    let mut game = Snake::new_display(0, 10, Some(window));

    let mut network = load_network().unwrap();
    network.reset();

    let mut direction = Direction::Center;

    while game.turn(direction) {
        let mut inputs = game.walls();
        inputs.append(&mut game.snake());
        inputs.append(&mut game.food());

        network.prop(inputs);

        let output = network.get_outputs();
        assert_eq!(output.len(), 1);

        println!("Network Output: {}", output[0]);
        thread::sleep(time::Duration::from_millis(100));

        direction = match output[0] {
            v if v < 0.25 => Direction::Left,
            v if v < 0.5 => Direction::Up,
            v if v < 0.75 => Direction::Down,
            _ => Direction::Right,
        };
    }
}
