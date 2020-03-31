use std::collections::HashMap;

pub(crate) struct InnovationCounter {
    count: u16,
    connections: HashMap<(u16, u16), u16>,
}

impl InnovationCounter {
    pub fn new(start: u16) -> InnovationCounter {
        InnovationCounter {
            count: start - 1,
            connections: HashMap::new(),
        }
    }

    pub fn add(&mut self, conn: (u16, u16)) -> u16 {
        if let Some(innovation) = self.connections.get(&conn) {
            return *innovation;
        }

        self.count += 1;
        self.connections.insert(conn, self.count);

        self.count
    }

    pub fn get(&self, conn: (u16, u16)) -> Option<u16> {
        if let Some(innovation) = self.connections.get(&conn) {
            Some(*innovation)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let mut test = InnovationCounter::new(4);

        assert_eq!(test.add((0, 3)), 4);
        assert_eq!(test.add((2, 3)), 5);
        assert_eq!(test.add((0, 3)), 4);
    }
}
